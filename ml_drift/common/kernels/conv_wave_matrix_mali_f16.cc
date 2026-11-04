// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift/common/kernels/conv_wave_matrix_mali_f16.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
inline bool IsConvKernelXis1(const Convolution2DAttributes& conv_attr) {
  return conv_attr.GetWeightsShape().w == 1 && conv_attr.dilations.w == 1 &&
         conv_attr.strides.w == 1 && conv_attr.padding.prepended.w == 0 &&
         conv_attr.padding.appended.w == 0;
}

inline bool IsConvKernelYis1(const Convolution2DAttributes& conv_attr) {
  return conv_attr.GetWeightsShape().h == 1 && conv_attr.dilations.h == 1 &&
         conv_attr.strides.h == 1 && conv_attr.padding.prepended.h == 0 &&
         conv_attr.padding.appended.h == 0;
}

std::string GetCode(const GpuInfo& gpu_info,
                    const ConvWaveMatrixMaliF16::ConvParams& conv_params,
                    const ConvWaveMatrixMaliF16::KernelParams& kernel_params,
                    const TensorDescriptor& src_desc) {
  auto GetGlobalID = [&](int id) {
    int3 wg_order = kernel_params.slices_first ? int3(1, 0, 2) : int3(0, 1, 2);
    int3 launch_remap;
    launch_remap[wg_order.x] = 0;
    launch_remap[wg_order.y] = 1;
    launch_remap[wg_order.z] = 2;
    std::string result;
    const std::string sid = std::to_string(id);
    if (wg_order[id] == id) {
      return "ucl::GetGlobalId<" + sid + ">()";
    } else {
      return "(ucl::GetGroupId<" + std::to_string(launch_remap[id]) +
             ">() * ucl::GetGroupSize<" + sid + ">() + ucl::GetLocalId<" + sid +
             ">())";
    }
  };
  std::string c = R"(
half4 arm_simd16_mmul8x4x8(half2 a, half2 b, half4 acc) {
  acc.xy = arm_matrix_multiply_af0(a, b, acc.xy);
  acc.zw = arm_matrix_multiply_af1(a, b, acc.zw);
  return acc;
}

MAIN_FUNCTION($0) {
  // WG must be [M*16, N, K]
)";
  c += "  int subgroup_id = " + GetGlobalID(0) + " / 16;\n";
  c += "  int subgroup_local_id = " + GetGlobalID(0) + " % 16;\n";
  c += "  int slices_sub_id = subgroup_local_id % 2;\n";
  c += "  int x_sub_id = subgroup_local_id / 2;\n";
  c += "  int subgroup_base_x = subgroup_id * " +
       std::to_string(kernel_params.x_block_size * 8) + ";\n";
  c += "  int subgroup_dst_s_base = " + GetGlobalID(1) + " * " +
       std::to_string(kernel_params.s_block_size * 4) + ";\n";
  c += "  int dst_y = " + GetGlobalID(2) + ";\n";
  c += R"(
  if (subgroup_base_x >= args.dst.Width() || dst_y >= args.dst.Height() || subgroup_dst_s_base >= args.dst.Slices()) {
    return;
  }
)";
  std::string weights_offset = "subgroup_dst_s_base";
  if (conv_params.different_weights_for_height) {
    // subgroup_dst_s_base is already pre-multiplied by
    // kernel_params.s_block_size * 4
    weights_offset = "(subgroup_dst_s_base * args.src.Height() + dst_y * " +
                     std::to_string(kernel_params.s_block_size * 4) + ")";
  }
  weights_offset = weights_offset + " * 4 * args.src.Slices()";
  if (!conv_params.x_kernel_is_1) {
    weights_offset += " * args.kernel_size_x";
  }
  if (!conv_params.y_kernel_is_1) {
    weights_offset += " * args.kernel_size_y";
  }
  c += "  __global half8* w_base = (__global half8*)(args.weights.GetPtr() + " +
       weights_offset + ");\n";
  for (int s = 0; s < kernel_params.s_block_size; s++) {
    for (int x = 0; x < kernel_params.x_block_size; x++) {
      std::string patch = R"(
  half4 r_w$0_s$1 = (half4)(0.0f);
  half4 r_w$0_s$2 = (half4)(0.0f);
)";
      c += absl::Substitute(patch, x, s * 2 + 0, s * 2 + 1);
    }
  }
  if (!conv_params.y_kernel_is_1) {
    c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
    c += "  int src_y = ky * args.dilation_y + dst_y * args.stride_y + "
         "args.padding_y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::kHeight, gpu_info)) {
      c += "  bool in_y = src_y >= 0 && src_y < args.src.Height();\n";
      if (!src_desc.CanReadOutOfBorder(Axis::kHeight, gpu_info)) {
        c += "  src_y = clamp(src_y, 0, args.src.Height() - 1);\n";
      }
    }
  } else {
    c += "  int src_y = min(dst_y, args.src.Height() - 1);\n";
  }
  if (!conv_params.x_kernel_is_1) {
    c += "  for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
    for (int x = 0; x < kernel_params.x_block_size; x++) {
      const std::string src_x = "src_x" + std::to_string(x);
      c += "  int " + src_x +
           " = kx * args.dilation_x + (subgroup_base_x + x_sub_id + " +
           std::to_string(x * 8) + ") * args.stride_x + args.padding_x;\n";
      if (!src_desc.SupportsZeroClamp(Axis::kWidth, gpu_info)) {
        c += "  bool in_x" + std::to_string(x) + " = " + src_x + " >= 0 && " +
             src_x + " < args.src.Width();\n";
        if (!src_desc.CanReadOutOfBorder(Axis::kWidth, gpu_info)) {
          c += "  " + src_x + " = clamp(" + src_x +
               ", 0, args.src.Width() - 1);\n";
        }
      }
    }
  } else {
    for (int x = 0; x < kernel_params.x_block_size; x++) {
      std::string patch =
          "  int src_x$0 = min(subgroup_base_x + x_sub_id + $1, "
          "args.src.Width() - 1);\n";
      c += absl::Substitute(patch, x, x * 8);
    }
  }
  c += "  for (int src_s = 0; src_s < args.src.Slices(); src_s += 2) {\n";
  const int src_slices = DivideRoundUp(conv_params.weights_shape.i, 4);
  std::string src_slice = "src_s + slices_sub_id";
  if (src_slices % 2 != 0) {
    src_slice = "min(" + src_slice + ", args.src.Slices() - 1)";
  }
  for (int x = 0; x < kernel_params.x_block_size; x++) {
    std::string patch;
    patch += "    half4 s$0 = args.src.Read(src_x$0, src_y, $1);\n";
    std::string check;
    if (!conv_params.x_kernel_is_1 &&
        !src_desc.SupportsZeroClamp(Axis::kWidth, gpu_info)) {
      check += "in_x" + std::to_string(x);
    }
    if (!conv_params.y_kernel_is_1 &&
        !src_desc.SupportsZeroClamp(Axis::kHeight, gpu_info)) {
      if (!check.empty()) {
        check += " && ";
      }
      check += "in_y";
    }
    if (!check.empty()) {
      patch += "    s$0 *= ucl::Convert<half>(" + check + ");\n";
    }
    c += absl::Substitute(patch, x, src_slice);
  }
  for (int s = 0; s < kernel_params.s_block_size; s++) {
    c += R"(
    {
    half8 W = w_base[subgroup_local_id];
    w_base += 16;
)";
    for (int x = 0; x < kernel_params.x_block_size; x++) {
      std::string patch = R"(
    r_w$0_s$1 = arm_simd16_mmul8x4x8(s$0.xy, W.s01, r_w$0_s$1);
    r_w$0_s$1 = arm_simd16_mmul8x4x8(s$0.zw, W.s23, r_w$0_s$1);
    r_w$0_s$2 = arm_simd16_mmul8x4x8(s$0.xy, W.s45, r_w$0_s$2);
    r_w$0_s$2 = arm_simd16_mmul8x4x8(s$0.zw, W.s67, r_w$0_s$2);
)";
      c += absl::Substitute(patch, x, s * 2 + 0, s * 2 + 1);
    }
    c += "    }\n";
  }
  c += "  }\n";
  if (!conv_params.x_kernel_is_1) {
    c += "  }\n";
  }
  if (!conv_params.y_kernel_is_1) {
    c += "  }\n";
  }
  c += R"(
  int DST_S = subgroup_dst_s_base + subgroup_local_id % 4;
  int dst_s_final = 0;
)";
  c += "  int DST_X = subgroup_base_x + x_sub_id / 2 * 2;\n";
  for (int s = 0; s < kernel_params.s_block_size; s++) {
    c += "  dst_s_final = DST_S + " + std::to_string(s * 4) + ";\n";
    c += "  if (dst_s_final < args.dst.Slices()) {\n";
    if (conv_params.has_bias) {
      c += "    half4 bias_val = args.biases.Read(dst_s_final);\n";
    }
    c += "    int dst_x_final;\n";
    for (int x = 0; x < kernel_params.x_block_size; x++) {
      std::string patch = R"(
    dst_x_final = DST_X + $0 * 8 + 0;
    if (dst_x_final < args.dst.Width()) {
      half4 R = (half4)(r_w$0_s$1.x, r_w$0_s$1.y, r_w$0_s$2.x, r_w$0_s$2.y);
)";
      if (conv_params.has_bias) {
        patch += "      R += bias_val;\n";
      }
      patch += R"(      args.dst.Write(R, dst_x_final, dst_y, dst_s_final);
    }
    dst_x_final = DST_X + $0 * 8 + 1;
    if (dst_x_final < args.dst.Width()) {
      half4 R = (half4)(r_w$0_s$1.z, r_w$0_s$1.w, r_w$0_s$2.z, r_w$0_s$2.w);
)";
      if (conv_params.has_bias) {
        patch += "      R += bias_val;\n";
      }
      patch += R"(      args.dst.Write(R, dst_x_final, dst_y, dst_s_final);
    }
)";
      c += absl::Substitute(patch, x, s * 2 + 0, s * 2 + 1);
    }
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

void AddXKernelParams(const Convolution2DAttributes& attr, GPUOperation* op) {
  const OHWI& weights_shape = attr.GetWeightsShape();
  op->args_.AddInt("stride_x", attr.strides.w);
  op->args_.AddInt("padding_x", -attr.padding.prepended.w);
  op->args_.AddInt("kernel_size_x", weights_shape.w);
  op->args_.AddInt("dilation_x", attr.dilations.w);
}

void AddYKernelParams(const Convolution2DAttributes& attr, GPUOperation* op) {
  const OHWI& weights_shape = attr.GetWeightsShape();
  op->args_.AddInt("stride_y", attr.strides.h);
  op->args_.AddInt("padding_y", -attr.padding.prepended.h);
  op->args_.AddInt("kernel_size_y", weights_shape.h);
  op->args_.AddInt("dilation_y", attr.dilations.h);
}

}  // namespace

ConvWaveMatrixMaliF16::ConvWaveMatrixMaliF16(const GpuInfo& gpu_info,
                                             const ConvParams& conv_params,
                                             const TensorDescriptor& src_desc,
                                             const TensorDescriptor& dst_desc)
    : conv_params_(conv_params) {
  kernel_params_.x_block_size = 2;
  kernel_params_.s_block_size = 2;
  kernel_params_.slices_first = false;
  work_group_size_ = int3(16, 1, 1);
  code_ = GetCode(gpu_info, conv_params_, kernel_params_, src_desc);
  work_group_launch_order_ =
      kernel_params_.slices_first ? int3(1, 0, 2) : int3(0, 1, 2);
  compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);

  AddSrcTensor("src", src_desc);
  AddDstTensor("dst", dst_desc);
}

std::vector<int3> ConvWaveMatrixMaliF16::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  switch (tuning_type) {
    case TuningType::kExhaustive:
      return GetWorkGroupsXMultipleOf(/*wave_size=*/16, gpu_info, kernel_info,
                                      grid_size_);
    case TuningType::kFast:
    default:
      return {{32, 2, 1}};
  }
}

int3 ConvWaveMatrixMaliF16::GetGridSize() const {
  int x_tiles =
      DivideRoundUp(dst_[0]->Width(), 8 * kernel_params_.x_block_size);
  const int grid_x = x_tiles * /*wave_size=*/16;
  const int grid_y =
      DivideRoundUp(dst_[0]->Slices(), 4 * kernel_params_.s_block_size);
  const int grid_z = dst_[0]->Height();
  return int3(grid_x, grid_y, grid_z);
}

void ConvWaveMatrixMaliF16::AddWeights(
    const Tensor<OHWI, DataType::kFloat32>& weights) {
  const WeightsDescription weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kFloat16;
  buffer_desc.element_size = 4;
  buffer_desc.memory_type = MemoryType::kGlobal;
  buffer_desc.size = elements_count * SizeOf(weights_desc.type);
  buffer_desc.data.resize(buffer_desc.size);
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(buffer_desc.data));
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

bool SupportsConvWaveMatrixMaliF16(const GpuInfo& gpu_info) {
  if (!gpu_info.SupportsExtension("cl_arm_matrix_multiply")) {
    return false;
  }
  if (!gpu_info.mali_info.IsMaliG2()) {
    return false;
  }
  return true;
}

ConvWaveMatrixMaliF16 CreateConvWaveMatrixMaliF16(
    const GpuInfo& gpu_info, const TensorDescriptor& src_desc,
    const TensorDescriptor& dst_desc, const Convolution2DAttributes& attr) {
  ConvWaveMatrixMaliF16::ConvParams conv_params;
  conv_params.weights_shape = attr.GetWeightsShape();
  conv_params.x_kernel_is_1 = IsConvKernelXis1(attr);
  conv_params.y_kernel_is_1 = IsConvKernelYis1(attr);
  conv_params.has_bias = !attr.bias.empty();
  ConvWaveMatrixMaliF16 result =
      ConvWaveMatrixMaliF16(gpu_info, conv_params, src_desc, dst_desc);
  result.AddWeights(GetFloatWeights(attr));
  if (conv_params.has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, DataType::kFloat16, attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  if (!conv_params.x_kernel_is_1) {
    AddXKernelParams(attr, &result);
  }
  if (!conv_params.y_kernel_is_1) {
    AddYKernelParams(attr, &result);
  }
  return result;
}

ConvWaveMatrixMaliF16 CreateConvWaveMatrixMaliF16ExternalWeights(
    const GpuInfo& gpu_info, const TensorDescriptor& src_desc,
    const TensorDescriptor& dst_desc, const Convolution2DAttributes& attr,
    const TensorDescriptor* bias, bool different_weights_for_height) {
  ConvWaveMatrixMaliF16::ConvParams conv_params;
  conv_params.weights_shape = attr.GetWeightsShape();
  conv_params.x_kernel_is_1 = IsConvKernelXis1(attr);
  conv_params.y_kernel_is_1 = IsConvKernelYis1(attr);
  conv_params.different_weights_for_height = different_weights_for_height;
  if (different_weights_for_height) {
    conv_params.y_kernel_is_1 = true;
  }
  conv_params.has_bias = bias != nullptr;
  ConvWaveMatrixMaliF16 result =
      ConvWaveMatrixMaliF16(gpu_info, conv_params, src_desc, dst_desc);
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kFloat16;
  buffer_desc.element_size = 4;
  buffer_desc.memory_type = MemoryType::kGlobal;
  result.AddSrcBuffer("weights", buffer_desc);
  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }
  if (!conv_params.x_kernel_is_1) {
    AddXKernelParams(attr, &result);
  }
  if (!conv_params.y_kernel_is_1) {
    AddYKernelParams(attr, &result);
  }
  return result;
}

}  // namespace ml_drift
