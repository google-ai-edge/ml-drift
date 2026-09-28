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

#include "ml_drift/common/kernels/depthwise_conv_tiled.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
template <typename T>
void RearrangeWeightsForDWConv2D(
    const Tensor<OHWI, DataType::kFloat32>& weights, absl::Span<T> dst) {
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_depth = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        T filter_val;
        for (int i = 0; i < 4; ++i) {
          const int d_ch = d * 4 + i;
          if (d_ch < dst_channels) {
            const int f_index = weights.shape.LinearIndex(
                {d_ch % weights.shape.o, y, x, d_ch / weights.shape.o});
            filter_val[i] = weights.data[f_index];
          } else {
            filter_val[i] = 0.0f;
          }
        }
        dst[counter++] = filter_val;
      }
    }
  }
}

template void RearrangeWeightsForDWConv2D(
    const Tensor<OHWI, DataType::kFloat32>& weights, absl::Span<float4> dst);
template void RearrangeWeightsForDWConv2D(
    const Tensor<OHWI, DataType::kFloat32>& weights, absl::Span<half4> dst);

bool UseBuffersForWeights(const GpuInfo& gpu_info) {
  if (gpu_info.IsApple() &&
      gpu_info.apple_info.IsFamilyOrLower(AppleInfo::Family::kApple2)) {
    return false;
  }
  return !gpu_info.SupportsImages() || gpu_info.IsMali() ||
         gpu_info.IsBroadcom() || gpu_info.IsApple() || gpu_info.IsAMD();
}

void PrepareObjects(const GpuInfo& gpu_info,
                    const DepthwiseConvolution2DAttributes& attr,
                    DataType dst_type, Arguments* args) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  const int dst_channels = weights_shape.i * weights_shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights_shape.w;
  const int kernel_y = weights_shape.h;

  const int elements_count = kernel_x * kernel_y * dst_slices;

  std::vector<uint8_t> data(SizeOf(dst_type) * 4 * elements_count);

  if (dst_type == DataType::kFloat32) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    RearrangeWeightsForDWConv2D(GetFloatWeights(attr),
                                absl::MakeSpan(ptr, elements_count));
  } else if (dst_type == DataType::kFloat16) {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    RearrangeWeightsForDWConv2D(GetFloatWeights(attr),
                                absl::MakeSpan(ptr, elements_count));
  }

  if (UseBuffersForWeights(gpu_info)) {
    BufferDescriptor desc;
    desc.element_type = dst_type;
    desc.element_size = 4;
    desc.size = SizeOf(dst_type) * 4 * elements_count;
    desc.data = std::move(data);
    args->AddObject("weights", std::make_unique<BufferDescriptor>(desc));
  } else {
    TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
        dst_type, TensorStorageType::kTexture2D, kernel_x * kernel_y,
        dst_slices, data.data());
    args->AddObject("weights", std::make_unique<TensorDescriptor>(desc));
  }

  if (!attr.bias.data.empty()) {
    TensorDescriptor bias_tensor_desc =
        CreateConstantLinearTensorDescriptor(gpu_info, dst_type, attr.bias);
    args->AddObject("biases", std::make_unique<TensorDescriptor>(
                                  std::move(bias_tensor_desc)));
  }
}
}  // namespace

bool IsDepthwiseConvTiledSupported(
    const DepthwiseConvolution2DAttributes& attr) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  if (weights_shape.o != 1) {
    return false;
  }
  if (attr.dilations.h == 2 && attr.strides.h == 1 &&
      (weights_shape.h == 3 || weights_shape.h == 5)) {
    return true;
  }
  if (attr.dilations.h == 1 && (attr.strides.h == 1 || attr.strides.h == 2) &&
      (weights_shape.h == 3 || weights_shape.h == 5)) {
    return true;
  }
  if (attr.dilations.h == 1 && attr.strides.h == 3 && weights_shape.h == 7) {
    return true;
  }
  return false;
}

class DepthwiseConvTiled : public GPUOperation {
 public:
  DepthwiseConvTiled() = default;
  int3 GetGridSize() const override {
    return int3(dst_[0]->Width() * dst_[0]->Batch(),
                DivideRoundUp(dst_[0]->Height(), y_tile_size_),
                dst_[0]->Slices());
  }
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    if (tuning_type == TuningType::kFast) {
      return {work_group_size_};
    } else {
      return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                   grid_size_);
    }
  }
  int y_tile_size_;
};

std::unique_ptr<GPUOperation> CreateDepthwiseConvTiled(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr) {
  DepthwiseConvTiled op;
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  const int dst_slices = DivideRoundUp(weights_shape.i, 4);
  if (gpu_info.IsPowerVR() &&
      gpu_info.powervr_info.gpu_version >= PowerVRGpu::kAXE &&
      dst_slices % 8 == 0) {
    op.work_group_size_ = int3(16, 1, 8);
  } else if (!gpu_info.IsMali() && !gpu_info.IsIntel() && dst_slices % 4 == 0) {
    op.work_group_size_ = int3(16, 1, 4);
  } else if (dst_slices % 2 == 0) {
    op.work_group_size_ = int3(16, 1, 2);
  } else {
    op.work_group_size_ = int3(16, 1, 1);
  }
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.args_.AddInt("stride_w", attr.strides.w);
  op.args_.AddInt("stride_h", attr.strides.h);
  op.args_.AddInt("padding_w", -attr.padding.prepended.w);
  op.args_.AddInt("padding_h", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_w", attr.dilations.w);
  op.args_.AddInt("kernel_w", weights_shape.w);
  op.args_.AddInt("kernel_h", weights_shape.h);
  PrepareObjects(gpu_info, attr, definition.dst_tensors[0].GetDataType(),
                 &op.args_);
  int y_tile_size = 4;
  if (weights_shape.h == 5) {
    y_tile_size = 3;
  }
  if (weights_shape.h == 3) {
    y_tile_size = 4;
  }
  if (attr.strides.h >= 2) {
    y_tile_size = 2;
  }
  if (gpu_info.IsMali() &&
      gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV1) {
    y_tile_size = 3;
  }
  std::string c;
  if (gpu_info.IsApiOpenCl() && gpu_info.IsPowerVR()) {
    c += "__attribute__((reqd_work_group_size(" +
         std::to_string(op.work_group_size_.x) + ", " +
         std::to_string(op.work_group_size_.y) + ", " +
         std::to_string(op.work_group_size_.z) + ")))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  if (definition.src_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int dst_x = linear_id / args.dst_tensor.Batch();\n";
    c += "  int dst_b = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(dst_b);\n";
    c += "  args.dst_tensor.SetBatchRef(dst_b);\n";
  } else {
    c += "  int dst_x = ucl::GetGlobalId<0>();\n";
  }
  c += "  int dst_y = ucl::GetGlobalId<1>() * " + std::to_string(y_tile_size) +
       ";\n";
  c += "  int dst_s = ucl::GetGlobalId<2>();\n";
  c += "  if (dst_x >= args.dst_tensor.Width() || dst_y >= "
       "args.dst_tensor.Height() || dst_s >= args.dst_tensor.Slices()) {\n";
  c += "    return;\n";
  c += "  }\n";
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  if (weights_are_buffer) {
    c += "  int weights_offset = dst_s * args.kernel_w * args.kernel_h;\n";
  }
  c += "  int x_src = dst_x * args.stride_w + args.padding_w;\n";
  for (int block = 0; block < y_tile_size; ++block) {
    c += "  AccType r" + std::to_string(block) +
         " = ucl::Init<AccType>(0.0f);\n";
  }
  const int h_kernel_size = (weights_shape.h - 1) * attr.dilations.h + 1;
  const int read_block_size =
      h_kernel_size + (y_tile_size - 1) * attr.strides.h;
  for (int id = 0; id < read_block_size; ++id) {
    const std::string y = "y" + std::to_string(id);
    c += "  int " + y + " = dst_y * args.stride_h + args.padding_h + " +
         std::to_string(id) + ";\n";
    if (!definition.src_tensors[0].SupportsZeroClamp(Axis::kHeight, gpu_info)) {
      c += "  bool " + y + "_in = " + y + " >= 0 && " + y +
           " < args.src_tensor.Height();\n";
      c += "  " + y + " = clamp(" + y + ", 0, args.src_tensor.Height() - 1);\n";
    }
  }

  c += "  for (int kx = 0; kx < args.kernel_w; kx += 1) {\n";
  c += "    int x = x_src + kx * args.dilation_w;\n";
  if (!definition.src_tensors[0].SupportsZeroClamp(Axis::kWidth, gpu_info)) {
    c += "    bool x_in = x >= 0 && x < args.src_tensor.Width();\n";
    c += "    x = clamp(x, 0, args.src_tensor.Width() - 1);\n";
  }
  for (int ky = 0; ky < weights_shape.h; ++ky) {
    if (weights_are_buffer) {
      c += "    Type f" + std::to_string(ky) +
           " = args.weights.Read(weights_offset + kx + args.kernel_w * " +
           std::to_string(ky) + ");\n";
    } else {
      c += "    Type f" + std::to_string(ky) +
           " = args.weights.Read(kx + args.kernel_w * " + std::to_string(ky) +
           ", dst_s);\n";
    }
  }
  for (int id = 0; id < read_block_size; ++id) {
    const std::string s = "s" + std::to_string(id);
    const std::string y = "y" + std::to_string(id);
    std::string check;
    if (!definition.src_tensors[0].SupportsZeroClamp(Axis::kWidth, gpu_info)) {
      check += "x_in";
    }
    if (!definition.src_tensors[0].SupportsZeroClamp(Axis::kHeight, gpu_info)) {
      const std::string y_in = y + "_in";
      check += check.empty() ? y_in : (" && " + y_in);
    }
    const std::string multiplier =
        check.empty() ? "" : " * ucl::Convert<SType>(" + check + ")";
    c += "    Type " + s + " = args.src_tensor.Read(x, " + y + ", dst_s)" +
         multiplier + ";\n";
  }
  for (int block = 0; block < y_tile_size; ++block) {
    const std::string r = "r" + std::to_string(block);
    for (int ky = 0; ky < weights_shape.h; ++ky) {
      const std::string s =
          "s" + std::to_string(block * attr.strides.h + ky * attr.dilations.h);
      const std::string f = "f" + std::to_string(ky);
      c += "    " + r + " += ucl::Convert<AccType>(" + s + " * " + f + ");\n";
    }
  }
  c += "  }\n";

  if (attr.bias.data.empty()) {
    c += "  Type bias_val = ucl::Init<Type>(0.0f);\n";
  } else {
    c += "  Type bias_val = args.biases.Read(dst_s);\n";
  }
  for (int block = 0; block < y_tile_size; ++block) {
    const std::string r = "r" + std::to_string(block);
    const std::string dst_y = "dst_y + " + std::to_string(block);
    c += "  if (" + dst_y + " < args.dst_tensor.Height()) {\n";
    c += "    Type result = ucl::Convert<Type>(" + r + ");\n";
    c += "    result += bias_val;\n";
    c += "    args.dst_tensor.Write(result, dst_x, " + dst_y + ", dst_s);\n";
    c += "  }\n";
  }
  c += "}\n";

  const DataType acc_type = precision == CalculationsPrecision::kF16
                                ? DataType::kFloat16
                                : DataType::kFloat32;
  const DataType type = definition.src_tensors[0].GetDataType();
  absl::StrReplaceAll({{"SType", ToUclDataType(type, 1)},
                       {"Type", ToUclDataType(type, 4)},
                       {"AccType", ToUclDataType(acc_type, 4)}},
                      &c);

  op.y_tile_size_ = y_tile_size;
  op.code_ = std::move(c);
  return std::make_unique<DepthwiseConvTiled>(std::move(op));
}

}  // namespace ml_drift
