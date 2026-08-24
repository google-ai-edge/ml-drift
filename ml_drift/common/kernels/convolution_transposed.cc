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

#include "ml_drift/common/kernels/convolution_transposed.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
bool UseBufferForWeights(const GpuInfo& gpu_info) {
  return !gpu_info.SupportsImages() || gpu_info.IsMali() ||
         gpu_info.IsBroadcom() || gpu_info.IsApple() || gpu_info.IsAMD();
}

WeightsLayout GetLayout(const GpuInfo& gpu_info) {
  if (UseBufferForWeights(gpu_info)) {
    if (gpu_info.IsDotPreferred()) {
      return WeightsLayout::kOSpatialIOGroupO4I4;
    } else {
      return WeightsLayout::kOSpatialIOGroupI4O4;
    }
  } else {
    if (gpu_info.IsDotPreferred()) {
      return WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
    } else {
      return WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }
  }
}
}  // namespace

ConvolutionTransposed::ConvolutionTransposed(
    const OperationDef& definition, CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr, const GpuInfo& gpu_info)
    : stride_(attr.stride.w, attr.stride.h, 1, 1), block_size_(2, 2, 1, 2) {
  weights_layout_ = GetLayout(gpu_info);
  weights_data_type_ = DeduceDataTypeFromPrecision(precision);
  const bool is_f16 = precision == CalculationsPrecision::F16;
  if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.IsMidgard()) {
      block_size_ = is_f16 ? int4(2, 1, 1, 2) : int4(2, 1, 1, 1);
    } else {
      block_size_ = is_f16 ? int4(2, 2, 1, 2) : int4(2, 2, 1, 1);
    }
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  if (dst_depth == 1 || dst_depth == 3) {
    if (!gpu_info.IsMali()) {
      block_size_.y *= block_size_.w;
    }
    block_size_.w = 1;
  }

  args_.AddInt("stride_x", stride_.x);
  args_.AddInt("stride_y", stride_.y);
  args_.AddInt("padding_x", attr.padding.prepended.w);
  args_.AddInt("padding_y", attr.padding.prepended.h);
  args_.AddInt("kernel_size_x", attr.weights.shape.w);
  args_.AddInt("kernel_size_y", attr.weights.shape.h);
  args_.AddInt("grid_size_y");
  code_ = GenerateConvolutionTransposedCode(
      definition, precision, gpu_info, block_size_, !attr.bias.data.empty());
}

ConvolutionTransposed::ConvolutionTransposed(
    const OperationDef& definition, CalculationsPrecision precision,
    const ConvolutionTransposed3DAttributes& attr, const GpuInfo& gpu_info)
    : stride_(attr.stride.w, attr.stride.h, attr.stride.d, 1),
      block_size_(2, 2, 1, 2) {
  weights_layout_ = GetLayout(gpu_info);
  weights_data_type_ = DeduceDataTypeFromPrecision(precision);
  const bool is_f16 = precision == CalculationsPrecision::F16;
  if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.IsMidgard()) {
      block_size_ = is_f16 ? int4(2, 1, 1, 2) : int4(2, 1, 1, 1);
    } else {
      block_size_ = is_f16 ? int4(2, 2, 1, 2) : int4(2, 2, 1, 1);
    }
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  if (dst_depth == 1 || dst_depth == 3) {
    if (!gpu_info.IsMali()) {
      block_size_.y *= block_size_.w;
    }
    block_size_.w = 1;
  }

  args_.AddInt("stride_x", stride_.x);
  args_.AddInt("stride_y", stride_.y);
  args_.AddInt("stride_z", stride_.z);
  args_.AddInt("padding_x", attr.padding.prepended.w);
  args_.AddInt("padding_y", attr.padding.prepended.h);
  args_.AddInt("padding_z", attr.padding.prepended.d);
  args_.AddInt("kernel_size_x", attr.weights.shape.w);
  args_.AddInt("kernel_size_y", attr.weights.shape.h);
  args_.AddInt("kernel_size_z", attr.weights.shape.d);
  args_.AddInt("grid_size_y");
  code_ = GenerateConvolutionTransposedCode(
      definition, precision, gpu_info, block_size_, !attr.bias.data.empty());
}

std::string ConvolutionTransposed::GenerateConvolutionTransposedCode(
    const OperationDef& op_def, CalculationsPrecision precision,
    const GpuInfo& gpu_info, const int4& block_size, bool has_bias) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  if (op_def.src_tensors.size() != 1) {
    // dynamic weights
    if (weights_layout_ == WeightsLayout::kOSpatialIOGroupI4O4 ||
        weights_layout_ == WeightsLayout::kOSpatialIOGroupO4I4) {
      BufferDescriptor desc;
      desc.element_type = op_def.src_tensors[1].GetDataType();
      desc.element_size = 16;
      desc.memory_type = MemoryType::GLOBAL;
      AddSrcBuffer("weights", desc);
    } else {
      for (int i = 0; i < 4; ++i) {
        const std::string name = "weights" + std::to_string(i);
        AddSrcTensor(name, op_def.src_tensors[1 + i]);
      }
    }
  }

  const auto& src_def = op_def.src_tensors[0];

  std::string c;

  const bool weights_are_buffer = UseBufferForWeights(gpu_info);
  bool use_fma = gpu_info.IsAMD() && gpu_info.IsApiOpenCl();
  if (GetWeightsDescription().IsI4O4()) {
    switch (precision) {
      case CalculationsPrecision::F32:
      case CalculationsPrecision::F16:
        if (use_fma) {
          c += "#define CONV(R, S)    \\\n";
          c += "R = fma(w0, S.x, R); \\\n";
          c += "R = fma(w1, S.y, R); \\\n";
          c += "R = fma(w2, S.z, R); \\\n";
          c += "R = fma(w3, S.w, R);   \n";
        } else {
          c += "#define CONV(R, S)    \\\n";
          c += "R += S.x * w0; \\\n";
          c += "R += S.y * w1; \\\n";
          c += "R += S.z * w2; \\\n";
          c += "R += S.w * w3;   \n";
        }
        break;
      case CalculationsPrecision::F32_F16:
        c += "#define CONV(R, S) \\\n";
        c += "R += ucl::Convert<AccType>(S.x * w0 + S.y * w1 + S.z * w2 + S.w "
             "* w3);\n";
        break;
    }
  } else {
    // O4I4
    c += "#define CONV(R, S)    \\\n";
    c += "R.x += dot(S, w0); \\\n";
    c += "R.y += dot(S, w1); \\\n";
    c += "R.z += dot(S, w2); \\\n";
    c += "R.w += dot(S, w3);   \n";
  }

  auto generate_id = [&](const std::string& x, const std::string& y,
                         const std::string& z) {
    std::string id;
    if (src_def.HasAxis(Axis::WIDTH)) {
      id += "_w" + x;
    }
    if (src_def.HasAxis(Axis::HEIGHT)) {
      id += "_h" + y;
    }
    if (src_def.HasAxis(Axis::DEPTH)) {
      id += "_d" + z;
    }
    return id;
  };

  auto generate_id_full = [&](const std::string& x, const std::string& y,
                              const std::string& z, const std::string& s) {
    return generate_id(x, y, z) + "_s" + s;
  };

  auto generate_check = [&](const std::string& x, const std::string& y,
                            const std::string& z) {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"in_x", "in_y", "in_z"};
    const std::vector<std::string> coords{x, y, z};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_def.HasAxis(axis) && !src_def.SupportsZeroClamp(axis, gpu_info) &&
          block_size[i] != 1) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i] + coords[i];
      }
    }
    return check;
  };

  switch (precision) {
    case CalculationsPrecision::F32:
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT16 half16\n";
      break;
  }

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int dst_x = (linear_id / args.dst_tensor.Batch());\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int dst_x = ucl::GetGlobalId<0>();\n";
  }
  c += "  int rem_x = dst_x % args.stride_x;\n";
  c += "  int ceil_x = dst_x / args.stride_x;\n";
  c += "  dst_x = ceil_x * args.stride_x * " + std::to_string(block_size.x) +
       " + rem_x;\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_y = ucl::GetGlobalId<1>();\n";
    c += "  int dst_y = linear_id_y % args.grid_size_y;\n";
    c += "  int dst_z = linear_id_y / args.grid_size_y;\n";
    c += "  int rem_z = dst_z % args.stride_z;\n";
    c += "  int ceil_z = dst_z / args.stride_z;\n";
    c += "  dst_z = ceil_z * args.stride_z * " + std::to_string(block_size.z) +
         " + rem_z;\n";
    c += "  if (dst_z >= args.dst_tensor.Depth()) return;\n";
  } else {
    c += "  int dst_y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int rem_y = dst_y % args.stride_y;\n";
  c += "  int ceil_y = dst_y / args.stride_y;\n";
  c += "  dst_y = ceil_y * args.stride_y * " + std::to_string(block_size.y) +
       " + rem_y;\n";
  c += "  int dst_s = ucl::GetGlobalId<2>() * " + std::to_string(block_size.w) +
       ";\n";
  c += "  if (dst_x >= args.dst_tensor.Width() || dst_y >= "
       "args.dst_tensor.Height() || dst_s >= "
       "args.dst_tensor.Slices()) return;\n";
  if (weights_are_buffer) {
    c += "  int f_base = dst_s * args.src_tensor.Slices() * args.kernel_size_x "
         "* args.kernel_size_y";
    if (src_def.HasAxis(Axis::DEPTH)) {
      c += " * args.kernel_size_z";
    }
    c += ";\n";
  }
  for (int s = 0; s < block_size.w; ++s) {
    const std::string sind = std::to_string(s);
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          c += "  AccType r" + generate_id_full(xind, yind, zind, sind) +
               " = ucl::Init<AccType>(0.0f);\n";
        }
      }
    }
  }
  c += "  int kernel_first_dst_x = dst_x + args.padding_x;\n";
  c += "  int kernel_first_dst_y = dst_y + args.padding_y;\n";
  c += "  int kernel_last_dst_x = kernel_first_dst_x - args.kernel_size_x;\n";
  c += "  int kernel_last_dst_y = kernel_first_dst_y - args.kernel_size_y;\n";
  c += "  int offset_x = abs(args.padding_x);\n";
  c += "  int offset_x_strided = offset_x * args.stride_x;\n";
  c +=
      "  int src_x = (kernel_first_dst_x + offset_x_strided) / args.stride_x - "
      "offset_x;\n";
  c += "  int offset_y = abs(args.padding_y);\n";
  c += "  int offset_y_strided = offset_y * args.stride_y;\n";
  c +=
      "  int src_y = (kernel_first_dst_y + offset_y_strided) / args.stride_y - "
      "offset_y;\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "  int kernel_first_dst_z = dst_z + args.padding_z;\n";
    c += "  int kernel_last_dst_z = kernel_first_dst_z - args.kernel_size_z;\n";
    c += "  int offset_z = abs(args.padding_z);\n";
    c += "  int offset_z_strided = offset_z * args.stride_z;\n";
    c += "  int src_z = (kernel_first_dst_z + offset_z_strided) / "
         "args.stride_z - offset_z;\n";
    c += "  int src_as_dst_z = src_z * args.stride_z;\n";
    c +=
        "  for (;src_as_dst_z > kernel_last_dst_z; src_z -= 1, src_as_dst_z -= "
        "args.stride_z) {\n";
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zindex = std::to_string(z);
      c += "    int sz" + zindex + " = src_z + " + zindex + ";\n";
      if (!src_def.SupportsZeroClamp(Axis::DEPTH, gpu_info)) {
        c += "    bool in_z" + zindex + " = sz" + zindex + " >= 0 && sz" +
             zindex + " < args.src_tensor.Depth();\n";
        if (!src_def.CanReadOutOfBorder(Axis::DEPTH, gpu_info)) {
          c += "    sz" + zindex + " = clamp(sz" + zindex +
               ", 0, args.src_tensor.Depth() - 1);\n";
        }
      }
    }
    if (block_size.z == 1 &&
        !src_def.SupportsZeroClamp(Axis::DEPTH, gpu_info)) {
      c += "    if (!in_z0) continue;\n";
    }
    c += "    int kernel_z = kernel_first_dst_z - src_as_dst_z;\n";
    c += "    int src_as_dst_y = src_y * args.stride_y;\n";
    c += "    int src_y_copy = src_y;\n";
    c += "    for (;src_as_dst_y > kernel_last_dst_y; src_y_copy -= 1, "
         "src_as_dst_y -= args.stride_y) {\n";
  } else {
    c += "  int src_as_dst_y = src_y * args.stride_y;\n";
    c += "  for (;src_as_dst_y > kernel_last_dst_y; src_y -= 1, src_as_dst_y "
         "-= args.stride_y) {\n";
  }
  for (int y = 0; y < block_size.y; ++y) {
    const std::string yindex = std::to_string(y);
    const std::string src_y =
        src_def.HasAxis(Axis::DEPTH) ? "src_y_copy" : "src_y";
    c += "    int sy" + yindex + " = " + src_y + " + " + yindex + ";\n";
    if (!src_def.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "    bool in_y" + yindex + " = sy" + yindex + " >= 0 && sy" +
           yindex + " < args.src_tensor.Height();\n";
      if (!src_def.CanReadOutOfBorder(Axis::HEIGHT, gpu_info)) {
        c += "    sy" + yindex + " = clamp(sy" + yindex +
             ", 0, args.src_tensor.Height() - 1);\n";
      }
    }
  }
  if (block_size.y == 1 && !src_def.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "      if (!in_y0) continue;\n";
  }
  c += "    int kernel_y = kernel_first_dst_y - src_as_dst_y;\n";
  c += "    int src_as_dst_x = src_x * args.stride_x;\n";
  c += "    int src_x_copy = src_x;\n";
  c += "    for (;src_as_dst_x > kernel_last_dst_x; src_x_copy -= 1, "
       "src_as_dst_x "
       "-= args.stride_x) {\n";
  for (int x = 0; x < block_size.x; ++x) {
    const std::string xindex = std::to_string(x);
    c += "      int sx" + xindex + " = src_x_copy + " + xindex + ";\n";
    if (!src_def.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      c += "      bool in_x" + xindex + " = sx" + xindex + " >= 0 && sx" +
           xindex + " < args.src_tensor.Width();\n";
      if (!src_def.CanReadOutOfBorder(Axis::WIDTH, gpu_info)) {
        c += "      sx" + xindex + " = clamp(sx" + xindex +
             ", 0, args.src_tensor.Width() - 1);\n";
      }
    }
  }
  if (block_size.x == 1 && !src_def.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "      if (!in_x0) continue;\n";
  }
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zind = std::to_string(z);
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        const std::string id = generate_id(xind, yind, zind);
        const std::string check = generate_check(xind, yind, zind);
        std::string coords = "sx" + xind + ", sy" + yind;
        if (src_def.HasAxis(Axis::DEPTH)) {
          coords += ", sz" + zind;
        }
        if (src_def.IsLinear()) {
          c += "      int addr" + id + " = args.src_tensor.GetAddress(" +
               coords + ", 0);\n";
          if (src_def.ReturnsZeroForNegOneRead(gpu_info)) {
            c += "      addr" + id + " = select(-1, addr" + id + ", (" + check +
                 "));\n";
            c += "      int ds" + id +
                 " = select(0, args.src_tensor.SliceStride(), (" + check +
                 "));\n";
          }
        }
      }
    }
  }
  if (src_def.IsLinear() && !src_def.ReturnsZeroForNegOneRead(gpu_info)) {
    c += "      int ds = args.src_tensor.SliceStride();\n";
  }
  c += "      int kernel_x = kernel_first_dst_x - src_as_dst_x;\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "      int kernel_index = (kernel_z * args.kernel_size_y + kernel_y) "
         "*  args.kernel_size_x + kernel_x;\n";
  } else {
    c += "      int kernel_index = kernel_y * args.kernel_size_x + kernel_x;\n";
  }
  if (weights_are_buffer) {
    c += "      int f_offset = f_base + kernel_index * "
         "args.src_tensor.Slices() * " +
         std::to_string(block_size.w) + ";\n";
  } else {
    c += "      int x_c = kernel_index * args.src_tensor.Slices();\n";
  }
  c += "      for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zind = std::to_string(z);
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        const std::string id = generate_id(xind, yind, zind);
        std::string address;
        if (src_def.IsLinear()) {
          address = "addr" + id;
        } else {
          address = "sx" + xind + ", sy" + yind;
          if (src_def.HasAxis(Axis::DEPTH)) {
            address += ", sz" + zind;
          }
          address += ", s";
        }
        if (src_def.ReturnsZeroForNegOneRead(gpu_info)) {
          c += "        Type src" + id + " = args.src_tensor.Read(" + address +
               "); " + address + " += ds" + id + ";\n";
        } else {
          const std::string check = generate_check(xind, yind, zind);
          if (!check.empty()) {
            c += "        Type src" + id + " = args.src_tensor.Read(" +
                 address + ") * ucl::Convert<SType>(" + check + ");\n";
          } else {
            c += "        Type src" + id + " = args.src_tensor.Read(" +
                 address + ");\n";
          }
          if (src_def.IsLinear()) {
            c += "        addr" + id + " += ds;\n";
          }
        }
      }
    }
  }
  c += "      Type w0, w1, w2, w3;\n";
  for (int s = 0; s < block_size.w; ++s) {
    if (weights_are_buffer) {
      c +=
          "        args.weights.ReadVec16AsVec4x4(w0, w1, w2, w3, f_offset + " +
          std::to_string(s) + ");\n";
    } else {
      c += absl::Substitute(
          R"(        w0 = args.weights0.Read(dst_s + $0, x_c);
        w1 = args.weights1.Read(dst_s + $0, x_c);
        w2 = args.weights2.Read(dst_s + $0, x_c);
        w3 = args.weights3.Read(dst_s + $0, x_c);
)",
          s);
    }
    const std::string sind = std::to_string(s);
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          const std::string id = generate_id(xind, yind, zind);
          const std::string full_id = generate_id_full(xind, yind, zind, sind);
          c += "        CONV(r" + full_id + ", src" + id + ");\n";
        }
      }
    }
  }
  if (weights_are_buffer) {
    c += "        f_offset += " + std::to_string(block_size.w) + ";\n";
  } else {
    c += "        x_c++;\n";
  }
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  if (src_def.HasAxis(Axis::DEPTH)) {
    c += "  }\n";
  }
  for (int s = 0; s < block_size.w; ++s) {
    const std::string sind = std::to_string(s);
    c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
    if (has_bias) {
      c += "    Type bias_val = args.biases.Read(dst_s);\n";
    } else {
      c += "    Type bias_val = ucl::Init<Type>(0.0f);\n";
    }
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          const std::string id = generate_id_full(xind, yind, zind, sind);
          std::string checks =
              "xc < args.dst_tensor.Width() && yc < args.dst_tensor.Height()";
          std::string coords = "xc, yc";
          c += "    {\n";
          c += "      int xc = dst_x + args.stride_x * " + xind + ";\n";
          c += "      int yc = dst_y + args.stride_y * " + yind + ";\n";
          if (src_def.HasAxis(Axis::DEPTH)) {
            c += "      int zc = dst_z + args.stride_z * " + zind + ";\n";
            checks += " && zc < args.dst_tensor.Depth()";
            coords += ", zc";
          }
          c += "      if (" + checks + ") {\n";
          c += "        Type res = ucl::Convert<Type>(r" + id +
               ") + bias_val;\n";
          c += "        args.dst_tensor.Write(res, " + coords + ", dst_s);\n";
          c += "      }\n";
          c += "    }\n";
        }
      }
    }
    c += "  }\n";
    c += "  dst_s++;\n";
  }
  c += "}\n";
  const DataType acc_type = precision == CalculationsPrecision::F16
                                ? DataType::FLOAT16
                                : DataType::FLOAT32;
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll({{"SType", ToUclDataType(type, 1)},
                       {"Type", ToUclDataType(type, 4)},
                       {"AccType", ToUclDataType(acc_type, 4)}},
                      &c);
  return c;
}

absl::Status ConvolutionTransposed::BindArguments(ArgumentsBinder* args) {
  const int aligned_h = AlignByN(dst_[0]->Height(), stride_.y * block_size_.y);
  ABSL_RETURN_IF_ERROR(
      args->SetInt("grid_size_y", DivideRoundUp(aligned_h, block_size_.y)));
  return absl::OkStatus();
}

int3 ConvolutionTransposed::GetGridSize() const {
  const int aligned_w = AlignByN(dst_[0]->Width(), stride_.x * block_size_.x);
  const int aligned_h = AlignByN(dst_[0]->Height(), stride_.y * block_size_.y);
  const int aligned_d = AlignByN(dst_[0]->Depth(), stride_.z * block_size_.z);
  const int grid_x = DivideRoundUp(aligned_w, block_size_.x) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(aligned_h, block_size_.y) *
                     DivideRoundUp(aligned_d, block_size_.z);
  const int grid_z = DivideRoundUp(dst_[0]->Slices(), block_size_.w);
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> ConvolutionTransposed::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  return GetPossibleWorkGroupsConv(tuning_type, gpu_info, kernel_info,
                                   grid_size_);
}

ConvolutionTransposed CreateConvolutionTransposed(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  ConvolutionTransposed result(definition, precision, attr, gpu_info);
  result.UploadWeights(attr.weights, UseBufferForWeights(gpu_info));

  if (!attr.bias.data.empty()) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

ConvolutionTransposed CreateConvolutionTransposed3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposed3DAttributes& attr) {
  ConvolutionTransposed result(definition, precision, attr, gpu_info);
  result.UploadWeights(attr.weights, UseBufferForWeights(gpu_info));

  if (!attr.bias.data.empty()) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  OperationDef new_def = definition;
  new_def.src_tensors = {
      definition.src_tensors[0]};  // leaving only src_tensor def, weights defs
                                   // will be added later
  const DataType weights_type = definition.src_tensors[0].GetDataType();
  if (UseBufferForWeights(gpu_info)) {
    // add 1 src_tensor(buffer) for weights
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::BUFFER, Layout::HWC});
  } else {
    // add 4 src_tensors(4X textures 2d) for weights
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HW});
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HW});
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HW});
    new_def.src_tensors.push_back(
        {weights_type, TensorStorageType::TEXTURE_2D, Layout::HW});
  }
  ConvolutionTransposed result(new_def, precision, attr, gpu_info);

  if (!attr.bias.data.empty()) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

}  // namespace ml_drift
