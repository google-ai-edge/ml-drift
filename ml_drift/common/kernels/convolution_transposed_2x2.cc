// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/convolution_transposed_2x2.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/convolution_transposed_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GenerateConv(CalculationsPrecision precision,
                         const std::string& dst, const std::string& src,
                         int index) {
  std::string code;
  switch (precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      code += "    $0 += $1.x * weights_cache[$2];\n";
      code += "    $0 += $1.y * weights_cache[$3];\n";
      code += "    $0 += $1.z * weights_cache[$4];\n";
      code += "    $0 += $1.w * weights_cache[$5];\n";
      break;
    case CalculationsPrecision::F32_F16:
      code +=
          "    $0 += ucl::Convert<AccType>($1.x * weights_cache[$2] + $1.y * "
          "weights_cache[$3] + $1.z * "
          "weights_cache[$4] + $1.w * weights_cache[$5]);\n";
      break;
  }
  return absl::Substitute(code, dst, src, index * 4 + 0, index * 4 + 1,
                          index * 4 + 2, index * 4 + 3);
}

std::string GenerateConvolutionTransposedCode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, WeightsUploadType weights_upload_type,
    const int3& work_group_size, bool has_bias) {
  const auto& src_desc = op_def.src_tensors[0];
  const bool need_local_mem =
      weights_upload_type == WeightsUploadType::kLocalMemoryByThreads ||
      weights_upload_type == WeightsUploadType::kLocalMemoryWorkGroupLoad;
  const bool late_spatial_oob_check =
      need_local_mem || weights_upload_type == WeightsUploadType::kWaveMemory;

  const std::string weights_space =
      weights_upload_type == WeightsUploadType::kConstantMemory ? "__constant"
                                                                : "__global";

  std::string c;
  if (weights_upload_type == WeightsUploadType::kWaveMemory) {
    c += "#pragma OPENCL EXTENSION ucl_wave_memory: enable\n";
  } else {
    if (need_local_mem && gpu_info.IsApiOpenCl()) {
      c += "__attribute__((reqd_work_group_size(" +
           std::to_string(work_group_size.x) + ", " +
           std::to_string(work_group_size.y) + ", 1)))\n";
    }
  }
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  if (need_local_mem) {
    c += "  int Z = ucl::GetGroupId<2>();\n";
  } else {
    c += "  int Z = ucl::GetGlobalId<2>();\n";
  }
  if (!late_spatial_oob_check) {
    c += "  if (X * 2 + 1 >= args.dst_tensor.Width() || Y * 2 + 1 >= "
         "args.dst_tensor.Height()) return;\n";
  }
  c += "  if (Z >= args.dst_tensor.Slices()) return;\n";
  c += "  AccType r0 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r1 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r2 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r3 = ucl::Init<AccType>(0.0f);\n";
  c += "  int f_offset = Z * 4 * 4 * args.src_tensor.Slices();\n";
  if (weights_upload_type == WeightsUploadType::kWaveMemory) {
    c += "  __wave Type weights_cache[16];\n";
  } else if (need_local_mem) {
    c += "  __local Type weights_cache[16];\n";
    c += "  int local_id = ucl::GetLocalId<1>() * 8 + ucl::GetLocalId<0>();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "  bool in_x = X < args.src_tensor.Width();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "  bool in_y = Y < args.src_tensor.Height();\n";
  }
  if (src_desc.IsLinear()) {
    if (src_desc.ReturnsZeroForNegOneRead(gpu_info)) {
      c += "  int addr = args.src_tensor.GetAddress(X, Y, 0);\n";
      c += "  addr = select(-1, addr, (in_x && in_y));\n";
      c += "  int dz = select(0, args.src_tensor.SliceStride(), (in_x && "
           "in_y));\n";
    } else {
      c += "  int xc = clamp(X, 0, args.src_tensor.Width() - 1);\n";
      c += "  int yc = clamp(Y, 0, args.src_tensor.Height() - 1);\n";
      c += "  int addr = args.src_tensor.GetAddress(xc, yc, 0);\n";
      c += "  int dz = args.src_tensor.SliceStride();\n";
    }
  }
  auto generate_check = [&]() {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT};
    const std::vector<std::string> names{"in_x", "in_y"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) &&
          !src_desc.SupportsZeroClamp(axis, gpu_info)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i];
      }
    }
    return check;
  };
  c += "  for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  if (src_desc.IsLinear()) {
    if (src_desc.ReturnsZeroForNegOneRead(gpu_info)) {
      c += "    Type src0 = args.src_tensor.Read(addr); addr += dz;\n";
    } else {
      c += "    Type src0 = args.src_tensor.Read(addr) * "
           "ucl::Convert<SType>(in_x && in_y); addr += dz;\n";
    }
  } else {
    std::string check = generate_check();
    if (!check.empty()) {
      check = " * ucl::Convert<SType>(" + check + ")";
    }
    c += "    Type src0 = args.src_tensor.Read(X, Y, s)" + check + ";\n";
  }
  if (weights_upload_type == WeightsUploadType::kWaveMemory) {
    if (!gpu_info.IsAdreno()) {
      c += "    ucl::SyncThreads<WaveLoad>();\n";
    }
    c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), f_offset, "
         "16);\n";
    c += "    ucl::SyncThreads<WaveLoad>();\n";
  } else if (weights_upload_type == WeightsUploadType::kLocalMemoryByThreads) {
    c += "    if (local_id < 16) {\n";
    c += "      weights_cache[local_id] = args.weights.Read(f_offset + "
         "local_id);\n";
    c += "    };\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  } else if (weights_upload_type ==
             WeightsUploadType::kLocalMemoryWorkGroupLoad) {
    c += "    async_work_group_copy(weights_cache, "
         "args.weights.GetPtr(f_offset), 16, "
         "0);\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  } else {
    c += "    " + weights_space +
         " Type* weights_cache = args.weights.GetPtr(f_offset);\n";
  }
  c += "    f_offset += 16;\n";
  c += GenerateConv(precision, "r0", "src0", 0);
  c += GenerateConv(precision, "r1", "src0", 1);
  c += GenerateConv(precision, "r2", "src0", 2);
  c += GenerateConv(precision, "r3", "src0", 3);
  if (need_local_mem) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  }
  c += "  }\n";
  if (late_spatial_oob_check) {
    c += "  if (X * 2 + 1 >= args.dst_tensor.Width() || Y * 2 + 1 >= "
         "args.dst_tensor.Height()) return;\n";
  }
  c += "\n";
  c += "  X = X * 2;\n";
  c += "  Y = Y * 2;\n";
  c += "\n";
  if (has_bias) {
    c += "  Type bias_val = args.biases.Read(Z);\n";
  } else {
    c += "  Type bias_val = ucl::Init<Type>(0.0f);\n";
  }
  c += "  if (X < args.dst_tensor.Width() && Y < args.dst_tensor.Height()) {\n";
  c += "    Type result = ucl::Convert<Type>(r0) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "  }\n";
  c += "  if (X + 1 < args.dst_tensor.Width() && Y < args.dst_tensor.Height()) "
       "{\n";
  c += "    Type result = ucl::Convert<Type>(r1) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X + 1, Y, Z);\n";
  c += "  }\n";
  c += "  if (X < args.dst_tensor.Width() && Y + 1 < args.dst_tensor.Height()) "
       "{\n";
  c += "    Type result = ucl::Convert<Type>(r2) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X, Y + 1, Z);\n";
  c += "  }\n";
  c += "  if (X + 1 < args.dst_tensor.Width() && Y + 1 < "
       "args.dst_tensor.Height()) {\n";
  c += "    Type result = ucl::Convert<Type>(r3) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X + 1, Y + 1, Z);\n";
  c += "  }\n";
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

}  // namespace

ConvolutionTransposed2x2::ConvolutionTransposed2x2(
    const OperationDef& definition, CalculationsPrecision precision,
    const GpuInfo& gpu_info, bool has_bias) {
  weights_upload_type_ =
      GetWeightsUploadTypeForFixedSizeConvTransposed(gpu_info);
  work_group_size_ = int3(8, 4, 1);
  if (weights_upload_type_ == WeightsUploadType::kWaveMemory) {
    if (gpu_info.IsAdreno()) {
      wave_size_ = 128;
      work_group_size_ = int3(16, 8, 1);
    } else if (gpu_info.IsIntel()) {
      wave_size_ = 32;
      work_group_size_ = int3(8, 4, 1);
    } else if (gpu_info.IsPowerVR()) {
      wave_size_ = 128;
      work_group_size_ = int3(16, 8, 1);
    }
  }
  weights_data_type_ = DeduceDataTypeFromPrecision(precision);
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  code_ = GenerateConvolutionTransposedCode(gpu_info, definition, precision,
                                            weights_upload_type_,
                                            work_group_size_, has_bias);
}

int3 ConvolutionTransposed2x2::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width(), 2) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> ConvolutionTransposed2x2::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (weights_upload_type_ == WeightsUploadType::kWaveMemory) {
    switch (tuning_type) {
      case TuningType::kExhaustive:
        return GetWorkGroupsXYMultipleOf(wave_size_, gpu_info, kernel_info,
                                         grid_size_);
      case TuningType::kFast:
      default:
        return {GetConvWorkGroupXYisMultipleK(wave_size_, grid_size_)};
    }
  } else if (weights_upload_type_ == WeightsUploadType::kLocalMemoryByThreads ||
             weights_upload_type_ ==
                 WeightsUploadType::kLocalMemoryWorkGroupLoad) {
    return {work_group_size_};
  } else {  // no wg size restrictions
    switch (tuning_type) {
      case TuningType::kExhaustive:
        return GetPossibleWorkGroupsConv(tuning_type, gpu_info, kernel_info,
                                         grid_size_);
      case TuningType::kFast:
      default:
        return {work_group_size_};
    }
  }
}

std::vector<int> ConvolutionTransposed2x2::GetSpatialWeightsRemap() const {
  return std::vector<int>{0, 1, 2, 3};
}

bool IsConvolutionTransposed2x2Supported(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  const bool supported_attributes =
      attr.weights.shape.w == 2 && attr.weights.shape.h == 2 &&
      attr.stride.w == 2 && attr.stride.h == 2 &&
      attr.padding.prepended.w == 0 && attr.padding.prepended.h == 0;
  return supported_attributes;
}

ConvolutionTransposed2x2 CreateConvolutionTransposed2x2(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  bool has_bias = !attr.bias.data.empty();
  ConvolutionTransposed2x2 result(definition, precision, gpu_info, has_bias);
  {
    const auto weights_desc = result.GetWeightsDescription();
    const int flt_count =
        GetTotalElementsCountForLayout(weights_desc, attr.weights.shape);
    BufferDescriptor buffer_desc =
        GetWeightsBufferDescForFixedSizeConvTransposed(
            gpu_info, result.weights_data_type_, result.weights_upload_type_);
    buffer_desc.size = flt_count * SizeOf(result.weights_data_type_);
    buffer_desc.data.resize(buffer_desc.size);
    RearrangeWeights(attr.weights, weights_desc,
                     absl::MakeSpan(buffer_desc.data));
    result.args_.AddObject(
        "weights", std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
  }
  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

ConvolutionTransposed2x2 CreateConvolutionTransposed2x2DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  bool has_bias = !attr.bias.data.empty();
  ConvolutionTransposed2x2 result(definition, precision, gpu_info, has_bias);
  result.AddSrcBuffer("weights", GetWeightsBufferDescForFixedSizeConvTransposed(
                                     gpu_info, result.weights_data_type_,
                                     result.weights_upload_type_));

  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

}  // namespace ml_drift
