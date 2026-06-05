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

#include "ml_drift/common/kernels/convolution_transposed_4x4.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
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
std::string GenerateConv(CalculationsPrecision precision, bool is_i4o4,
                         const std::string& dst, const std::string& src,
                         int index) {
  std::string code;
  if (is_i4o4) {
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
  } else {
    code += "    $0.x += dot($1, weights_cache[$2]);\n";
    code += "    $0.y += dot($1, weights_cache[$3]);\n";
    code += "    $0.z += dot($1, weights_cache[$4]);\n";
    code += "    $0.w += dot($1, weights_cache[$5]);\n";
  }
  return absl::Substitute(code, dst, src, index * 4 + 0, index * 4 + 1,
                          index * 4 + 2, index * 4 + 3);
}
}  // namespace

ConvolutionTransposed4x4::ConvolutionTransposed4x4(
    const OperationDef& definition, CalculationsPrecision precision,
    const GpuInfo& gpu_info, bool has_bias) {
  work_group_size_ = int3(8, 4, 1);
  if (gpu_info.IsApple()) {
    work_group_launch_order_ = int3(2, 0, 1);
  }

  weights_upload_type_ =
      GetWeightsUploadTypeForFixedSizeConvTransposed(gpu_info);
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

  if (gpu_info.IsDotPreferred()) {
    weights_layout_ = WeightsLayout::kOICustomSpatialO4I4;
  } else {
    weights_layout_ = WeightsLayout::kOICustomSpatialI4O4;
  }
  weights_data_type_ = DeduceDataTypeFromPrecision(precision);

  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  code_ = GenerateConvolutionTransposedCode(gpu_info, definition, precision,
                                            weights_upload_type_, has_bias);
  if (precision == CalculationsPrecision::F16 && gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
}

std::string ConvolutionTransposed4x4::GenerateConvolutionTransposedCode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, WeightsUploadType weights_upload_type,
    bool has_bias) {
  auto src_desc = op_def.src_tensors[0];

  const bool need_local_mem =
      weights_upload_type == WeightsUploadType::kLocalMemoryByThreads ||
      weights_upload_type == WeightsUploadType::kLocalMemoryWorkGroupLoad;
  const bool is_wave_memory_upload =
      weights_upload_type == WeightsUploadType::kWaveMemory;
  const bool late_spatial_oob_check = need_local_mem || is_wave_memory_upload;

  const int wg_total_size =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const std::string scope = gpu_info.IsApiMetal() && wg_total_size == 32 &&
                                    gpu_info.IsWaveSizeEqualTo32()
                                ? "SubGroup"
                                : "WorkGroup";
  const std::string local_mem_barrier =
      "ucl::SyncThreads<" + scope + ", Local>()";
  const bool is_i4o4 = GetWeightsDescription().IsI4O4();

  std::string c;
  const std::string weights_space =
      weights_upload_type == WeightsUploadType::kConstantMemory ? "__constant"
                                                                : "__global";

  if (is_wave_memory_upload) {
    c += "#pragma OPENCL EXTENSION ucl_wave_memory: enable\n";
  } else {
    if (need_local_mem && gpu_info.IsApiOpenCl()) {
      c += "__attribute__((reqd_work_group_size(" +
           std::to_string(work_group_size_.x) + ", " +
           std::to_string(work_group_size_.y) + ", 1)))\n";
    }
  }
  c += "MAIN_FUNCTION($0) {\n";
  std::string grid_coords[3];
  int3 launch_remap;
  launch_remap[work_group_launch_order_.x] = 0;
  launch_remap[work_group_launch_order_.y] = 1;
  launch_remap[work_group_launch_order_.z] = 2;
  if (work_group_launch_order_[0] == 0) {
    grid_coords[0] = "ucl::GetGlobalId<0>()";
  } else {
    grid_coords[0] = "(ucl::GetGroupId<" + std::to_string(launch_remap[0]) +
                     ">() * ucl::GetGroupSize<0>() + ucl::GetLocalId<0>());\n";
  }
  if (work_group_launch_order_[1] == 1) {
    grid_coords[1] = "ucl::GetGlobalId<1>()";
  } else {
    grid_coords[1] = "(ucl::GetGroupId<" + std::to_string(launch_remap[1]) +
                     ">() * ucl::GetGroupSize<1>() + ucl::GetLocalId<1>());\n";
  }
  if (work_group_launch_order_[2] == 2) {
    grid_coords[2] = "ucl::GetGlobalId<2>()";
    if (need_local_mem) {
      grid_coords[2] = "ucl::GetGroupId<2>()";
    }
  } else {
    grid_coords[2] = "(ucl::GetGroupId<" + std::to_string(launch_remap[2]) +
                     ">() * ucl::GetGroupSize<2>() + ucl::GetLocalId<2>());\n";
    if (need_local_mem) {
      grid_coords[2] =
          "ucl::GetGroupId<" + std::to_string(launch_remap[2]) + ">()";
    }
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = " + grid_coords[0] + ";\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = " + grid_coords[0] + ";\n";
  }
  c += "  int Y = " + grid_coords[1] + ";\n";
  c += "  int Z = " + grid_coords[2] + ";\n";
  if (!late_spatial_oob_check) {
    c += "  if (X * 2 > args.dst_tensor.Width() || Y * 2 > "
         "args.dst_tensor.Height()) return;\n";
  }
  c += "  if (Z >= args.dst_tensor.Slices()) return;\n";
  c += "  AccType r0 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r1 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r2 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r3 = ucl::Init<AccType>(0.0f);\n";
  c += "  int f_offset = Z * args.src_tensor.Slices() * 64;\n";
  if (need_local_mem) {
    c += "  __local Type weights_cache[64];\n";
  }
  if (is_wave_memory_upload) {
    c += "  __wave Type weights_cache[24];\n";
  }
  if (weights_upload_type == WeightsUploadType::kLocalMemoryByThreads) {
    c += "  int local_id = ucl::GetLocalId<1>() * 8 + ucl::GetLocalId<0>();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "  bool in_x0 = X - 1 >= 0 && X - 1 < args.src_tensor.Width();\n";
    c += "  bool in_x1 = X >= 0 && X < args.src_tensor.Width();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "  bool in_y0 = Y - 1 >= 0 && Y - 1 < args.src_tensor.Height();\n";
    c += "  bool in_y1 = Y >= 0 && Y < args.src_tensor.Height();\n";
  }
  auto generate_check = [&](int x, int y) {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT};
    const std::vector<std::string> names{"in_x" + std::to_string(x),
                                         "in_y" + std::to_string(y)};
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
  if (src_desc.IsLinear()) {
    if (src_desc.ReturnsZeroForNegOneRead(gpu_info)) {
      c += "  int addr_0 = args.src_tensor.GetAddress(X - 1, Y - 1, 0);\n";
      c += "  int addr_1 = args.src_tensor.GetAddress(X, Y - 1, 0);\n";
      c += "  int addr_2 = args.src_tensor.GetAddress(X - 1, Y, 0);\n";
      c += "  int addr_3 = args.src_tensor.GetAddress(X, Y, 0);\n";
      c += "  addr_0 = select(-1, addr_0, (in_x0 && in_y0));\n";
      c += "  addr_1 = select(-1, addr_1, (in_x1 && in_y0));\n";
      c += "  addr_2 = select(-1, addr_2, (in_x0 && in_y1));\n";
      c += "  addr_3 = select(-1, addr_3, (in_x1 && in_y1));\n";
      c += "  int dz_0 = select(0, args.src_tensor.SliceStride(), (in_x0 && "
           "in_y0));\n";
      c += "  int dz_1 = select(0, args.src_tensor.SliceStride(), (in_x1 && "
           "in_y0));\n";
      c += "  int dz_2 = select(0, args.src_tensor.SliceStride(), (in_x0 && "
           "in_y1));\n";
      c += "  int dz_3 = select(0, args.src_tensor.SliceStride(), (in_x1 && "
           "in_y1));\n";
    } else {
      c += "  int xc0 = clamp(X - 1, 0, args.src_tensor.Width() - 1);\n";
      c += "  int xc1 = clamp(X, 0, args.src_tensor.Width() - 1);\n";
      c += "  int yc0 = clamp(Y - 1, 0, args.src_tensor.Height() - 1);\n";
      c += "  int yc1 = clamp(Y, 0, args.src_tensor.Height() - 1);\n";
      c += "  int addr_0 = args.src_tensor.GetAddress(xc0, yc0, 0);\n";
      c += "  int addr_1 = args.src_tensor.GetAddress(xc1, yc0, 0);\n";
      c += "  int addr_2 = args.src_tensor.GetAddress(xc0, yc1, 0);\n";
      c += "  int addr_3 = args.src_tensor.GetAddress(xc1, yc1, 0);\n";
      c += "  int dz = args.src_tensor.SliceStride();\n";
    }
  }
  auto read_src = [&](int x, int y) {
    if (src_desc.IsLinear()) {
      const std::string id = std::to_string(y * 2 + x);
      const std::string addr = "addr_" + std::to_string(y * 2 + x);
      if (src_desc.ReturnsZeroForNegOneRead(gpu_info)) {
        return "args.src_tensor.Read(" + addr + "); " + addr + " += dz_" + id +
               ";";
      } else {
        return "args.src_tensor.Read(" + addr + ") * ucl::Convert<SType>(in_x" +
               std::to_string(x) + " && in_y" + std::to_string(y) + "); " +
               addr + " += dz;";
      }
    } else {
      std::string check = generate_check(x, y);
      if (!check.empty()) {
        check = " * ucl::Convert<SType>(" + check + ")";
      }
      return "args.src_tensor.Read(X + " + std::to_string(x - 1) + ", Y + " +
             std::to_string(y - 1) + ", s)" + check + ";";
    }
  };
  c += "  for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  if (is_wave_memory_upload) {
    c += "    Type src0 = " + read_src(0, 0) + "\n";
    c += "    Type src1 = " + read_src(1, 0) + "\n";
    if (!gpu_info.IsAdreno()) {
      c += "    ucl::SyncThreads<WaveLoad>();\n";
    }
    c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), f_offset, "
         "24);\n";
    c += "    f_offset += 24;\n";
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    c += GenerateConv(precision, is_i4o4, "r0", "src0", 0);
    c += GenerateConv(precision, is_i4o4, "r1", "src0", 1);
    c += GenerateConv(precision, is_i4o4, "r2", "src0", 2);
    c += GenerateConv(precision, is_i4o4, "r3", "src0", 3);
    c += GenerateConv(precision, is_i4o4, "r0", "src1", 4);
    c += GenerateConv(precision, is_i4o4, "r1", "src1", 5);
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    c += "    Type src2 = " + read_src(0, 1) + "\n";
    c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), f_offset, "
         "24);\n";
    c += "    f_offset += 24;\n";
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    c += GenerateConv(precision, is_i4o4, "r2", "src1", 0);
    c += GenerateConv(precision, is_i4o4, "r3", "src1", 1);
    c += GenerateConv(precision, is_i4o4, "r0", "src2", 2);
    c += GenerateConv(precision, is_i4o4, "r1", "src2", 3);
    c += GenerateConv(precision, is_i4o4, "r2", "src2", 4);
    c += GenerateConv(precision, is_i4o4, "r3", "src2", 5);
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    c += "    Type src3 = " + read_src(1, 1) + "\n";
    c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), f_offset, "
         "16);\n";
    c += "    f_offset += 16;\n";
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    c += GenerateConv(precision, is_i4o4, "r0", "src3", 0);
    c += GenerateConv(precision, is_i4o4, "r1", "src3", 1);
    c += GenerateConv(precision, is_i4o4, "r2", "src3", 2);
    c += GenerateConv(precision, is_i4o4, "r3", "src3", 3);
  } else {
    if (need_local_mem) {
      c += "    " + local_mem_barrier + ";\n";
    }
    if (weights_upload_type == WeightsUploadType::kLocalMemoryWorkGroupLoad) {
      c += "    async_work_group_copy(weights_cache, "
           "args.weights.GetPtr(f_offset), 64, "
           "0);\n";
    } else if (weights_upload_type ==
               WeightsUploadType::kLocalMemoryByThreads) {
      c += "    weights_cache[local_id] = args.weights.Read(f_offset + "
           "local_id);\n";
      c += "    weights_cache[local_id + 32] = args.weights.Read(f_offset + "
           "local_id + "
           "32);\n";
    } else {  // GLOBAL_MEM/CONSTANT_MEM
      // TODO: Add a GLOBAL_MEM/CONSTANT_MEM version compatible with APIs like
      // WebGPU. For now, we just log an error rather than weight for WGSL
      // compiler to complain.
      if (!gpu_info.SupportsPointersInKernels()) {
        ABSL_LOG(ERROR)
            << "ConvolutionTransposed4x4 operation currently only "
            << "supports GLOBAL_MEM and CONSTANT_MEM weight uploads "
            << "for OpenCL and Metal.";
      }
      c += "    " + weights_space +
           " Type* weights_cache = args.weights.GetPtr(f_offset);\n";
    }
    c += "    Type src0 = " + read_src(0, 0) + ";\n";
    c += "    Type src1 = " + read_src(1, 0) + ";\n";
    c += "    Type src2 = " + read_src(0, 1) + ";\n";
    c += "    Type src3 = " + read_src(1, 1) + ";\n";
    c += "    f_offset += 64;\n";
    if (need_local_mem) {
      c += "    " + local_mem_barrier + ";\n";
    }
    c += GenerateConv(precision, is_i4o4, "r0", "src0", 0);
    c += GenerateConv(precision, is_i4o4, "r1", "src0", 1);
    c += GenerateConv(precision, is_i4o4, "r2", "src0", 2);
    c += GenerateConv(precision, is_i4o4, "r3", "src0", 3);
    c += GenerateConv(precision, is_i4o4, "r0", "src1", 4);
    c += GenerateConv(precision, is_i4o4, "r1", "src1", 5);
    c += GenerateConv(precision, is_i4o4, "r2", "src1", 6);
    c += GenerateConv(precision, is_i4o4, "r3", "src1", 7);
    c += GenerateConv(precision, is_i4o4, "r0", "src2", 8);
    c += GenerateConv(precision, is_i4o4, "r1", "src2", 9);
    c += GenerateConv(precision, is_i4o4, "r2", "src2", 10);
    c += GenerateConv(precision, is_i4o4, "r3", "src2", 11);
    c += GenerateConv(precision, is_i4o4, "r0", "src3", 12);
    c += GenerateConv(precision, is_i4o4, "r1", "src3", 13);
    c += GenerateConv(precision, is_i4o4, "r2", "src3", 14);
    c += GenerateConv(precision, is_i4o4, "r3", "src3", 15);
  }
  c += "  }\n";
  c += "\n";
  if (late_spatial_oob_check) {
    c += "  if (X * 2 > args.dst_tensor.Width() || Y * 2 > "
         "args.dst_tensor.Height()) return;\n";
  }
  c += "  X = X * 2 - 1;\n";
  c += "  Y = Y * 2 - 1;\n";
  c += "\n";
  if (has_bias) {
    c += "  Type bias_val = args.biases.Read(Z);\n";
  } else {
    c += "  Type bias_val = ucl::Init<Type>(0.0f);\n";
  }
  c += "  if (X >= 0 && Y >= 0) {\n";
  c += "    Type result = ucl::Convert<Type>(r0) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "  }\n";
  c += "  if (X + 1 < args.dst_tensor.Width() && Y >= 0) {\n";
  c += "    Type result = ucl::Convert<Type>(r1) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X + 1, Y, Z);\n";
  c += "  }\n";
  c += "  if (X >= 0 && Y + 1 < args.dst_tensor.Height()) {\n";
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
                       {"AccType", ToUclDataType(acc_type, 4)},
                       {"Type", ToUclDataType(type, 4)}},
                      &c);
  return c;
}

int3 ConvolutionTransposed4x4::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width() + 2, 2) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height() + 2, 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> ConvolutionTransposed4x4::GetPossibleKernelWorkGroups(
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
  } else if (weights_upload_type_ ==
                 WeightsUploadType::kLocalMemoryWorkGroupLoad ||
             weights_upload_type_ == WeightsUploadType::kLocalMemoryByThreads) {
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

std::vector<int> ConvolutionTransposed4x4::GetSpatialWeightsRemap() const {
  return std::vector<int>{10, 11, 14, 15, 8, 9, 12, 13, 2, 3, 6, 7, 0, 1, 4, 5};
}

void ConvolutionTransposed4x4::UploadWeights(
    const GpuInfo& gpu_info, const Tensor<OHWI, DataType::FLOAT32>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor buffer_desc = GetWeightsBufferDescForFixedSizeConvTransposed(
      gpu_info, weights_data_type_, weights_upload_type_);
  buffer_desc.size = flt_count * SizeOf(weights_data_type_);
  buffer_desc.data.resize(buffer_desc.size);

  RearrangeWeights(weights, weights_desc, absl::MakeSpan(buffer_desc.data));
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

bool IsConvolutionTransposed4x4Supported(
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.w == 4 && attr.weights.shape.h == 4 &&
         attr.stride.w == 2 && attr.stride.h == 2 &&
         attr.padding.prepended.w == 1 && attr.padding.prepended.h == 1;
}

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  bool has_bias = !attr.bias.data.empty();
  ConvolutionTransposed4x4 result(definition, precision, gpu_info, has_bias);
  result.UploadWeights(gpu_info, attr.weights);

  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  bool has_bias = !attr.bias.data.empty();
  ConvolutionTransposed4x4 result(definition, precision, gpu_info, has_bias);
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
