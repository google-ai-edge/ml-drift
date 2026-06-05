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

#include "ml_drift/common/kernels/convolution_transposed_3x3.h"

#include <cstdlib>
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

ConvolutionTransposed3x3::ConvolutionTransposed3x3(
    const OperationDef& definition, CalculationsPrecision precision,
    const GpuInfo& gpu_info, int2 padding, bool has_bias)
    : padding_(padding) {
  work_group_size_ = int3(8, 4, 1);
  work_group_launch_order_ = int3(2, 0, 1);
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
  const int padding_x =
      padding_.x >= 1 ? (padding_.x - 1) / 2 : (padding_.x - 2) / 2;
  const int padding_y =
      padding_.y >= 1 ? (padding_.y - 1) / 2 : (padding_.y - 2) / 2;
  args_.AddInt("padding_x", padding_x);
  args_.AddInt("padding_y", padding_y);
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  code_ = GenerateConvolutionTransposedCode(gpu_info, definition, precision,
                                            weights_upload_type_, padding_,
                                            work_group_launch_order_, has_bias);
  if (precision == CalculationsPrecision::F16 && gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
}

std::string ConvolutionTransposed3x3::GenerateConvolutionTransposedCode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, WeightsUploadType weights_upload_type,
    int2 padding, int3 work_group_launch_order, bool has_bias) {
  auto src_desc = op_def.src_tensors[0];

  const bool need_local_mem =
      weights_upload_type == WeightsUploadType::kLocalMemoryByThreads ||
      weights_upload_type == WeightsUploadType::kLocalMemoryWorkGroupLoad;
  const bool is_wave_memory_upload =
      weights_upload_type == WeightsUploadType::kWaveMemory;
  const bool late_spatial_oob_check = need_local_mem || is_wave_memory_upload;

  const bool is_i4o4 = GetWeightsDescription().IsI4O4();
  const int wg_total_size =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const std::string scope = gpu_info.IsApiMetal() && wg_total_size == 32 &&
                                    gpu_info.IsWaveSizeEqualTo32()
                                ? "SubGroup"
                                : "WorkGroup";
  const std::string local_mem_barrier =
      "ucl::SyncThreads<" + scope + ", Local>()";
  const std::string weights_space =
      weights_upload_type == WeightsUploadType::kConstantMemory ? "__constant"
                                                                : "__global";

  std::string c;
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
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  auto GetGlobalID = [&](int id) {
    std::string result;
    const std::string sid = std::to_string(id);
    if (work_group_launch_order[id] == id) {
      return "ucl::GetGlobalId<" + sid + ">()";
    } else {
      return "ucl::GetGroupId<" + std::to_string(launch_remap[id]) +
             ">() * ucl::GetGroupSize<" + sid + ">() + ucl::GetLocalId<" + sid +
             ">()";
    }
  };
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = " + GetGlobalID(0) + ";\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = " + GetGlobalID(0) + ";\n";
  }
  c += "  int DST_X = X * 2;\n";
  c += "  int SRC_X = X + args.padding_x;\n";
  c += "  int Y = " + GetGlobalID(1) + ";\n";
  c += "  int DST_Y = Y * 2;\n";
  c += "  int SRC_Y = Y + args.padding_y;\n";
  std::string z_coord;
  if (work_group_launch_order_[2] == 2) {
    z_coord = "ucl::GetGlobalId<2>()";
    if (need_local_mem) {
      z_coord = "ucl::GetGroupId<2>()";
    }
  } else {
    z_coord = "(ucl::GetGroupId<" + std::to_string(launch_remap[2]) +
              ">() * ucl::GetGroupSize<2>() + ucl::GetLocalId<2>());\n";
    if (need_local_mem) {
      z_coord = "ucl::GetGroupId<" + std::to_string(launch_remap[2]) + ">()";
    }
  }
  c += "  int Z = " + z_coord + ";\n";
  if (!late_spatial_oob_check) {
    c += "  if (DST_X >= args.dst_tensor.Width() || DST_Y >= "
         "args.dst_tensor.Height()) return;\n";
  }
  c += "  if (Z >= args.dst_tensor.Slices()) return;\n";
  c += "  AccType r0 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r1 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r2 = ucl::Init<AccType>(0.0f);\n";
  c += "  AccType r3 = ucl::Init<AccType>(0.0f);\n";
  c += "  int f_offset = Z * args.src_tensor.Slices() * 4 * 9;\n";
  if (need_local_mem) {
    c += "  __local Type weights_cache[36];\n";
  }
  if (is_wave_memory_upload) {
    c += "  __wave Type weights_cache[24];\n";
  }
  if (weights_upload_type == WeightsUploadType::kLocalMemoryByThreads) {
    c += "  int local_id = ucl::GetLocalId<1>() * 8 + ucl::GetLocalId<0>();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "  bool in_x0 = SRC_X >= 0 && SRC_X < args.src_tensor.Width();\n";
    c += "  bool in_x1 = SRC_X + 1 >= 0 && SRC_X + 1 < "
         "args.src_tensor.Width();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "  bool in_y0 = SRC_Y >= 0 && SRC_Y < args.src_tensor.Height();\n";
    c += "  bool in_y1 = SRC_Y + 1 >= 0 && SRC_Y + 1 < "
         "args.src_tensor.Height();\n";
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
      c += "  int addr_0 = args.src_tensor.GetAddress(SRC_X, SRC_Y, 0);\n";
      c += "  int addr_1 = args.src_tensor.GetAddress(SRC_X + 1, SRC_Y, 0);\n";
      c += "  int addr_2 = args.src_tensor.GetAddress(SRC_X, SRC_Y + 1, 0);\n";
      c += "  int addr_3 = args.src_tensor.GetAddress(SRC_X+1, SRC_Y+1, 0);\n";
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
      c += "  int xc0 = clamp(SRC_X, 0, args.src_tensor.Width() - 1);\n";
      c += "  int xc1 = clamp(SRC_X + 1, 0, args.src_tensor.Width() - 1);\n";
      c += "  int yc0 = clamp(SRC_Y, 0, args.src_tensor.Height() - 1);\n";
      c += "  int yc1 = clamp(SRC_Y + 1, 0, args.src_tensor.Height() - 1);\n";
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
               ";\n";
      } else {
        return "args.src_tensor.Read(" + addr + ") * ucl::Convert<SType>(in_x" +
               std::to_string(x) + " && in_y" + std::to_string(y) + "); " +
               addr + " += dz;\n";
      }
    } else {
      std::string check = generate_check(x, y);
      if (!check.empty()) {
        check = " * ucl::Convert<SType>(" + check + ")";
      }
      return "args.src_tensor.Read(SRC_X + " + std::to_string(x) +
             ", SRC_Y + " + std::to_string(y) + ", s)" + check + ";\n";
    }
  };
  const int padding_x_rem = abs(padding.x) % 2;
  const int padding_y_rem = abs(padding.y) % 2;
  std::vector<std::pair<int, int>> permutation;
  if (padding_x_rem == 1 && padding_y_rem == 1) {
    permutation = {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 2},
                   {3, 0}, {3, 1}, {3, 2}, {3, 3}};
  } else if (padding_x_rem == 0 && padding_y_rem == 1) {
    permutation = {{0, 0}, {0, 1}, {1, 1}, {2, 0}, {2, 1},
                   {2, 2}, {2, 3}, {3, 1}, {3, 3}};
  } else if (padding_x_rem == 1 && padding_y_rem == 0) {
    permutation = {{0, 0}, {0, 2}, {1, 0}, {1, 1}, {1, 2},
                   {1, 3}, {2, 2}, {3, 2}, {3, 3}};
  } else {  // padding_x_rem == 0 && padding_y_rem == 0
    permutation = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1},
                   {1, 3}, {2, 2}, {2, 3}, {3, 3}};
  }
  c += "  for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  if (is_wave_memory_upload) {
    c += "    Type src0 = " + read_src(0, 0);
    c += "    Type src1 = " + read_src(1, 0);
    c += "    Type src2 = " + read_src(0, 1);
    c += "    Type src3 = " + read_src(1, 1);
    auto conv_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        const std::string r_name = "r" + std::to_string(permutation[i].first);
        const std::string s_name =
            "src" + std::to_string(permutation[i].second);
        const int w_id = i - start;
        c += GenerateConv(precision, is_i4o4, r_name, s_name, w_id);
      }
    };
    if (!gpu_info.IsAdreno()) {
      c += "    ucl::SyncThreads<WaveLoad>();\n";
    }
    c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), f_offset, "
         "24);\n";
    c += "    f_offset += 24;\n";
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    conv_in_range(0, 6);
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), f_offset, "
         "12);\n";
    c += "    f_offset += 12;\n";
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    conv_in_range(6, 9);
  } else {
    if (need_local_mem) {
      c += "    " + local_mem_barrier + ";\n";
    }
    if (weights_upload_type == WeightsUploadType::kLocalMemoryWorkGroupLoad) {
      c += "    async_work_group_copy(weights_cache, "
           "args.weights.GetPtr(f_offset), 36, "
           "0);\n";
    } else if (weights_upload_type ==
               WeightsUploadType::kLocalMemoryByThreads) {
      c += "    weights_cache[local_id] = args.weights.Read(f_offset + "
           "local_id);\n";
      c += "    if (local_id < 4) {\n";
      c += "      weights_cache[local_id + 32] = args.weights.Read(f_offset + "
           "local_id + "
           "32);\n";
      c += "    };\n";
    } else {  // GLOBAL_MEM/CONSTANT_MEM
      // TODO: Add a GLOBAL_MEM/CONSTANT_MEM version compatible with APIs like
      // WebGPU. For now, we just log an error rather than weight for WGSL
      // compiler to complain.
      if (!gpu_info.SupportsPointersInKernels()) {
        ABSL_LOG(ERROR)
            << "ConvolutionTransposed3x3 operation currently only "
            << "supports GLOBAL_MEM and CONSTANT_MEM weight uploads "
            << "for OpenCL and Metal.";
      }
      c += "    " + weights_space +
           " Type* weights_cache = args.weights.GetPtr(f_offset);\n";
    }
    c += "    Type src0 = " + read_src(0, 0);
    c += "    Type src1 = " + read_src(1, 0);
    c += "    Type src2 = " + read_src(0, 1);
    c += "    Type src3 = " + read_src(1, 1);
    c += "    f_offset += 36;\n";
    if (need_local_mem) {
      c += "    " + local_mem_barrier + ";\n";
    }
    for (int i = 0; i < 9; ++i) {
      const std::string r_name = "r" + std::to_string(permutation[i].first);
      const std::string s_name = "src" + std::to_string(permutation[i].second);
      const std::string w_name = std::to_string(i * 4);
      c += GenerateConv(precision, is_i4o4, r_name, s_name, i);
    }
  }
  c += "  }\n";
  if (late_spatial_oob_check) {
    c += "  if (DST_X >= args.dst_tensor.Width() || DST_Y >= "
         "args.dst_tensor.Height()) return;\n";
  }
  if (has_bias) {
    c += "  Type bias_val = args.biases.Read(Z);\n";
  } else {
    c += "  Type bias_val = ucl::Init<Type>(0.0f);\n";
  }
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
      const std::string s_x = std::to_string(x);
      const std::string s_y = std::to_string(y);
      const std::string id = std::to_string(y * 2 + x);
      const std::string x_c = "DST_X + " + s_x;
      const std::string y_c = "DST_Y + " + s_y;
      c += "  if (" + x_c + " < args.dst_tensor.Width() && " + y_c +
           " < args.dst_tensor.Height()) {\n";
      c += "    Type res0 = ucl::Convert<Type>(r" + id + ") + bias_val;\n";
      c += "    args.dst_tensor.Write(res0, " + x_c + ", " + y_c + ", Z);\n";
      c += "  }\n";
    }
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

std::vector<int3> ConvolutionTransposed3x3::GetPossibleKernelWorkGroups(
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

int3 ConvolutionTransposed3x3::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width(), 2) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int> ConvolutionTransposed3x3::GetSpatialWeightsRemap() const {
  const int padding_x_rem = abs(padding_.x) % 2;
  const int padding_y_rem = abs(padding_.y) % 2;

  std::vector<int> remap;
  if (padding_x_rem == 1 && padding_y_rem == 1) {
    return std::vector<int>{4, 5, 3, 7, 1, 8, 6, 2, 0};
  } else if (padding_x_rem == 0 && padding_y_rem == 1) {
    return std::vector<int>{5, 3, 4, 8, 6, 2, 0, 7, 1};
  } else if (padding_x_rem == 1 && padding_y_rem == 0) {
    return std::vector<int>{7, 1, 8, 6, 2, 0, 4, 5, 3};
  } else {  // padding_x_rem == 0 && padding_y_rem == 0
    return std::vector<int>{8, 6, 2, 0, 7, 1, 5, 3, 4};
  }
}

void ConvolutionTransposed3x3::UploadWeights(
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

bool IsConvolutionTransposed3x3Supported(
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
         attr.stride.w == 2 && attr.stride.h == 2;
}

ConvolutionTransposed3x3 CreateConvolutionTransposed3x3(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  const int2 padding = int2(attr.padding.prepended.w, attr.padding.prepended.h);
  bool has_bias = !attr.bias.data.empty();
  ConvolutionTransposed3x3 result(definition, precision, gpu_info, padding,
                                  has_bias);
  result.UploadWeights(gpu_info, attr.weights);

  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

ConvolutionTransposed3x3 CreateConvolutionTransposed3x3DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  const int2 padding = int2(attr.padding.prepended.w, attr.padding.prepended.h);
  bool has_bias = !attr.bias.data.empty();
  ConvolutionTransposed3x3 result(definition, precision, gpu_info, padding,
                                  has_bias);
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
