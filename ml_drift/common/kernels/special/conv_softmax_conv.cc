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

#include "ml_drift/common/kernels/special/conv_softmax_conv.h"

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
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/wave_memory_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
bool IsSupportedMali(const GpuInfo& gpu_info) {
  return gpu_info.IsMali() && gpu_info.mali_info.IsValhallGen4();
}

std::string GenerateConv(const GpuInfo& gpu_info,
                         CalculationsPrecision precision,
                         const std::string& dst, const std::string& src,
                         int index, bool dot_conv) {
  std::string w0 = "weights_cache[" + std::to_string(index * 4 + 0) + "]";
  std::string w1 = "weights_cache[" + std::to_string(index * 4 + 1) + "]";
  std::string w2 = "weights_cache[" + std::to_string(index * 4 + 2) + "]";
  std::string w3 = "weights_cache[" + std::to_string(index * 4 + 3) + "]";
  std::string code;
  const bool use_fma = gpu_info.IsAMD() && gpu_info.IsApiOpenCl();
  switch (precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      if (dot_conv) {
        code += "    $0.x += dot($1, $2);\n";
        code += "    $0.y += dot($1, $3);\n";
        code += "    $0.z += dot($1, $4);\n";
        code += "    $0.w += dot($1, $5);\n";
      } else if (use_fma) {
        code += "    $0 = fma($1.x, $2, $0);\n";
        code += "    $0 = fma($1.y, $3, $0);\n";
        code += "    $0 = fma($1.z, $4, $0);\n";
        code += "    $0 = fma($1.w, $5, $0);\n";
      } else {
        code += "    $0 += $1.x * $2;\n";
        code += "    $0 += $1.y * $3;\n";
        code += "    $0 += $1.z * $4;\n";
        code += "    $0 += $1.w * $5;\n";
      }
      return absl::Substitute(code, dst, src, w0, w1, w2, w3);
    case CalculationsPrecision::F32_F16:
      if (dot_conv) {
        code += "    $0.x += dot($1, $2);\n";
        code += "    $0.y += dot($1, $3);\n";
        code += "    $0.z += dot($1, $4);\n";
        code += "    $0.w += dot($1, $5);\n";
      } else {
        code +=
            "    $0 += ucl::Convert<AccType>($1.x * $2 + $1.y * $2 + $1.z * $2 "
            "+ $1.w "
            "* $2);\n";
      }
      return absl::Substitute(code, dst, src, w0, w1, w2, w3);
  }
}

std::string GenerateLoad(const ConvSoftmaxConv::Params& params, bool first) {
  std::string c;
  const int weights_vals = params.block_slices_out * 4;
  const std::string offset_name = first ? "f_offset_0" : "f_offset_1";
  const std::string buffer_name = first ? "weights_0" : "weights_1";
  const std::string offset_stride =
      std::to_string(weights_vals * (first ? params.block_slices_size : 1));
  if (params.weights_upload_type ==
      ConvSoftmaxConv::WeightsUploadType::kWaveMemory) {
    c += "    ucl::SyncThreads<WaveLoad>();\n";
    c += "    ucl::WaveLoad(weights_cache, args." + buffer_name +
         ".GetPtr(), " + offset_name + ", " + std::to_string(weights_vals) +
         ");\n";
    c += "    ucl::SyncThreads<WaveLoad>();\n";
  } else if (params.weights_upload_type ==
             ConvSoftmaxConv::WeightsUploadType::kGlobalToLocalByThreads) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    if (ucl::GetLocalId<0>() < " + std::to_string(weights_vals) +
         ") {\n";
    c += "      weights_cache[ucl::GetLocalId<0>()] = "
         "args." +
         buffer_name + ".Read(ucl::GetLocalId<0>() + " + offset_name + ");\n";
    c += "    }\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  } else if (params.weights_upload_type ==
             ConvSoftmaxConv::WeightsUploadType::kGlobal) {
    c += "    weights_cache = args." + buffer_name + ".GetPtr(" + offset_name +
         ");\n";
  }
  c += "    " + offset_name + " += " + offset_stride + ";\n";
  return c;
}

std::string GenerateCode(const GpuInfo& gpu_info, const OperationDef& op_def,
                         CalculationsPrecision precision,
                         const ConvSoftmaxConv::Params& params,
                         const int3* wg_size) {
  std::string c;
  if (params.weights_upload_type ==
      ConvSoftmaxConv::WeightsUploadType::kWaveMemory) {
    c += "#pragma OPENCL EXTENSION ucl_wave_memory: enable\n";
    if (gpu_info.IsIntel()) {
      c += "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size: enable\n";
    }
  }
  const bool use_intel_wave_matmul =
      params.weights_upload_type ==
      ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul;
  if (use_intel_wave_matmul) {
    c += "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size: enable\n";
    c += "#pragma OPENCL EXTENSION "
         "cl_intel_subgroup_matrix_multiply_accumulate : enable\n";
  }
  const int weights_vals = params.block_slices_out * 4;
  const bool work_group_src_reduce = params.block_slices_size != 1;
  const bool late_xy_exit =
      params.weights_upload_type ==
          ConvSoftmaxConv::WeightsUploadType::kGlobalToLocalByThreads ||
      work_group_src_reduce ||
      (params.weights_upload_type ==
           ConvSoftmaxConv::WeightsUploadType::kWaveMemory &&
       !gpu_info.IsAdreno());
  if (use_intel_wave_matmul) {
    c += "__attribute__((intel_reqd_sub_group_size(" +
         std::to_string(params.wave_size) + ")))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = ucl::GetGlobalId<0>();\n";
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int S = ucl::GetGlobalId<2>();\n";
  if (!late_xy_exit) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| S * " +
         std::to_string(params.block_slices_out) +
         " >= args.dst_tensor.Slices()) return;\n";
  } else {
    c += "  if (S * " + std::to_string(params.block_slices_out) +
         " >= args.dst_tensor.Slices()) return;\n";
    c += "  bool xy_oob = X >= args.dst_tensor.Width() || Y >= "
         "args.dst_tensor.Height();\n";
    c += "  X = min(X, args.dst_tensor.Width() - 1);\n";
    c += "  Y = min(Y, args.dst_tensor.Height() - 1);\n";
  }
  c += "\n";
  for (int x = 0; x < params.block_x; ++x) {
    for (int s = 0; s < params.block_slices_out; ++s) {
      std::string val_name =
          "r_x" + std::to_string(x) + "_s" + std::to_string(s);
      c += "  AccType " + val_name + " = ucl::Init<AccType>(0.0f);\n";
    }
  }
  c += "\n";
  c += "  int f_offset_0 = Y * " +
       std::to_string(weights_vals * params.block_slices_size) +
       " * args.interm_slices + S * " + std::to_string(weights_vals) + ";\n";
  c += "  int f_offset_1 = (S * args.src_tensor.Height() + Y) * " +
       std::to_string(weights_vals) +
       " * "
       "args.interm_slices;\n";
  c += "\n";
  if (params.weights_upload_type ==
      ConvSoftmaxConv::WeightsUploadType::kWaveMemory) {
    c += "  __wave Type weights_cache[" + std::to_string(weights_vals) + "];\n";
  } else if (params.weights_upload_type ==
             ConvSoftmaxConv::WeightsUploadType::kGlobalToLocalByThreads) {
    c +=
        "  __local Type weights_cache[" + std::to_string(weights_vals) + "];\n";
  } else if (params.weights_upload_type ==
             ConvSoftmaxConv::WeightsUploadType::kGlobal) {
    c += "  __global Type* weights_cache;\n";
  }
  if (work_group_src_reduce) {
    c += "  __local Type interm_cache[" + std::to_string(wg_size->z) + "][" +
         std::to_string(wg_size->x * params.block_x) + "];\n";
  }
  c += "\n";
  if (params.block_x != 1) {
    c += "  int x_coord;\n";
  }
  if (op_def.src_tensors[0].IsLinear()) {
    for (int x = 0; x < params.block_x; ++x) {
      std::string x_coord = "X";
      if (params.block_x != 1) {
        x_coord = "x_coord";
        c += "  x_coord = min(X * " + std::to_string(params.block_x) + " + " +
             std::to_string(x) + ", args.src_tensor.Width() - 1);\n";
      }
      c += "  int addr_x" + std::to_string(x) +
           " = args.src_tensor.GetAddress(" + x_coord + ", Y, S * " +
           std::to_string(params.block_slices_out) + ");\n";
    }
    c += "  int dz = args.src_tensor.SliceStride();\n";
  }
  std::string x_coord = "X";
  auto read_src = [&](int x_index, int s_index) {
    if (op_def.src_tensors[0].IsLinear()) {
      return std::string("args.src_tensor.Read(addr_x" +
                         std::to_string(x_index) + "); addr_x" +
                         std::to_string(x_index) + " += dz;");
    } else {
      return std::string("args.src_tensor.Read(" + x_coord + ", Y, S * " +
                         std::to_string(params.block_slices_out) + " + " +
                         std::to_string(s_index) + ");");
    }
  };
  for (int x = 0; x < params.block_x; ++x) {
    if (params.block_x != 1) {
      x_coord = "x_coord";
      c += "  x_coord = min(X * " + std::to_string(params.block_x) + " + " +
           std::to_string(x) + ", args.src_tensor.Width() - 1);\n";
    }
    for (int s = 0; s < params.block_slices_in; ++s) {
      std::string val_name =
          "src_x" + std::to_string(x) + "_s" + std::to_string(s);
      c += "  Type " + val_name + " = " + read_src(x, s) + "\n";
    }
  }
  bool softmax = true;
  if (softmax) {
    for (int x = 0; x < params.block_x; ++x) {
      c += "  SType exp_sum_adj_x" + std::to_string(x) + " = 0.0f;\n";
      c += "  SType max_val_x" + std::to_string(x) + " = -65000.0f;\n";
    }
  }
  if (use_intel_wave_matmul) {
    c += "  int8 S0, S1, S2;\n";
    c += "  S0.s01 = as_int2(src_x0_s0);\n";
    c += "  S0.s23 = as_int2(src_x0_s1);\n";
    c += "  S0.s45 = as_int2(src_x0_s2);\n";
    c += "  S0.s67 = as_int2(src_x0_s3);\n";
    c += "  S1.s01 = as_int2(src_x0_s4);\n";
    c += "  S1.s23 = as_int2(src_x0_s5);\n";
    c += "  S1.s45 = as_int2(src_x0_s6);\n";
    c += "  S1.s67 = as_int2(src_x0_s7);\n";
    c += "  S2.s01 = as_int2(src_x0_s8);\n";
    c += "  S2.s23 = as_int2(src_x0_s9);\n";
    c += "  S2.s45 = 0;\n";
    c += "  S2.s67 = 0;\n";
    c += "  int simd_id_0 = ucl::GetSubGroupLocalId();\n";
    c += "  __global short4* wptr1 = (__global "
         "short4*)(args.weights_1.GetPtr(f_offset_1));\n";
  }
  c += "  int coord_s = 0;\n";
  c += "  do {\n";
  for (int interm_s = 0; interm_s < params.block_interm_slices; ++interm_s) {
    for (int x = 0; x < params.block_x; ++x) {
      const std::string value_name =
          "interm_x" + std::to_string(x) + "_s" + std::to_string(interm_s);
      c += "    Type " + value_name + " = ucl::Init<Type>(0.0f);\n";
    }
  }
  if (use_intel_wave_matmul) {
    const int s16_count = DivideRoundUp(params.block_slices_in, 4);
    c += "    __global short4* wptr0 = (__global "
         "short4*)(args.weights_0.GetPtr(((coord_s / 4) * "
         "args.src_tensor.Height() + Y) * " +
         std::to_string(s16_count * params.block_interm_slices * 16) + "));\n";
    for (int s = 0; s < s16_count; ++s) {
      for (int interm_s = 0; interm_s < params.block_interm_slices;
           ++interm_s) {
        const std::string value_name = "interm_x0_s" + std::to_string(interm_s);
        c += "    " + value_name +
             " = intel_sub_group_f16_f16_matrix_mad_k16(wptr0[simd_id_0 + " +
             std::to_string((s * params.block_interm_slices + interm_s) * 16) +
             "], S" + std::to_string(s) + ", " + value_name + ");\n";
      }
    }
  }
  for (int interm_s = 0; interm_s < params.block_interm_slices; ++interm_s) {
    if (use_intel_wave_matmul) {
      break;
    }
    c += GenerateLoad(params, /*first=*/true);
    for (int src_s = 0; src_s < params.block_slices_out; ++src_s) {
      for (int x = 0; x < params.block_x; ++x) {
        std::string src_name =
            "src_x" + std::to_string(x) + "_s" + std::to_string(src_s);
        const std::string dst_name =
            "interm_x" + std::to_string(x) + "_s" + std::to_string(interm_s);
        c += GenerateConv(gpu_info, precision, dst_name, src_name, src_s,
                          params.dot_conv);
      }
    }
    if (work_group_src_reduce) {
      c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
      for (int x = 0; x < params.block_x; ++x) {
        const std::string val_name =
            "interm_x" + std::to_string(x) + "_s" + std::to_string(interm_s);
        c += "    interm_cache[ucl::GetLocalId<2>()][ucl::GetLocalId<0>() + " +
             std::to_string(x * wg_size->x) + "] = " + val_name + ";\n";
      }
      c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
      for (int x = 0; x < params.block_x; ++x) {
        const std::string val_name =
            "interm_x" + std::to_string(x) + "_s" + std::to_string(interm_s);
        c += "    " + val_name + " = interm_cache[0][ucl::GetLocalId<0>() + " +
             std::to_string(x * wg_size->x) + "];\n";
      }
      for (int i = 1; i < params.block_slices_size; ++i) {
        for (int x = 0; x < params.block_x; ++x) {
          const std::string val_name =
              "interm_x" + std::to_string(x) + "_s" + std::to_string(interm_s);
          c += "    " + val_name + " += interm_cache[" + std::to_string(i) +
               "][ucl::GetLocalId<0>() + " + std::to_string(x * wg_size->x) +
               "];\n";
        }
      }
    }
  }
  if (softmax) {
    for (int x = 0; x < params.block_x; ++x) {
      const std::string max_name = "max_val_x" + std::to_string(x);
      const std::string exp_sum_name = "exp_sum_adj_x" + std::to_string(x);
      c += "    {  // softmax block\n";
      c += "    Type temp_max = interm_x" + std::to_string(x) + "_s0;\n";
      for (int s = 1; s < params.block_interm_slices; ++s) {
        const std::string src_name =
            "interm_x" + std::to_string(x) + "_s" + std::to_string(s);
        c += "    temp_max = max(temp_max, " + src_name + ");\n";
      }
      c += "    SType interm_max = max(" + max_name + ", temp_max.x);\n";
      c += "    interm_max = max(interm_max, temp_max.y);\n";
      c += "    interm_max = max(interm_max, temp_max.z);\n";
      c += "    interm_max = max(interm_max, temp_max.w);\n";
      c +=
          "    SType scale = ucl::Exp<SType>(" + max_name + " - interm_max);\n";
      c += "    " + max_name + " = interm_max;\n";
      for (int s = 0; s < params.block_interm_slices; ++s) {
        const std::string src_name =
            "interm_x" + std::to_string(x) + "_s" + std::to_string(s);
        c += "    " + src_name + " = ucl::Exp<Type>(" + src_name +
             " - interm_max);\n";
      }
      c += "    Type temp_sum = interm_x" + std::to_string(x) + "_s0;\n";
      for (int s = 1; s < params.block_interm_slices; ++s) {
        const std::string src_name =
            "interm_x" + std::to_string(x) + "_s" + std::to_string(s);
        c += "    temp_sum += " + src_name + ";\n";
      }
      c += "    SType interm_sum = temp_sum.x;\n";
      c += "    interm_sum += temp_sum.y;\n";
      c += "    interm_sum += temp_sum.z;\n";
      c += "    interm_sum += temp_sum.w;\n";
      c += "    " + exp_sum_name + " = " + exp_sum_name +
           " * scale + interm_sum;\n";
      for (int s = 0; s < params.block_slices_out; ++s) {
        std::string val_name =
            "r_x" + std::to_string(x) + "_s" + std::to_string(s);
        c += "    " + val_name + " *= scale;\n";
      }
      c += "    }\n";
    }
  }
  if (use_intel_wave_matmul) {
    c += "    int8 S3;\n";
    c += "    S3.s01 = as_int2(interm_x0_s0);\n";
    c += "    S3.s23 = as_int2(interm_x0_s1);\n";
    c += "    S3.s45 = as_int2(interm_x0_s2);\n";
    c += "    S3.s67 = as_int2(interm_x0_s3);\n";
    for (int s = 0; s < params.block_slices_out; ++s) {
      const std::string value_name = "r_x0_s" + std::to_string(s);
      c += "    " + value_name +
           " = intel_sub_group_f16_f16_matrix_mad_k16(wptr1[simd_id_0 + " +
           std::to_string(s * 16) + "], S3, " + value_name + ");\n";
    }
    c += "    wptr1 += " + std::to_string(params.block_slices_out * 16) + ";\n";
  }
  for (int interm_s = 0; interm_s < params.block_interm_slices; ++interm_s) {
    if (use_intel_wave_matmul) {
      break;
    }
    c += GenerateLoad(params, /*first=*/false);
    for (int s_out = 0; s_out < params.block_slices_out; ++s_out) {
      for (int s_x = 0; s_x < params.block_x; ++s_x) {
        const std::string src_name =
            "interm_x" + std::to_string(s_x) + "_s" + std::to_string(interm_s);
        const std::string dst_name =
            "r_x" + std::to_string(s_x) + "_s" + std::to_string(s_out);
        c += GenerateConv(gpu_info, precision, dst_name, src_name, s_out,
                          params.dot_conv);
      }
    }
  }
  c += "    coord_s += " + std::to_string(params.block_interm_slices) + ";\n";
  c += "  } while (coord_s < args.interm_slices);\n";
  if (late_xy_exit) {
    c += "  if (xy_oob) return;\n";
  }
  if (softmax) {
    for (int x = 0; x < params.block_x; ++x) {
      for (int s = 0; s < params.block_slices_out; ++s) {
        std::string val_name =
            "r_x" + std::to_string(x) + "_s" + std::to_string(s);
        c += "  " + val_name + " /= exp_sum_adj_x" + std::to_string(x) + ";\n";
      }
    }
  }
  c += "\n";
  c += "  S *= " + std::to_string(params.block_slices_out) + ";\n";
  if (params.block_x != 1) {
    c += "  X *= " + std::to_string(params.block_x) + ";\n";
  }
  for (int s = 0; s < params.block_slices_out; ++s) {
    for (int x = 0; x < params.block_x; ++x) {
      std::string val_name =
          "r_x" + std::to_string(x) + "_s" + std::to_string(s);
      std::string check =
          "S + " + std::to_string(s) + " < args.dst_tensor.Slices()";
      if (params.block_x != 1 && x != 0) {
        check += " && X + " + std::to_string(x) + " < args.dst_tensor.Width()";
      }
      c += "  if (" + check + ") {\n";
      c += "    Type res = ucl::Convert<Type>(" + val_name + ");\n";
      c += "    args.dst_tensor.Write(res, X + " + std::to_string(x) +
           ", Y, S + " + std::to_string(s) + ");\n";
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
}  // namespace

ConvSoftmaxConv::ConvSoftmaxConv(CalculationsPrecision precision,
                                 const ConvSoftmaxConv::Params& params)
    : params_(params) {}

int3 ConvSoftmaxConv::GetGridSize() const {
  const int grid_x =
      DivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(), params_.block_x);
  const int grid_y = dst_[0]->Height();
  const int grid_z = DivideRoundUp(dst_[0]->Slices(), params_.block_slices_out);
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> ConvSoftmaxConv::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (params_.weights_upload_type ==
          ConvSoftmaxConv::WeightsUploadType::kGlobalToLocalByThreads ||
      params_.block_slices_size != 1) {
    return {work_group_size_};
  }
  if (gpu_info.IsApple()) {
    return {work_group_size_};
  }
  const bool wave_granularity =
      params_.weights_upload_type ==
          ConvSoftmaxConv::WeightsUploadType::kWaveMemory ||
      params_.weights_upload_type ==
          ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul;
  switch (tuning_type) {
    case TuningType::kExhaustive:
      if (wave_granularity) {
        return GetWorkGroupsXMultipleOf(params_.wave_size, gpu_info,
                                        kernel_info, grid_size_);
      } else {
        return {{32, 1, 1}};
      }
    case TuningType::kFast:
    default:
      if (wave_granularity) {
        if (gpu_info.IsAdreno()) {
          return {{128, 4, 1}};
        } else {
          return {{params_.wave_size, 1, 1}};
        }
      } else {
        return {{64, 1, 1}};
      }
  }
}

ConvSoftmaxConv::Params GetConvParams(const GpuInfo& gpu_info,
                                      CalculationsPrecision precision,
                                      int src_ch, int interm_ch, int dst_ch) {
  ConvSoftmaxConv::Params params;
  if (gpu_info.SupportsExtension("ucl_wave_memory")) {
    params.weights_upload_type =
        ConvSoftmaxConv::WeightsUploadType::kWaveMemory;
    if (gpu_info.IsAdreno()) {
      params.wave_size = 128;
    } else if (gpu_info.IsIntel()) {
      params.wave_size = 32;
    } else if (gpu_info.IsPowerVR()) {
      params.wave_size = 128;
    }
  } else {
    params.weights_upload_type = ConvSoftmaxConv::WeightsUploadType::kGlobal;
  }
  params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  params.dot_conv = gpu_info.IsDotPreferred();
  const int src_slices = DivideRoundUp(src_ch, 4);
  const int interm_slices = DivideRoundUp(interm_ch, 4);
  const int dst_slices = DivideRoundUp(dst_ch, 4);
  params.block_slices_in = src_slices;
  params.block_slices_out = dst_slices;
  params.block_slices_size = 1;
  params.block_x = 1;
  if (IsSupportedMali(gpu_info) || gpu_info.IsAMD()) {
    params.block_x = 2;
  }
  if (IsSupportedMali(gpu_info) || gpu_info.IsAMD() || gpu_info.IsApple()) {
    if (dst_slices % 2 == 0 && dst_slices >= 8) {
      params.block_slices_in = src_slices / 2;
      params.block_slices_out = dst_slices / 2;
      params.block_slices_size = 2;
    }
  }
  params.block_interm_slices = 1;
  if (gpu_info.IsAdreno() && interm_slices % 2 == 0) {
    params.block_interm_slices = 2;
  }
  if (gpu_info.SupportsExtension(
          "cl_intel_subgroup_matrix_multiply_accumulate") &&
      gpu_info.SupportsExtension("cl_intel_required_subgroup_size") &&
      gpu_info.SupportsSubGroupWithSize(16) &&
      !gpu_info.SupportsSubGroupWithSize(8) && interm_slices % 4 == 0) {
    params.block_interm_slices = 4;
    params.weights_upload_type =
        ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul;
    params.wave_size = 16;
  }
  return params;
}

std::vector<WeightsDescription> GetWeightsDescsForConvSoftmaxConv(
    const GpuInfo& gpu_info, CalculationsPrecision precision, int src_ch,
    int interm_ch, int dst_ch) {
  ConvSoftmaxConv::Params params =
      GetConvParams(gpu_info, precision, src_ch, interm_ch, dst_ch);

  std::vector<WeightsDescription> result(2);
  {
    WeightsDescription& desc = result[0];
    desc.type = params.weights_data_type;
    desc.layout = WeightsLayout::kCustomGroups;
    if (params.weights_upload_type ==
        ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul) {
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
      desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 16});
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
    } else if (params.dot_conv) {
      desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 4});
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
    } else {
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
      desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 4});
    }
    desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 0});
    if (params.weights_upload_type !=
        ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul) {
          desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 0});
    }
    desc.group_sizes.push_back({Axis::WIDTH, 0});
    desc.group_sizes.push_back({Axis::HEIGHT, 0});
    if (params.weights_upload_type ==
        ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul) {
          desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 0});
    }
  }
  {
    WeightsDescription& desc = result[1];
    desc.type = params.weights_data_type;
    desc.layout = WeightsLayout::kCustomGroups;
    if (params.weights_upload_type ==
        ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul) {
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
      desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 16});
    } else if (params.dot_conv) {
      desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 4});
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
    } else {
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
      desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 4});
    }
    desc.group_sizes.push_back(
        {Axis::OUTPUT_CHANNELS, params.block_slices_out});
    desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 0});
    desc.group_sizes.push_back({Axis::WIDTH, 0});
    desc.group_sizes.push_back({Axis::HEIGHT, 0});
    desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 0});
  }
  return result;
}

bool IsConvSoftmaxConvSupported(const GpuInfo& gpu_info,
                                CalculationsPrecision precision, int src_ch,
                                int interm_ch, int dst_ch) {
  bool supported_gpu =
      gpu_info.IsAdreno() && gpu_info.SupportsExtension("ucl_wave_memory");
  supported_gpu =
      supported_gpu || (gpu_info.IsApiMetal() && gpu_info.IsApple() &&
                        gpu_info.apple_info.IsBionic());
  supported_gpu = supported_gpu || (gpu_info.IsApiOpenCl() && gpu_info.IsAMD());
  supported_gpu =
      supported_gpu || (gpu_info.IsApiOpenCl() && IsSupportedMali(gpu_info));
  if (gpu_info.IsIntel()) {
    const bool is_battlemage = !gpu_info.SupportsSubGroupWithSize(8);
    if (is_battlemage) {
      supported_gpu = true;
    }
  }
  const int src_slices = DivideRoundUp(src_ch, 4);
  const int dst_slices = DivideRoundUp(dst_ch, 4);
  return supported_gpu && precision == CalculationsPrecision::F16 &&
         src_slices == dst_slices && dst_slices <= 10 && interm_ch % 4 == 0;
}

ConvSoftmaxConv CreateConvSoftmaxConv(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvSoftmaxConv::WeightsDesc& weights_desc) {
  int src_ch, interm_ch, dst_ch;
  if (weights_desc.constant) {
    src_ch = weights_desc.weights0->shape.i;
    interm_ch = weights_desc.weights0->shape.o;
    dst_ch = weights_desc.weights1->shape.o;
  } else {
    src_ch = weights_desc.src_ch;
    interm_ch = weights_desc.interm_ch;
    dst_ch = weights_desc.dst_ch;
  }
  ConvSoftmaxConv::Params params =
      GetConvParams(gpu_info, precision, src_ch, interm_ch, dst_ch);

  ConvSoftmaxConv result(precision, params);
  result.args_.AddInt("interm_slices", DivideRoundUp(interm_ch, 4));

  result.work_group_size_ = {64, 1, 1};
  if (gpu_info.IsApple()) {
    result.work_group_size_ = {32, 1, 1};
  }

  if (params.weights_upload_type ==
          ConvSoftmaxConv::WeightsUploadType::kWaveMemory ||
      params.weights_upload_type ==
          ConvSoftmaxConv::WeightsUploadType::kIntelWaveMatmul) {
    result.work_group_size_ = {params.wave_size, 1, 1};
  }

  result.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);

  if (params.block_slices_size != 1) {
    result.work_group_size_.z = params.block_slices_size;
    if (gpu_info.IsApple()) {
      result.work_group_size_.x =
          gpu_info.apple_info.IsFamilyOrLower(AppleInfo::Family::kApple8) ? 16
                                                                          : 32;
    } else if (gpu_info.IsMali()) {
      result.work_group_size_.x = 16;
    } else if (gpu_info.IsAMD()) {
      result.work_group_size_.x = 32;
    }
  }

  result.AddSrcTensor("src_tensor", definition.src_tensors[0]);

  if (weights_desc.constant) {
    std::vector<WeightsDescription> descs = GetWeightsDescsForConvSoftmaxConv(
        gpu_info, precision, src_ch, interm_ch, dst_ch);
    {
      BufferDescriptor buffer_desc = GetBufferDescForWaveMemoryUpload(
          gpu_info, descs[0], weights_desc.weights0->shape);
      RearrangeWeights(*weights_desc.weights0, descs[0],
                       absl::MakeSpan(buffer_desc.data));
      result.args_.AddObject("weights_0", std::make_unique<BufferDescriptor>(
                                              std::move(buffer_desc)));
    }
    {
      BufferDescriptor buffer_desc = GetBufferDescForWaveMemoryUpload(
          gpu_info, descs[1], weights_desc.weights1->shape);
      RearrangeWeights(*weights_desc.weights1, descs[1],
                       absl::MakeSpan(buffer_desc.data));
      result.args_.AddObject("weights_1", std::make_unique<BufferDescriptor>(
                                              std::move(buffer_desc)));
    }
  } else {
    result.AddSrcBuffer("weights_0", params.GetWeightsBufferDesc(gpu_info));
    result.AddSrcBuffer("weights_1", params.GetWeightsBufferDesc(gpu_info));
  }

  result.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  result.code_ = GenerateCode(gpu_info, definition, precision, result.params_,
                              &result.work_group_size_);
  result.work_group_launch_order_ = int3(0, 1, 2);
  return result;
}

}  // namespace ml_drift
