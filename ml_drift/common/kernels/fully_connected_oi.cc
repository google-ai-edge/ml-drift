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

#include "ml_drift/common/kernels/fully_connected_oi.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/fully_connected_util.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string AddBatchOffset(const std::string& stride) {
  return absl::Substitute(R"(
    a0 += w_batch_id * $0;
    a1 += w_batch_id * $0;
    a2 += w_batch_id * $0;
    a3 += w_batch_id * $0;
)",
                          stride);
}

std::string GetRingedOAddresses(bool batched_weights) {
  std::string c;
  // kBOIo4i4
  c += R"(
    int o0 = (dst_s * 4 + ring_o_offset) % ring_size;
    int o1 = (dst_s * 4 + 1 + ring_o_offset) % ring_size;
    int o2 = (dst_s * 4 + 2 + ring_o_offset) % ring_size;
    int o3 = (dst_s * 4 + 3 + ring_o_offset) % ring_size;
    int a0 = ((o0 / 4) * args.src_tensor.Slices() + src_s) * 4 + o0 % 4;
    int a1 = ((o1 / 4) * args.src_tensor.Slices() + src_s) * 4 + o1 % 4;
    int a2 = ((o2 / 4) * args.src_tensor.Slices() + src_s) * 4 + o2 % 4;
    int a3 = ((o3 / 4) * args.src_tensor.Slices() + src_s) * 4 + o3 % 4;
)";
  if (batched_weights) {
    c += AddBatchOffset("ring_size * args.src_tensor.Slices()");
  }
  return c;
}

std::string GetRingedIAddresses(bool batched_weights) {
  std::string c;
  // kBOIi4o4
  c += R"(
    int i0 = (src_s * 4 + ring_i_offset) % ring_size;
    int i1 = (src_s * 4 + 1 + ring_i_offset) % ring_size;
    int i2 = (src_s * 4 + 2 + ring_i_offset) % ring_size;
    int i3 = (src_s * 4 + 3 + ring_i_offset) % ring_size;
    int a0 = dst_s * ring_size + i0;
    int a1 = dst_s * ring_size + i1;
    int a2 = dst_s * ring_size + i2;
    int a3 = dst_s * ring_size + i3;
)";
  if (batched_weights) {
    c += AddBatchOffset("ring_size * args.dst_tensor.Slices()");
  }
  return c;
}

// We are reading ringed weights as 4 vec4 elements with individual addresses.
std::string ReadRingedWeightsAsFloat(
    const FullyConnectedOI::ConvParams& conv_params,
    const WeightsDescription& weights_desc) {
  std::string c;
  if (conv_params.runtime_check.ring_o_offset_index.has_value()) {
    c += GetRingedOAddresses(conv_params.batched_weights);
  } else if (conv_params.runtime_check.ring_i_offset_index.has_value()) {
    c += GetRingedIAddresses(conv_params.batched_weights);
  }
  if (conv_params.weights_type == DataType::INT8) {
    c += R"(
    uint4 w;
    w.x = args.weights.Read(a0);
    w.y = args.weights.Read(a1);
    w.z = args.weights.Read(a2);
    w.w = args.weights.Read(a3);
    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);
  )";
  } else {
    c += R"(
    w0 = args.weights.Read(a0);
    w1 = args.weights.Read(a1);
    w2 = args.weights.Read(a2);
    w3 = args.weights.Read(a3);
  )";
  }
  return c;
}

std::string ReadWeightsAsFloat(const FullyConnectedOI::ConvParams& conv_params,
                               const WeightsDescription& weights_desc) {
  std::string c;
  if (conv_params.runtime_check.ring_o_offset_index.has_value() ||
      conv_params.runtime_check.ring_i_offset_index.has_value()) {
    return ReadRingedWeightsAsFloat(conv_params, weights_desc);
  }
  if (weights_desc.IsLinearLayout()) {
    const std::string slices_divisor =
        conv_params.src_n == 1 ? "" : " / " + std::to_string(conv_params.src_n);
    c += "    int linear_i4o4 = dst_s * args.src_tensor.Slices()" +
         slices_divisor + " + src_s;\n";
    if (conv_params.batched_weights) {
      c += "    linear_i4o4 += w_batch_id * args.src_tensor.Slices()" +
           slices_divisor +
           " * "
           "args.dst_tensor.Slices();\n";
    }
  }
  std::string x_c = "src_s";
  std::string y_c = "dst_s";
  if (conv_params.batched_weights) {
    y_c = "w_batch_id * args.dst_tensor.Slices() + dst_s";
  }
  std::string coords = x_c + ", " + y_c;
  if (conv_params.weights_type == DataType::FLOAT32 ||
      conv_params.weights_type == DataType::FLOAT16) {
    if (weights_desc.IsLinearLayout()) {
      c += "    args.weights.ReadVec16AsVec4x4(w0, w1, w2, w3, linear_i4o4);\n";
    } else {
      c += "    w0 = args.weights0.Read(" + coords + ");\n";
      c += "    w1 = args.weights1.Read(" + coords + ");\n";
      c += "    w2 = args.weights2.Read(" + coords + ");\n";
      c += "    w3 = args.weights3.Read(" + coords + ");\n";
    }
  } else if (conv_params.sparse_2x4) {
    c += "    uint2 w_ui2 = args.weights.Read(linear_i4o4);\n";
    c += "    uint wind = args.weights_indices.Read(linear_i4o4);\n";
    c += "    ushort2 wind_us2 = ucl::Reinterpret<uchar4, ushort2>(wind);\n";
    c += "    ucl::U32Sparse2x4ToU4x16AsVec4x4<SType>(w_ui2.x, wind_us2.x, w0, "
         "w1, w2, w3);\n";
  } else if (conv_params.weights_type == DataType::INT8) {
    if (weights_desc.IsLinearLayout()) {
      c += "    uint4 w = args.weights.Read(linear_i4o4);\n";
    } else {
      c += "    uint4 w = args.weights.Read(" + coords + ");\n";
    }
    c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
  } else if (conv_params.weights_type == DataType::INT4) {
    if (weights_desc.IsLinearLayout()) {
      if (conv_params.src_n == 2) {
        c += "    uint4 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w.xy, w0, w1, w2, w3);\n";
      } else {
        c += "    uint2 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      }
    } else {
      if (conv_params.src_n == 2) {
        c += "    uint4 w = args.weights.Read(" + coords + ");\n";
        c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w.xy, w0, w1, w2, w3);\n";
      } else {
        c += "    ushort4 w = args.weights.Read(" + coords + ");\n";
        c += "    ucl::U16x4ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      }
    }
  } else if (conv_params.weights_type == DataType::INT2) {
    if (weights_desc.IsLinearLayout()) {
      if (conv_params.src_n == 4) {
        c += "    uint4 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w.x, w0, w1, w2, w3);\n";
      } else if (conv_params.src_n == 2) {
        c += "    uint2 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w.x, w0, w1, w2, w3);\n";
      } else {
        c += "    uint w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      }
    } else {
      if (conv_params.src_n == 4) {
        c += "    uint4 w = args.weights.Read(" + coords + ");\n";
        c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w.x, w0, w1, w2, w3);\n";
      } else {
        c += "    uchar4 w = args.weights.Read(" + coords + ");\n";
        c += "    ucl::U8x4ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      }
    }
  }
  return c;
}

std::string DecodeWeightsAsFloat(
    const FullyConnectedOI::ConvParams& conv_params,
    const WeightsDescription& weights_desc, int src_id) {
  std::string c;
  if (conv_params.sparse_2x4) {
    c += "    ucl::U32Sparse2x4ToU4x16AsVec4x4<SType>(w_ui2.y, wind_us2.y, "
         "w0, w1, w2, w3);\n";
  }
  if (conv_params.weights_type == DataType::INT4 && !conv_params.sparse_2x4) {
    std::string postfixes[2] = {"xy", "zw"};
    c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w." + postfixes[src_id] +
         ", w0, w1, w2, w3);\n";
  }
  if (conv_params.weights_type == DataType::INT2 && !conv_params.sparse_2x4) {
    std::string postfixes[4] = {"x", "y", "z", "w"};
    c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w." + postfixes[src_id] +
         ", w0, w1, w2, w3);\n";
  }
  return c;
}

std::string GetFullyConnectedOIO4KernelCode(
    const GpuInfo& gpu_info, const FullyConnectedOI::ConvParams& conv_params,
    CalculationsPrecision precision, const WeightsDescription& weights_desc,
    const int3& wg_size, const TensorDescriptor& src_desc,
    int scale_zp_group_size) {
  const int block_spatial = conv_params.block_size.b *
                            conv_params.block_size.w * conv_params.block_size.h;

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int local_x = ucl::GetLocalId<0>();\n";
  c += "  int local_y = ucl::GetLocalId<1>();\n";
  c += "  int dst_s = ucl::GetGroupId<1>() * WG_SIZE_Y + local_y;\n";
  c += fc::GetActiveDstSlices(conv_params.runtime_check);
  // We require uniform execution for the entire workgroup since we use a
  // memory barrier below.
  c += "  int dst_s_wg_offset = ucl::GetGroupId<1>() * WG_SIZE_Y;\n";
  c += "  if (dst_s_wg_offset >= dst_end_slice) return;\n";
  if (conv_params.runtime_check.ring_o_offset_index.has_value()) {
    c += fc::GetRingOOffset(conv_params.runtime_check);
  }
  if (conv_params.runtime_check.ring_i_offset_index.has_value()) {
    c += fc::GetRingIOffset(conv_params.runtime_check);
  }
  if (conv_params.batched_weights) {
    c += fc::GetWeightsBatchId(conv_params.runtime_batch_ids);
  }
  if (conv_params.runtime_check.packed_groups.has_value()) {
    c += fc::GetPackedGroupsParams(conv_params.runtime_check, /*dim_id=*/0,
                                   conv_params.block_size.w);
  }

  for (int sp_id = 0; sp_id < block_spatial; ++sp_id) {
    const std::string r_name = "r_sp" + std::to_string(sp_id);
    c += "  AccType " + r_name + " = ucl::Init<AccType>(0.0f);\n";
  }
  if (conv_params.softmax_input_activation) {
    for (int sp_id = 0; sp_id < block_spatial; ++sp_id) {
      const int3 bhw = fc::GetBlockSpatialCoords(sp_id, conv_params.block_size);
      std::string y_coord = std::to_string(bhw.y);
      if (conv_params.batched_weights) {
        y_coord = "w_batch_id";
        if (conv_params.runtime_batch_ids) {
          y_coord = "0";
        }
      }
      c += "  SType2 exp_val" + std::to_string(sp_id) +
           " = args.src_exp.Read(" + std::to_string(bhw.z) + ", " + y_coord +
           ", 0, " + std::to_string(bhw.x) + ").xy;\n";
    }
  }
  c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
  if (SizeInBitsOf(conv_params.weights_type) <= 8) {
    c += "  Type w_scale;\n";
    c += "  Type w_bias;\n";
  }
  const bool blocked_quantization = fc::IsBlockQuantized(
      conv_params.weights_type, conv_params.scale_zp_shape);
  if (blocked_quantization) {
    c += "  int src_group = -1;\n";
    // reading later in loop
  } else if (fc::IsLinearQuantized(conv_params.weights_type,
                                   conv_params.scale_zp_shape)) {
    c += fc::ReadScaleZeroPointLinear("dst_s", conv_params.scale_zp_shape,
                                      conv_params.has_zero_point,
                                      conv_params.weights_type);
  } else if (fc::IsScalarQuantized(conv_params.weights_type,
                                   conv_params.scale_zp_shape)) {
    c += fc::ReadScaleZeroPointScalar(conv_params.has_zero_point,
                                      conv_params.weights_type);
  }
  const bool late_weights_scaling =
      fc::IsQuantized(conv_params.weights_type) && block_spatial == 1;
  const std::string slices_divisor =
      conv_params.src_n == 1 ? "" : " / " + std::to_string(conv_params.src_n);
  std::string src_end_slices = "args.src_tensor.Slices()";
  if (conv_params.runtime_check.src_end_ch_index.has_value()) {
    c += "  int src_end_slices = " +
         conv_params.runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.src_end_ch_index)",
             "args.src_tensor.Slices()") +
         ";\n";
    src_end_slices = "src_end_slices";
  }
  c += "  for (int src_s = local_x; src_s < " + src_end_slices +
       slices_divisor + ";  src_s += WG_SIZE_X) {\n";
  if (blocked_quantization) {
    c += "  int group_id = src_s / " + std::to_string(scale_zp_group_size) +
         ";\n";
    c += "  if (group_id != src_group) {\n";
    c += "    src_group = group_id;\n";
    c += fc::ReadScaleZeroPointBlock("dst_s", conv_params.scale_zp_shape,
                                     conv_params.has_zero_point,
                                     conv_params.weights_type);
    c += "  }\n";
  }
  for (int sp_id = 0; sp_id < block_spatial; ++sp_id) {
    const int3 bhw = fc::GetBlockSpatialCoords(sp_id, conv_params.block_size);
    std::string y_coord = std::to_string(bhw.y);
    if (conv_params.batched_weights) {
      y_coord = "w_batch_id";
      if (conv_params.runtime_batch_ids) {
        y_coord = "min(dst_h, args.src_tensor.Height() - 1)";
      }
    }
    std::string x_coord = std::to_string(bhw.z);
    if (conv_params.runtime_check.packed_groups.has_value()) {
      x_coord = "dst_w + " + x_coord;
      if (!src_desc.CanReadOutOfBorder(Axis::WIDTH, gpu_info)) {
        x_coord = "min(" + x_coord + ", args.src_tensor.Width() - 1)";
      }
      y_coord = "0";
    }
    for (int s_id = 0; s_id < conv_params.src_n; ++s_id) {
      const std::string src_name =
          "src_sp" + std::to_string(sp_id) + "_s" + std::to_string(s_id);
      c += "    Type " + src_name + " = args.src_tensor.Read(" + x_coord +
           ", " + y_coord + ", src_s * " + std::to_string(conv_params.src_n) +
           " + " + std::to_string(s_id) + ");\n";
      if (conv_params.softmax_input_activation) {
        const std::string exp_name = "exp_val" + std::to_string(sp_id);
        c += "  " + src_name + " = ucl::Exp<Type>(" + src_name + " - " +
             exp_name + ".y) * " + exp_name + ".x;\n";
      }
    }
  }
  c += "    Type w0, w1, w2, w3;\n";
  if (late_weights_scaling) {
    c += "    Type rl = ucl::Init<Type>(0.0f);\n";
    c += "    SType s_sum = ucl::Init<SType>(0.0f);\n";
  }
  c += ReadWeightsAsFloat(conv_params, weights_desc);
  const bool isI4O4 = weights_desc.IsI4O4() || conv_params.sparse_2x4;
  const bool use_fma = fc::UseFMA(gpu_info);
  for (int s_id = 0; s_id < conv_params.src_n; ++s_id) {
    if (s_id != 0) {
      c += DecodeWeightsAsFloat(conv_params, weights_desc, s_id);
    }
    if (fc::IsQuantized(conv_params.weights_type) && !late_weights_scaling) {
      const std::string w_scale = "w_scale";
      const std::string w_bias = "w_bias";
      c += fc::WeightsScaleAddBias(w_scale, w_bias, isI4O4, use_fma);
    }
    for (int sp_id = 0; sp_id < block_spatial; ++sp_id) {
      const std::string src_name =
          "src_sp" + std::to_string(sp_id) + "_s" + std::to_string(s_id);
      const std::string r_name =
          late_weights_scaling ? "rl" : "r_sp" + std::to_string(sp_id);
      const CalculationsPrecision prec =
          late_weights_scaling ? CalculationsPrecision::F32 : precision;
      c += fc::AccumulateFloat(r_name, src_name, prec, isI4O4, use_fma);
      if (late_weights_scaling) {
        c += "    s_sum += " + src_name + ".x + " + src_name + ".y + " +
             src_name + ".z + " + src_name + ".w;\n";
      }
    }
  }
  if (late_weights_scaling) {
    if (precision == CalculationsPrecision::F32_F16) {
      c += "    r_sp0 += ucl::Convert<float4>(rl * w_scale);\n";
      c += "    r_sp0 += ucl::Convert<float4>(w_bias * s_sum);\n";
    } else {
      if (use_fma) {
        c += "    r_sp0 = fma(rl, w_scale, r_sp0);\n";
        c += "    r_sp0 = fma(w_bias, ucl::Init<Type>(s_sum), r_sp0);\n";
      } else {
        c += "    r_sp0 += rl * w_scale;\n";
        c += "    r_sp0 += w_bias * s_sum;\n";
      }
    }
  }
  c += "  }\n";
  c += "  }  // end if dst_s < dst_slices\n";
  const int local_batch_size =
      fc::GetLocalBatchSize(gpu_info, precision, block_spatial, wg_size);
  c += "  __local AccType temp";
  if (local_batch_size != 1) {
    c += "[" + std::to_string(local_batch_size) + "]";
  }
  c += "[" + std::to_string(wg_size.x * wg_size.y) + "];\n";

  const int upload_groups = DivideRoundUp(block_spatial, local_batch_size);
  for (int group = 0; group < upload_groups; ++group) {
    const int first = group * local_batch_size;
    const int last = std::min(block_spatial - 1, first + local_batch_size - 1);
    c += fc::GetReductionCode(first, last, local_batch_size, "local_y",
                              "local_x", "WG_SIZE_X", "temp");
    if (group != upload_groups - 1) {
      c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
    }
  }
  c += "  if (local_x != 0) return;\n";
  c += fc::GenerateDstWrite(conv_params.block_size, conv_params.runtime_check,
                            conv_params.has_bias, conv_params.batched_weights,
                            conv_params.runtime_batch_ids);
  c += "}\n";
  const DataType acc_type = precision == CalculationsPrecision::F16
                                ? DataType::FLOAT16
                                : DataType::FLOAT32;
  absl::StrReplaceAll(
      {
          {"WG_SIZE_X", std::to_string(wg_size.x)},
          {"WG_SIZE_Y", std::to_string(wg_size.y)},
          {"SType", ToUclDataType(src_desc.GetDataType(), 1)},
          {"Type", ToUclDataType(src_desc.GetDataType(), 4)},
          {"AccSType", ToUclDataType(acc_type, 1)},
          {"AccType", ToUclDataType(acc_type, 4)},
      },
      &c);
  return c;
}

int3 GetWorkGroupSize(const GpuInfo& gpu_info, int src_slices, int dst_slices,
                      int batch_size = 1) {
  int3 work_group_size = int3(32, 1, 1);
  const int task_size = dst_slices * batch_size;
  if (gpu_info.IsIntel()) {
    if (task_size < 512 && src_slices >= 128) {
      work_group_size = int3(128, 1, 1);
    } else if (task_size < 1024 && src_slices >= 64) {
      work_group_size = int3(64, 2, 1);
    } else if (task_size < 2048) {
      work_group_size = int3(32, 4, 1);
    } else if (task_size < 1024 * 8) {
      work_group_size = int3(16, 8, 1);
    } else {
      work_group_size = int3(8, 16, 1);
    }
  } else if (gpu_info.IsApple() && gpu_info.apple_info.IsM5Series()) {
    if (task_size < 1024) {
      work_group_size = int3(64, 1, 1);
    } else if (task_size < 1024 * 8) {
      work_group_size = int3(32, 1, 1);
    } else {
      work_group_size = int3(16, 2, 1);
    }
  } else if (gpu_info.IsApple()) {
    if (task_size < 1024) {
      work_group_size = int3(64, 1, 1);
    } else if (task_size < 1024 * 8) {
      work_group_size = int3(16, 4, 1);
    } else {
      work_group_size = int3(8, 8, 1);
    }
  } else if (gpu_info.IsPowerVR()) {
    if (task_size < 256 && src_slices >= 64) {
      work_group_size = int3(64, 2, 1);
    } else if (task_size < 512) {
      work_group_size = int3(32, 4, 1);
    } else if (task_size < 1024) {
      work_group_size = int3(16, 8, 1);
    } else if (task_size < 2048) {
      work_group_size = int3(8, 16, 1);
    } else {
      work_group_size = int3(4, 32, 1);
    }
  } else if (gpu_info.IsAdreno()) {
    if (task_size < 1024) {
      work_group_size = int3(64, 2, 1);
    } else {
      work_group_size = int3(32, 4, 1);
    }
  }
  return work_group_size;
}

int GetSrcN(const ExternalWeights& weights) {
  auto weights_type = fc::GetDataTypeForWeights(weights.desc.type);
  int src_n = 1;
  if (weights_type == DataType::INT2 || weights_type == DataType::INT4) {
    src_n = 2;
  }
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  if (src_slices % src_n != 0 || !weights.desc.IsLinearLayout()) {
    src_n = 1;
  }
  return src_n;
}
}  // namespace

FullyConnectedOI::FullyConnectedOI(const TensorDescriptor& src,
                                   const TensorDescriptor& dst,
                                   CalculationsPrecision precision,
                                   const GpuInfo& gpu_info,
                                   const OHWI& weights_shape,
                                   const WeightsDescription& weights_desc,
                                   const ConvParams& conv_params)
    : conv_params_(conv_params) {
  AddSrcTensor("src_tensor", src);
  AddDstTensor("dst_tensor", dst);

  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  work_group_size_ = GetWorkGroupSize(gpu_info, src_slices, dst_slices);
  const int scale_zp_group_size = src_slices / conv_params.scale_zp_shape.i;
  code_ = GetFullyConnectedOIO4KernelCode(gpu_info, conv_params, precision,
                                          weights_desc, work_group_size_, src,
                                          scale_zp_group_size);
  compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
}

FullyConnectedOI CreateFullyConnectedOI(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias,
    const BHWC* dst_shape_ptr, const TensorDescriptor* src_exp,
    const ConvRuntimeCheckDesc& runtime_check) {
  FullyConnectedOI::ConvParams conv_params;
  conv_params.weights_type = fc::GetDataTypeForWeights(weights.desc.type);
  if (weights.scale) {
    conv_params.scale_zp_shape = weights.scale_zp_shape;
  }
  conv_params.has_bias = bias != nullptr;
  conv_params.has_zero_point =
      weights.zero_point != nullptr || weights.scalar_zero_point.has_value();
  conv_params.src_n = GetSrcN(weights);
  conv_params.runtime_check = runtime_check;
  conv_params.batched_weights = weights.shape.h != 1;
  conv_params.softmax_input_activation = src_exp != nullptr;
  conv_params.block_size =
      fc::GetBlockSize(dst_shape_ptr, conv_params.batched_weights);
  if (runtime_check.packed_groups.has_value()) {
    conv_params.block_size = BHWC(1, 1, 1, 1);
    conv_params.block_size.w = 4;
    if (dst_shape_ptr) {
      const int average_task_size = DivideRoundUp(
          dst_shape_ptr->w, runtime_check.packed_groups->num_groups);
      const int max_tile = 16;
      conv_params.block_size.w = std::min(max_tile, average_task_size);
    }
  }
  const bool ringed_weights =
      conv_params.runtime_check.ring_o_offset_index.has_value() ||
      conv_params.runtime_check.ring_i_offset_index.has_value();
  if (ringed_weights) {
    conv_params.src_n = 1;
  }
  const int vec_size = ringed_weights ? 4 : 16 * conv_params.src_n;

  FullyConnectedOI result(src, dst, precision, gpu_info, weights.shape,
                          weights.desc, conv_params);
  fc::AddWeightsArguments(weights, vec_size, &result);
  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }
  if (src_exp) {
    result.AddSrcTensor("src_exp", *src_exp);
  }
  fc::AddRuntimeParam(runtime_check, &result);
  return result;
}

FullyConnectedOI CreateFullyConnectedOIWeightsBatchIds(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& batch_ids,
    const TensorDescriptor& dst, const ExternalWeights& weights,
    const TensorDescriptor* bias, const BHWC* dst_shape_ptr) {
  FullyConnectedOI::ConvParams conv_params;
  conv_params.weights_type = fc::GetDataTypeForWeights(weights.desc.type);
  if (weights.scale) {
    conv_params.scale_zp_shape = weights.scale_zp_shape;
  }
  conv_params.batched_weights = true;
  conv_params.has_bias = bias != nullptr;
  conv_params.has_zero_point =
      weights.zero_point != nullptr || weights.scalar_zero_point.has_value();
  conv_params.src_n = GetSrcN(weights);
  conv_params.block_size =
      fc::GetBlockSize(dst_shape_ptr, conv_params.batched_weights);
  conv_params.runtime_batch_ids = dst_shape_ptr ? dst_shape_ptr->h : 1;

  FullyConnectedOI result(src, dst, precision, gpu_info, weights.shape,
                          weights.desc, conv_params);
  result.AddSrcTensor("batch_ids", batch_ids);
  fc::AddWeightsArguments(weights, 16 * conv_params.src_n, &result);
  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }

  return result;
}

FullyConnectedOI CreateFullyConnectedOIInt4Sparse2x4(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias,
    const BHWC* dst_shape_ptr) {
  FullyConnectedOI::ConvParams conv_params;
  conv_params.weights_type = fc::GetDataTypeForWeights(weights.desc.type);
  if (weights.scale) {
    conv_params.scale_zp_shape = weights.scale_zp_shape;
  }
  conv_params.has_bias = bias != nullptr;
  conv_params.has_zero_point =
      weights.zero_point != nullptr || weights.scalar_zero_point.has_value();
  conv_params.sparse_2x4 = true;
  conv_params.src_n = 2;

  FullyConnectedOI result(src, dst, precision, gpu_info, weights.shape,
                          weights.desc, conv_params);
  fc::AddSparseWeightsArguments(weights, 16 * conv_params.src_n, &result);
  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }
  return result;
}

}  // namespace ml_drift
