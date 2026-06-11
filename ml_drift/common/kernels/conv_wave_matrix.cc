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

#include "ml_drift/common/kernels/conv_wave_matrix.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GenerateDstCoords(const int3& work_group_launch_order,
                              bool linear_spatial, bool need_depth,
                              bool need_batch) {
  std::string c;
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  if (linear_spatial) {
    if (work_group_launch_order[0] == 0) {
      c += "  int linear_spatial = ucl::GetGlobalId<0>();\n";
    } else {
      c += "  int linear_spatial = ucl::GetGroupId<" +
           std::to_string(launch_remap[0]) +
           ">() * ucl::GetGroupSize<0>() + ucl::GetLocalId<0>();\n";
    }
    if (need_batch) {
      c += "  int B = linear_spatial % args.dst_tensor.Batch();\n";
      c += "  linear_spatial = linear_spatial / args.dst_tensor.Batch();\n";
    }
    if (need_depth) {
      c += "  int DST_X = linear_spatial % args.dst_tensor.Width();\n";
      c += "  linear_spatial = linear_spatial / args.dst_tensor.Width();\n";
      c += "  int DST_Y = linear_spatial % args.dst_tensor.Height();\n";
      c += "  int DST_Z = linear_spatial / args.dst_tensor.Height();\n";
    } else {
      c += "  int DST_Y = linear_spatial / args.dst_tensor.Width();\n";
      c += "  int DST_X = linear_spatial % args.dst_tensor.Width();\n";
    }
    if (work_group_launch_order[1] == 1) {
      c += "  int DST_S = ucl::GetGlobalId<1>();\n";
    } else {
      c += "  int DST_S = ucl::GetGroupId<" + std::to_string(launch_remap[1]) +
           ">() * ucl::GetGroupSize<1>() + ucl::GetLocalId<1>();\n";
    }
  } else {
    if (work_group_launch_order[0] == 0) {
      c += "  int DST_X = ucl::GetGlobalId<0>();\n";
    } else {
      c += "  int DST_X = ucl::GetGroupId<" + std::to_string(launch_remap[0]) +
           ">() * ucl::GetGroupSize<0>() + ucl::GetLocalId<0>();\n";
    }
    if (need_batch) {
      c += "  int B = DST_X % args.dst_tensor.Batch();\n";
      c += "  DST_X = DST_X / args.dst_tensor.Batch();\n";
    }
    std::string global_id_1;
    if (work_group_launch_order[1] == 1) {
      global_id_1 = "ucl::GetGlobalId<1>()";
    } else {
      global_id_1 = "ucl::GetGroupId<" + std::to_string(launch_remap[1]) +
                    ">() * ucl::GetGroupSize<1>() + ucl::GetLocalId<1>()";
    }
    if (need_depth) {
      c += "  int linear_id_1 = " + global_id_1 + ";\n";
      c += "  int DST_Z = linear_id_1 / dst_tensor.Height();\n";
      c += "  int DST_Y = linear_id_1 % dst_tensor.Height();\n";
    } else {
      c += "  int DST_Y = " + global_id_1 + ";\n";
    }
    if (work_group_launch_order[2] == 2) {
      c += "  int DST_S = ucl::GetGlobalId<2>();\n";
    } else {
      c += "  int DST_S = ucl::GetGroupId<" + std::to_string(launch_remap[2]) +
           ">() * ucl::GetGroupSize<2>() + ucl::GetLocalId<2>();\n";
    }
  }

  return c;
}

std::string GetWorkGroupBaseDstS(const int3& work_group_launch_order,
                                 bool linear_spatial) {
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  const int s_dimension = linear_spatial ? 1 : 2;
  return "ucl::GetGroupId<" + std::to_string(launch_remap[s_dimension]) +
         ">() * ucl::GetGroupSize<" + std::to_string(s_dimension) + ">()";
}

std::string GetWorkGroupBaseDstX(const int3& work_group_launch_order,
                                 bool linear_spatial) {
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  return "ucl::GetGroupId<" + std::to_string(launch_remap[0]) +
         ">() * ucl::GetGroupSize<0>()";
}

std::string GenerateCheck(const OperationDef& definition,
                          const GpuInfo& gpu_info,
                          const ConvWaveMatrix::ConvParams& conv_params) {
  std::string check;
  const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
  const std::vector<std::string> names{"in_x", "in_y", "in_z"};
  const std::vector<bool> is_1{conv_params.x_kernel_is_1,
                               conv_params.y_kernel_is_1,
                               conv_params.z_kernel_is_1};
  for (int i = 0; i < axes.size(); ++i) {
    const auto& axis = axes[i];
    if (definition.src_tensors[0].HasAxis(axis) &&
        !definition.src_tensors[0].SupportsZeroClamp(axis, gpu_info) &&
        !is_1[i]) {
      if (!check.empty()) {
        check += " && ";
      }
      check += names[i];
    }
  }
  return check;
}

DataType GetAccumulatorType(const ConvWaveMatrix::ConvParams& conv_params,
                            CalculationsPrecision precision) {
  if (IsFloatType(conv_params.weights_data_type)) {
    return precision == CalculationsPrecision::F16 ? DataType::FLOAT16
                                                   : DataType::FLOAT32;
  } else {
    return IsSigned(conv_params.weights_data_type) ? DataType::INT32
                                                   : DataType::UINT32;
  }
}

void UpdateOffsetAndStrideForInt8(const GpuInfo& gpu_info, std::string& offset,
                                  std::string& stride) {
  if (gpu_info.IsApiVulkan()) {
    if (gpu_info.IsMali()) {
      offset = "(" + offset + ") / 4";
      stride = "(" + stride + ") / 1";
    } else {
      offset = "(" + offset + ") / 4";
      stride = "(" + stride + ") / 4";
    }
  }
}

void AddLinearizeWGReplacements(
    absl::flat_hash_map<std::string, std::string>& replacements,
    const int3& work_group_size) {
  replacements["ucl::GetLocalId<0>()"] = "local_wg_x";
  replacements["ucl::GetLocalId<1>()"] = "local_wg_y";
  replacements["ucl::GetLocalId<2>()"] = "local_wg_z";
  replacements["ucl::GetGroupSize<0>()"] = std::to_string(work_group_size.x);
  replacements["ucl::GetGroupSize<1>()"] = std::to_string(work_group_size.y);
  replacements["ucl::GetGroupSize<2>()"] = std::to_string(work_group_size.z);
  replacements["ucl::GetGlobalId<0>()"] = "ucl::GetGroupId<0>() * " +
                                          std::to_string(work_group_size.x) +
                                          " + local_wg_x";
  replacements["ucl::GetGlobalId<1>()"] = "ucl::GetGroupId<1>() * " +
                                          std::to_string(work_group_size.y) +
                                          " + local_wg_y";
  replacements["ucl::GetGlobalId<2>()"] = "ucl::GetGroupId<2>() * " +
                                          std::to_string(work_group_size.z) +
                                          " + local_wg_z";
  std::string patch = "  int linear_wg_id = ucl::GetLocalId<0>();\n";
  patch += "  int local_wg_x = linear_wg_id % " +
           std::to_string(work_group_size.x) + ";\n";
  patch += "  linear_wg_id = linear_wg_id / " +
           std::to_string(work_group_size.x) + ";\n";
  patch += "  int local_wg_y = linear_wg_id % " +
           std::to_string(work_group_size.y) + ";\n";
  patch += "  int local_wg_z = linear_wg_id / " +
           std::to_string(work_group_size.y) + ";\n";
  replacements["LINEAR_WG_ID"] = patch;
}

inline int GetRangeShift(DataType type) {
  return 1u << (SizeInBitsOf(type) - 1);
}

std::string ReadWeights(const ConvWaveMatrix::ConvParams& conv_params,
                        int src_x4_slices) {
  const bool weights_conversion =
      conv_params.weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4;
  const bool quantized_weights =
      weights_conversion && SizeInBitsOf(conv_params.weights_desc.type) <= 8;
  std::string c;
  if (quantized_weights && conv_params.scale_zp_shape.i != 1) {
    // grouped quantization
    std::string src_id = "s";
    if (conv_params.src_group_slices % src_x4_slices != 0) {
      src_id = "(s + sub_i)";
    }
    c += "    int src_group_id = " + src_id + " / " +
         std::to_string(conv_params.src_group_slices) + +";\n";
    c += "    if (last_src_group_id != src_group_id) {\n";
    c += "      last_src_group_id = src_group_id;\n";
    std::string w_batch = conv_params.scale_zp_shape.h != 1 ? "DST_Y" : "0";
    std::string coords = "w_o_slice, " + w_batch + ", src_group_id";
    c += "      weights_scale = args.weights_scale.Read(" + coords + ");\n";
    if (conv_params.has_zero_point) {
      c += "      Type wzp = args.weights_zero_point.Read(" + coords + ");\n";
    } else {
      c += "      Type wzp = ucl::Init<Type>(0.0f);\n";
    }
    c += "      weights_bias = -weights_scale * ucl::Init<Type>(" +
         std::to_string(GetRangeShift(conv_params.weights_desc.type)) +
         ") + wzp;\n";
    c += "    }\n";
  }
  c += "    Type w0, w1, w2, w3;\n";
  if (conv_params.weights_desc.type == DataType::UINT8) {
    c += "    uint4 u8_i4o4 = args.weights.Read(w_sg_offset);\n";
    c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(u8_i4o4, w0, w1, w2, w3);\n";
  } else if (conv_params.weights_desc.type == DataType::UINT4) {
    c += "    uint2 u4_i4o4 = args.weights.Read(w_sg_offset);\n";
    c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(u4_i4o4, w0, w1, w2, w3);\n";
  } else if (conv_params.weights_desc.type == DataType::UINT2) {
    c += "    uint u2_i4o4 = args.weights.Read(w_sg_offset);\n";
    c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(u2_i4o4, w0, w1, w2, w3);\n";
  } else {
    c += "    args.weights.ReadVec16AsVec4x4(w0, w1, w2, w3, w_sg_offset);\n";
  }
  c += "    w_sg_offset += stride;\n";
  if (quantized_weights) {
    c += R"(
    w0 = w0 * weights_scale + weights_bias;
    w1 = w1 * weights_scale + weights_bias;
    w2 = w2 * weights_scale + weights_bias;
    w3 = w3 * weights_scale + weights_bias;
)";
  }
  return c;
}

std::string GenerateConvolution(
    const OperationDef& definition, const GpuInfo& gpu_info,
    const ConvWaveMatrix::ConvParams& conv_params,
    const ConvWaveMatrix::KernelParams& kernel_params) {
  const DataType acc_type =
      GetAccumulatorType(conv_params, conv_params.precision);
  const auto& src_def = definition.src_tensors[0];
  const int wm_m = kernel_params.wave_matrix_m;
  const int wm_n = kernel_params.wave_matrix_n;
  const int wm_k = kernel_params.wave_matrix_k;
  const int wm_n_slices = wm_n / 4;
  const int wm_k_slices = wm_k / 4;
  std::string c;
  c += "#pragma OPENCL EXTENSION ucl_wave_matrix: enable\n";
  c += "MAIN_FUNCTION($0) {\n";
  if (kernel_params.linearized_wg) {
    c += "LINEAR_WG_ID\n";
  }
  c += GenerateDstCoords(
      kernel_params.work_group_launch_order, kernel_params.linear_spatial,
      src_def.HasAxis(Axis::DEPTH), src_def.HasAxis(Axis::BATCH));
  if (src_def.HasAxis(Axis::BATCH)) {
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  }
  const bool weights_conversion =
      conv_params.weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4;
  const bool quantized_weights =
      weights_conversion && SizeInBitsOf(conv_params.weights_desc.type) <= 8;
  // kOSpatialIOGroupIwmkOwmn
  const int o_group_size_xN = kernel_params.dst_slices_per_thread / wm_n_slices;
  const std::string i_slices_xK = "((args.src_tensor.Slices() * (" +
                                  std::to_string(conv_params.Is8Bit() ? 4 : 1) +
                                  ")) / " + std::to_string(wm_k_slices) + ")";
  if (!weights_conversion) {
    const std::string weights_dst_s = "min(DST_S, args.weight_o_slices_xN - 1)";
    std::string weights_offset;
    if (conv_params.different_weights_for_height) {
      std::string spatial_size = "args.src_tensor.Height()";
      weights_offset = "(" + weights_dst_s + " * " + spatial_size + " + DST_Y)";
    } else {
      std::string spatial_size = "";
      if (!conv_params.x_kernel_is_1) {
        spatial_size += " * args.kernel_size_x";
      }
      if (!conv_params.y_kernel_is_1) {
        spatial_size += " * args.kernel_size_y";
      }
      weights_offset = weights_dst_s + spatial_size;
    }
    weights_offset = weights_offset + " * " + i_slices_xK + " * " +
                     std::to_string(o_group_size_xN) + " * " +
                     std::to_string(wm_n * wm_k);
    c += "  int weights_wgs_offset = " + weights_offset + ";\n";
  }
  c += "  DST_S *= " + std::to_string(kernel_params.dst_slices_per_thread) +
       ";\n";
  if (conv_params.runtime_check.dst_end_ch_index.has_value()) {
    c += "  int dst_s_wg_first_slice = " +
         GetWorkGroupBaseDstS(kernel_params.work_group_launch_order,
                              kernel_params.linear_spatial) +
         " * " + std::to_string(kernel_params.dst_slices_per_thread) + ";\n";
    if (conv_params.runtime_check.dst_end_ch_index.has_value()) {
      c += "  int dst_end_slice = " +
           conv_params.runtime_check.GetRuntimeEndSlice(
               "args.params.Read(args.dst_end_ch_index)",
               "args.dst_tensor.Slices()") +
           ";\n";
      c += "  if (dst_s_wg_first_slice >= dst_end_slice) return;\n";
    }
  }
  if (conv_params.runtime_check.group_sizes_offset.has_value()) {
    c += "  int w_group_size = args.params.Read(args.group_sizes_offset + "
         "DST_Y);\n";
    c += "  int wg_first_w = " +
         GetWorkGroupBaseDstX(kernel_params.work_group_launch_order,
                              kernel_params.linear_spatial) +
         ";\n";
    c += "  if (wg_first_w >= w_group_size) return;\n";
  }
  const int src_x4_slices = kernel_params.GetX4SlicesCount();
  // in 8 bit conv used packed src kInt8C16. src_x4_slices actually is
  // int32_t x 4 (int8_t x 16).
  const int src_xK_slices =
      src_x4_slices * (conv_params.Is8Bit() ? 4 : 1) / wm_k_slices;
  const int dst_x4_slices = kernel_params.dst_slices_per_thread;
  const int dst_xN_slices = dst_x4_slices / wm_n_slices;
  if (kernel_params.linear_spatial) {
    c += "  int spatial_id = ucl::GetLocalId<0>();\n";
    c += "  int slice_id = ucl::GetLocalId<1>();\n";
  } else {
    c += "  int spatial_id = ucl::GetLocalId<1>() * ucl::GetGroupSize<0>() + "
         "ucl::GetLocalId<0>();\n";
    c += "  int slice_id = ucl::GetLocalId<2>();\n";
  }
  const DataType src_cache_type = src_def.GetDataType();
  const DataType dst_cache_type = acc_type;
  const int src_cache_size =
      kernel_params.wave_size * 4 * kernel_params.GetX4SlicesCount();
  const int dst_cache_size = kernel_params.wave_size *
                             kernel_params.wave_matrix_n *
                             kernel_params.GetX4SlicesCount();
  std::string src_cache_name, dst_cache_name, src_cache_x4_name,
      dst_cache_x4_name;
  const bool supports_x4_reinterpret = gpu_info.IsApiMetal();
  if (src_cache_type == dst_cache_type) {
    // Use the same cache for src and dst.
    const int cache_size = std::max(src_cache_size, dst_cache_size);
    c += "  __local " + ToUclDataType(src_cache_type, 1) + " local_cache[" +
         std::to_string(cache_size) + "];\n";
    if (supports_x4_reinterpret) {
      c += "  __local " + ToUclDataType(src_cache_type, 4) +
           "* local_cache_x4 = (__local " + ToUclDataType(src_cache_type, 4) +
           "*)local_cache;\n";
    }
    src_cache_name = "local_cache";
    dst_cache_name = "local_cache";
    src_cache_x4_name = "local_cache_x4";
    dst_cache_x4_name = "local_cache_x4";
  } else {
    c += "  __local " + ToUclDataType(src_cache_type, 1) + " local_cache_src[" +
         std::to_string(src_cache_size) + "];\n";
    c += "  __local " + ToUclDataType(dst_cache_type, 1) + " local_cache_dst[" +
         std::to_string(dst_cache_size) + "];\n";
    if (supports_x4_reinterpret) {
      c += "  __local " + ToUclDataType(src_cache_type, 4) +
           "* local_cache_src_x4 = (__local " +
           ToUclDataType(src_cache_type, 4) + "*)local_cache_src;\n";
      c += "  __local " + ToUclDataType(dst_cache_type, 4) +
           "* local_cache_dst_x4 = (__local " +
           ToUclDataType(dst_cache_type, 4) + "*)local_cache_dst;\n";
    }
    src_cache_name = "local_cache_src";
    dst_cache_name = "local_cache_dst";
    src_cache_x4_name = "local_cache_src_x4";
    dst_cache_x4_name = "local_cache_dst_x4";
  }
  c += "  // sp - spatial dimensions, ch - channels dimension\n";
  c += "  // indexing relative to wave\n";
  for (int sp = 0; sp < kernel_params.wave_size; sp += wm_m) {
    const std::string sp_start = std::to_string(sp);
    const std::string sp_end = std::to_string(sp + wm_m);
    const std::string dst_name_sp = "dst_sp" + sp_start + "_" + sp_end;
    for (int slice = 0; slice < dst_xN_slices; slice += 1) {
      const std::string sl_start = std::to_string(slice * wm_n);
      const std::string sl_end = std::to_string(slice * wm_n + wm_n);
      const std::string dst_name =
          dst_name_sp + "_ch" + sl_start + "_" + sl_end;
      c += "  ucl::wave_matrix<accum, " + ToUclDataType(acc_type, 1) +
           ", WM_M, WM_N> " + dst_name + " = ucl::wave_matrix<accum, " +
           ToUclDataType(acc_type, 1) +
           ", WM_M, "
           "WM_N>(" +
           GetZeroValue(acc_type) + ");\n";
    }
  }
  // We build remapping to load src from local memory in matrix optimized layout
  // Linear spatial case for simplicity
  // sp0_s0 is spatial 0, slice 0, slice is 4 channels
  // wg.x - spatial, size 8, wg.y - slices, size 4
  // sp0_s0 | sp1_s0 | sp2_s0 | sp3_s0 | sp4_s0 | sp5_s0 | sp6_s0 | sp7_s0
  // sp0_s1 | sp1_s1 | sp2_s1 | sp3_s1 | sp4_s1 | sp5_s1 | sp6_s1 | sp7_s1
  // sp0_s2 | sp1_s2 | sp2_s2 | sp3_s2 | sp4_s2 | sp5_s2 | sp6_s2 | sp7_s2
  // sp0_s3 | sp1_s3 | sp2_s3 | sp3_s3 | sp4_s3 | sp5_s3 | sp6_s3 | sp7_s3
  // int linear_wg_id = slice_id * wg.x + spatial_id;
  // Matrix A(left) is [wave_matrix_m, wave_matrix_k] (spatial x channels)
  // A is row major
  // wave_matrix_m mapped to spatial dimension
  // wave_matrix_k mapped to channels dimension
  // if wave_matrix_k == 4, A[wave_matrix_m, 4] trivial mapping,
  //   vec4(channels) is used already
  // if wave_matrix_k == 8, A[wave_matrix_m, 8] need remap to:
  //   sp0_s0 | sp0_s1 | sp1_s0 | sp1_s1 | sp2_s0 | sp2_s1 | etc
  // if wave_matrix_k == 16, A[wave_matrix_m, 16] need remap to:
  //   sp0_s0 | sp0_s1 | sp0_s2 | sp0_s3 | sp1_s0 | sp1_s1 | etc
  //
  // int linear_wg_id = slice_id * SPATIAL_THREADS + spatial_id;
  // int wave_matrix_k_slices = wave_matrix_k / 4;
  // int t0 = slice_id % wave_matrix_k_slices;
  // int t1 = slice_id / wave_matrix_k_slices;
  // int t2 = spatial_id * wave_matrix_k_slices + t0;
  // int tid2 = t1 * SPATIAL_THREADS * wave_matrix_k_slices + t2;
  const std::string wave_matrix_k_slices =
      std::to_string(wm_k / (conv_params.Is8Bit() ? 16 : 4));
  c += "  int t0 = slice_id % " + wave_matrix_k_slices + ";\n";
  c += "  int t1 = slice_id / " + wave_matrix_k_slices + ";\n";
  c += "  int t2 = spatial_id * " + wave_matrix_k_slices + " + t0;\n";
  c += "  int tid2 = t1 * SPATIAL_THREADS * " + wave_matrix_k_slices +
       " + t2;\n";
  if (!conv_params.y_kernel_is_1) {
    c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
    c += "  int temp_y = DST_Y * args.stride_y + args.padding_y;\n";
    c += "  int src_y = ky * args.dilation_y + temp_y;\n";
    if (!src_def.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "  bool in_y = src_y >= 0 && src_y < args.src_tensor.Height();\n";
      if (!src_def.CanReadOutOfBorder(Axis::HEIGHT, gpu_info)) {
        c += "  src_y = clamp(src_y, 0, args.src_tensor.Height() - 1);\n";
      }
    }
  } else {
    c += "  int src_y = min(DST_Y, args.src_tensor.Height() - 1);\n";
  }
  if (!conv_params.x_kernel_is_1) {
    c += "  for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
    c += "  int temp_x = DST_X * args.stride_x + args.padding_x;\n";
    c += "  int src_x = kx * args.dilation_x + temp_x;\n";
    if (!src_def.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      c += "  bool in_x = src_x >= 0 && src_x < args.src_tensor.Width();\n";
      if (!src_def.CanReadOutOfBorder(Axis::WIDTH, gpu_info)) {
        c += "  src_x = clamp(src_x, 0, args.src_tensor.Width() - 1);\n";
      }
    }
  } else {
    c += "  int src_x = min(DST_X, args.src_tensor.Width() - 1);\n";
  }
  if (conv_params.softmax_input_activation) {
    c += "  " + ToUclDataType(src_def.GetDataType(), 2) +
         " exp_val = args.src_exp.Read(src_x, src_y, 0).xy;\n";
  }
  if (src_def.IsLinear()) {
    c += "  int src_address = args.src_tensor.GetAddress(src_x, src_y, "
         "slice_id);\n";
  }
  std::string src_slices_count = "args.src_tensor.Slices()";
  if (conv_params.runtime_check.src_end_ch_index.has_value()) {
    c += "  int src_slices_dynamic = " +
         conv_params.runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.src_end_ch_index)",
             "args.src_tensor.Slices()") +
         ";\n";
    src_slices_count = "src_slices_dynamic";
  }
  if (weights_conversion) {
    // WeightsLayout::kBIOI4O4
    const int wg_slices = kernel_params.GetX4SlicesCount();
    const int i4o4_blocks = src_x4_slices * dst_x4_slices;
    const int weights_per_slice = i4o4_blocks * 16;
    c += "  __local SType weights_cache_wg[" +
         std::to_string(wg_slices * weights_per_slice) + "];\n";
    c += "  __local SType* weights_cache_sg = weights_cache_wg + slice_id * " +
         std::to_string(weights_per_slice) + ";\n";
    c += "  __local Type* weights_cache_sg_x4 = (__local "
         "Type*)weights_cache_sg;\n";
    const int o_groups = dst_x4_slices;
    c += "  int w_sg_offset, stride;\n";
    c += "  int o1, o2, i1, i2;\n";
    c += "  Type weights_scale, weights_bias;\n";
    if (quantized_weights && conv_params.scale_zp_shape.i != 1) {
      // grouped quantization
      c += "  int last_src_group_id = -1;\n";
    }
    if (i4o4_blocks < kernel_params.wave_size) {
      c += "  if (spatial_id < " + std::to_string(i4o4_blocks) + ") {\n";
    }
    // i4o4 blocks stored in layout BIO
    c += "  int sub_i = spatial_id / " + std::to_string(o_groups) + ";\n";
    c += "  int sub_o = spatial_id % " + std::to_string(o_groups) + ";\n";
    c += "  int w_o_slice = min(DST_S + sub_o, args.dst_tensor.Slices()-1);\n";
    c += "  w_sg_offset = (DST_Y * args.src_tensor.Slices() + sub_i) * "
         "args.dst_tensor.Slices() + w_o_slice;\n";
    c += "  stride = args.dst_tensor.Slices() * " +
         std::to_string(src_x4_slices) + ";\n";
    if (quantized_weights && conv_params.scale_zp_shape.i == 1) {
      // linear quantization
      const std::string coords = conv_params.scale_zp_shape.h != 1
                                     ? "w_o_slice, DST_Y, 0"
                                     : "w_o_slice";
      c += "  weights_scale = args.weights_scale.Read(" + coords + ");\n";
      if (conv_params.has_zero_point) {
        c += "  Type wzp = args.weights_zero_point.Read(" + coords + ");\n";
      } else {
        c += "  Type wzp = ucl::Init<Type>(0.0f);\n";
      }
      c += "  weights_bias = -weights_scale * ucl::Init<Type>(" +
           std::to_string(GetRangeShift(conv_params.weights_desc.type)) +
           ") + wzp;\n";
    }
    c += "  o1 = sub_o % " + std::to_string(wm_n_slices) + ";\n";
    c += "  o2 = sub_o / " + std::to_string(wm_n_slices) + ";\n";
    c += "  i1 = sub_i % " + std::to_string(wm_k_slices) + ";\n";
    c += "  i2 = sub_i / " + std::to_string(wm_k_slices) + ";\n";
    if (i4o4_blocks < kernel_params.wave_size) {
      c += "  }\n";
    }
  }
  c += "  for (int s = 0; s < " + src_slices_count +
       "; s += " + std::to_string(src_x4_slices) + ") {\n";
  const DataType right_matrix_type = conv_params.weights_data_type;
  for (int src_s = 0; src_s < src_xK_slices; ++src_s) {
    const std::string src_range = "i" + std::to_string(src_s * wm_k) + "_" +
                                  std::to_string(src_s * wm_k + wm_k);
    for (int dst_s = 0; dst_s < dst_xN_slices; ++dst_s) {
      const std::string dst_range = "o" + std::to_string(dst_s * wm_n) + "_" +
                                    std::to_string(dst_s * wm_n + wm_n);
      const std::string w_name = "w_" + dst_range + "_" + src_range;
      c += "    ucl::wave_matrix<right, " +
           ToUclDataType(right_matrix_type, 1) + ", WM_K, WM_N> " + w_name +
           ";\n";
    }
  }
  c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  if (weights_conversion) {
    const int i4o4_blocks = src_x4_slices * dst_x4_slices;
    if (i4o4_blocks < kernel_params.wave_size) {
      c += "  if (spatial_id < " + std::to_string(i4o4_blocks) + ") {\n";
    }
    c += ReadWeights(conv_params, src_x4_slices);
    for (int i = 0; i < 4; ++i) {
      std::string addr =
          "((i2 * " + std::to_string(dst_x4_slices / wm_n_slices) +
          " + o2) * " + std::to_string(wm_k) + " + i1 * 4 + " +
          std::to_string(i) + ") * " + std::to_string(wm_n_slices) + " + o1";
      c += "    weights_cache_sg_x4[" + addr + "] = w" + std::to_string(i) +
           ";\n";
    }
    if (i4o4_blocks < kernel_params.wave_size) {
      c += "  }\n";
    }
  } else {
    for (int src_s = 0; src_s < src_xK_slices; ++src_s) {
      const std::string src_range = "i" + std::to_string(src_s * wm_k) + "_" +
                                    std::to_string(src_s * wm_k + wm_k);
      for (int dst_s = 0; dst_s < dst_xN_slices; ++dst_s) {
        const std::string dst_range = "o" + std::to_string(dst_s * wm_n) + "_" +
                                      std::to_string(dst_s * wm_n + wm_n);
        const std::string w_name = "w_" + dst_range + "_" + src_range;
        std::string offset = "weights_wgs_offset + " +
                             std::to_string(src_s * dst_xN_slices + dst_s) +
                             " * WM_K * WM_N";
        std::string stride =
            kernel_params.load_right_transposed ? "WM_K" : "WM_N";
        if (conv_params.Is8Bit()) {
          UpdateOffsetAndStrideForInt8(gpu_info, offset, stride);
        }
        const std::string row_major =
            kernel_params.load_right_transposed ? "false" : "true";
        c += "    ucl::WaveMatrixLoad(" + w_name + ", args.weights.GetPtr(), " +
             offset + ", " + stride + ", " + row_major + ");\n";
      }
    }
  }
  std::string src_read_func =
      src_def.IsLinear() ? "args.src_tensor.Read(src_address)"
                         : "args.src_tensor.Read(src_x, src_y, s + slice_id)";
  const std::string check = GenerateCheck(definition, gpu_info, conv_params);
  if (!check.empty()) {
    src_read_func += " * ucl::Convert<" +
                     ToUclDataType(src_def.GetDataType(), 1) + ">(" + check +
                     ")";
  }
  c += "    Type src_val = " + src_read_func + ";\n";
  if (conv_params.softmax_input_activation) {
    c += "    src_val = exp(src_val - exp_val.y) * exp_val.x;\n";
  }
  if (supports_x4_reinterpret) {
    c += "    " + std::string(src_cache_x4_name) + "[tid2] = src_val;\n";
  } else {
    c += "    " + std::string(src_cache_name) + "[tid2 * 4 + 0] = src_val.x;\n";
    c += "    " + std::string(src_cache_name) + "[tid2 * 4 + 1] = src_val.y;\n";
    c += "    " + std::string(src_cache_name) + "[tid2 * 4 + 2] = src_val.z;\n";
    c += "    " + std::string(src_cache_name) + "[tid2 * 4 + 3] = src_val.w;\n";
  }
  if (!weights_conversion) {
    c += "    weights_wgs_offset += WM_N * WM_K * " +
         std::to_string(src_xK_slices * dst_xN_slices) + ";\n";
  }
  if (src_def.IsLinear()) {
    c += "    src_address += args.src_tensor.SliceStride() * " +
         std::to_string(src_x4_slices) + ";\n";
  }
  c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  if (weights_conversion) {
    for (int src_s = 0; src_s < src_xK_slices; ++src_s) {
      const std::string src_range = "i" + std::to_string(src_s * wm_k) + "_" +
                                    std::to_string(src_s * wm_k + wm_k);
      for (int dst_s = 0; dst_s < dst_xN_slices; ++dst_s) {
        const std::string dst_range = "o" + std::to_string(dst_s * wm_n) + "_" +
                                      std::to_string(dst_s * wm_n + wm_n);
        const std::string w_name = "w_" + dst_range + "_" + src_range;
        std::string offset =
            std::to_string(src_s * dst_xN_slices + dst_s) + " * WM_K * WM_N";
        std::string stride =
            kernel_params.load_right_transposed ? "WM_K" : "WM_N";
        if (conv_params.Is8Bit()) {
          UpdateOffsetAndStrideForInt8(gpu_info, offset, stride);
        }
        const std::string row_major =
            kernel_params.load_right_transposed ? "false" : "true";
        c += "    ucl::WaveMatrixLoad(" + w_name + ", weights_cache_sg, " +
             offset + ", " + stride + ", " + row_major + ");\n";
      }
    }
  }
  const DataType left_matrix_type =
      conv_params.Is8Bit() ? DataType::INT8 : src_def.GetDataType();
  c += "    ucl::wave_matrix<left, " + ToUclDataType(left_matrix_type, 1) +
       ", WM_M, WM_K> mat_src;\n";
  const int spatial_matrix_count = kernel_params.wave_size / wm_m;
  for (int src_s = 0; src_s < src_xK_slices; ++src_s) {
    const std::string src_s_range = std::to_string(src_s * wm_k) + "_" +
                                    std::to_string(src_s * wm_k + wm_k);
    for (int sp = 0; sp < kernel_params.wave_size; sp += wm_m) {
      const std::string sp_range =
          std::to_string(sp) + "_" + std::to_string(sp + wm_m);
      const int src_tile_offset = src_s * spatial_matrix_count + (sp / wm_m);
      std::string offset = std::to_string(src_tile_offset) + " * WM_M * WM_K";
      std::string stride = "WM_K";
      if (conv_params.Is8Bit()) {
        UpdateOffsetAndStrideForInt8(gpu_info, offset, stride);
      }
      c += "    ucl::WaveMatrixLoad(mat_src, " + std::string(src_cache_name) +
           ", " + offset + ", " + stride + ");  // loading sp[" + sp_range +
           "] src_ch[" + src_s_range + "]\n";
      for (int dst_s = 0; dst_s < dst_xN_slices; ++dst_s) {
        const std::string dst_s_range = std::to_string(dst_s * wm_n) + "_" +
                                        std::to_string(dst_s * wm_n + wm_n);
        const std::string dst_name = "dst_sp" + sp_range + "_ch" + dst_s_range;
        const std::string w_name = "w_o" + dst_s_range + "_i" + src_s_range;
        c += "    ucl::WaveMatrixMAC(" + dst_name + ", mat_src, " + w_name +
             ");\n";
      }
    }
  }
  c += "  }\n";
  if (!conv_params.x_kernel_is_1) {
    c += "  }\n";
  }
  if (!conv_params.y_kernel_is_1) {
    c += "  }\n";
  }
  c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  const DataType dst_type = definition.dst_tensors[0].GetDataType();
  for (int slice = 0; slice < dst_x4_slices; slice += 1) {
    c += "  " + ToUclDataType(dst_type, 4) + " r" + std::to_string(slice) +
         " = ucl::Init<" + ToUclDataType(dst_type, 4) + ">(" +
         GetZeroValue(dst_type) + ");\n";
  }
  c += "  // transferring from wave matrix to private registers.\n";
  for (int slice_offset = 0; slice_offset < dst_x4_slices;
       slice_offset += wm_n_slices) {
    const std::string ch = "ch" + std::to_string(slice_offset * 4) + "_" +
                           std::to_string(slice_offset * 4 + wm_n);
    for (int sp = 0; sp < kernel_params.wave_size; sp += wm_m) {
      const std::string sp_start = std::to_string(sp);
      const std::string sp_end = std::to_string(sp + wm_m);
      const std::string dst_name_sp = "dst_sp" + sp_start + "_" + sp_end;
      const int spatial_id = sp / wm_m;
      std::string dst_name = dst_name_sp + "_" + ch;
      c += "  ucl::WaveMatrixStore(" + dst_name + ", " +
           std::string(dst_cache_name) + ", " + std::to_string(spatial_id) +
           " * WM_M * WM_N + slice_id * SPATIAL_THREADS * WM_N, WM_N);\n";
    }
    c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
    for (int slice = 0; slice < wm_n_slices; slice += 1) {
      c += "  {\n";
      c += "    int base_id = (slice_id * "
           "SPATIAL_THREADS + spatial_id) * (WM_N / 4) + " +
           std::to_string(slice) + ";\n";
      if (supports_x4_reinterpret) {
        c += "    r" + std::to_string(slice_offset + slice) +
             " = ucl::Convert<" + ToUclDataType(dst_type, 4) + ">(" +
             std::string(dst_cache_x4_name) + "[base_id]);\n";
      } else {
        c += "    r" + std::to_string(slice_offset + slice) +
             ".x = ucl::Convert<" + ToUclDataType(dst_type, 1) + ">(" +
             std::string(dst_cache_name) + "[base_id * 4 + 0]);\n";
        c += "    r" + std::to_string(slice_offset + slice) +
             ".y = ucl::Convert<" + ToUclDataType(dst_type, 1) + ">(" +
             std::string(dst_cache_name) + "[base_id * 4 + 1]);\n";
        c += "    r" + std::to_string(slice_offset + slice) +
             ".z = ucl::Convert<" + ToUclDataType(dst_type, 1) + ">(" +
             std::string(dst_cache_name) + "[base_id * 4 + 2]);\n";
        c += "    r" + std::to_string(slice_offset + slice) +
             ".w = ucl::Convert<" + ToUclDataType(dst_type, 1) + ">(" +
             std::string(dst_cache_name) + "[base_id * 4 + 3]);\n";
      }
      c += "  }\n";
    }
    c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  }
  c += "  if (DST_X >= args.dst_tensor.Width() || DST_Y >= "
       "args.dst_tensor.Height()) {\n";
  c += "    return;\n";
  c += "  }\n";
  for (int slice = 0; slice < dst_x4_slices; slice += 2) {
    const std::string dst_s0 = "DST_S + " + std::to_string(slice);
    const std::string dst_s1 = "DST_S + " + std::to_string(slice + 1);
    const std::string r0_name = "r" + std::to_string(slice);
    const std::string r1_name = "r" + std::to_string(slice + 1);
    c += "  if (" + dst_s0 + " < args.dst_tensor.Slices()) {\n";
    if (conv_params.has_bias) {
      c += "    " + r0_name + " += args.biases.Read(" + dst_s0 + ");\n";
      c += "    " + r1_name + " += args.biases.Read(" + dst_s1 + ");\n";
    }
    c += "    args.dst_tensor.Write(" + r0_name + ", DST_X, DST_Y, " + dst_s0 +
         ");\n";
    c += "    args.dst_tensor.Write(" + r1_name + ", DST_X, DST_Y, " + dst_s1 +
         ");\n";
    c += "  }\n";
  }
  c += "}\n";
  const DataType type = src_def.GetDataType();
  absl::flat_hash_map<std::string, std::string> replacements = {
      {"WM_M", std::to_string(wm_m)},
      {"WM_N", std::to_string(wm_n)},
      {"WM_K", std::to_string(wm_k)},
      {"SPATIAL_THREADS", std::to_string(kernel_params.wave_size)},
      {"Type", ToUclDataType(type, 4)},
      {"SType", ToUclDataType(type, 1)}};
  if (kernel_params.linearized_wg) {
    AddLinearizeWGReplacements(replacements, kernel_params.work_group_size);
  }
  return absl::StrReplaceAll(c, replacements);
}

std::vector<int2> Get2DWorkgroupsEqualToWaveSize(int wave_size) {
  if (wave_size == 32) {
    return {{8, 4}, {16, 2}, {4, 8}, {32, 1}, {2, 16}, {1, 32}};
  } else if (wave_size == 16) {
    return {{8, 2}, {4, 4}, {2, 8}, {16, 1}, {1, 16}};
  } else {
    return {{wave_size, 1}};
  }
}

int Get2dGroupsCount(const BHWC& dst_shape, const int2 group_size) {
  int x_groups = DivideRoundUp(dst_shape.w * dst_shape.b, group_size.x);
  int y_groups = DivideRoundUp(dst_shape.h, group_size.y);
  return x_groups * y_groups;
}

int2 GetOptimalGroupSize(const ConvWaveMatrix::KernelParams& kernel_params,
                         const BHWC& dst_shape) {
  const auto base_work_groups =
      Get2DWorkgroupsEqualToWaveSize(kernel_params.wave_size);
  int min_2d_work_groups = Get2dGroupsCount(dst_shape, base_work_groups[0]);
  int min_index = 0;
  for (int i = 1; i < base_work_groups.size(); ++i) {
    int groups_count = Get2dGroupsCount(dst_shape, base_work_groups[i]);
    if (groups_count < min_2d_work_groups) {
      min_2d_work_groups = groups_count;
      min_index = i;
    }
  }
  return base_work_groups[min_index];
}

ConvWaveMatrix::KernelParams InitParamsForTaskSize(
    const GpuInfo& gpu_info, const OHWI& weights_shape, const BHWC& dst_shape,
    const ConvWaveMatrix::ConvParams& params) {
  ConvWaveMatrix::KernelParams kernel_params;
  if (gpu_info.IsApple()) {
    kernel_params.wave_size = 32;
    kernel_params.wave_matrix_m = 8;
    kernel_params.wave_matrix_n = 8;
    kernel_params.wave_matrix_k = 8;
  } else if (gpu_info.IsNvidia()) {
    kernel_params.wave_size = 32;
    kernel_params.wave_matrix_m = 16;
    kernel_params.wave_matrix_n = 16;
    kernel_params.wave_matrix_k = 16;
    if (params.Is8Bit()) {
      kernel_params.wave_matrix_m = 16;
      kernel_params.wave_matrix_n = 16;
      kernel_params.wave_matrix_k = 32;
    }
  } else if (gpu_info.IsMali()) {
    kernel_params.wave_size = 16;
    kernel_params.wave_matrix_m = 4;
    kernel_params.wave_matrix_n = 4;
    kernel_params.wave_matrix_k = 4;
    if (params.Is8Bit()) {
      kernel_params.wave_matrix_m = 4;
      kernel_params.wave_matrix_n = 16;
      kernel_params.wave_matrix_k = 16;
    }
  } else if (gpu_info.IsIntel()) {
    kernel_params.wave_size =
        32;  // can be 16 or 32 in Vulkan? Must be 32(max_wave_size) in WebGPU?
    kernel_params.wave_matrix_m = 8;
    kernel_params.wave_matrix_n = 16;
    kernel_params.wave_matrix_k = 16;
    if (params.Is8Bit()) {
      kernel_params.wave_matrix_m = 8;
      kernel_params.wave_matrix_n = 16;
      kernel_params.wave_matrix_k = 32;
    }
  }
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  int src_slices_per_wg = src_slices % 4 == 0 && dst_slices % 32 == 0 ? 4 : 2;
  if (params.runtime_check.src_end_ch_index.has_value() &&
      params.runtime_check.GetSlicesAlignment() % src_slices_per_wg != 0) {
    src_slices_per_wg = 2;
  }
  const int dst_group_slices = dst_slices / src_slices_per_wg;
  kernel_params.dst_slices_per_thread = dst_group_slices % 4 == 0 ? 4 : 2;
  if (dst_group_slices % 8 == 0) {
    kernel_params.dst_slices_per_thread = 8;
  }
  if (gpu_info.IsMali()) {
    src_slices_per_wg =
        gpu_info.mali_info.IsValhallGen4() && src_slices % 2 == 0 ? 2 : 1;
    kernel_params.dst_slices_per_thread = 8;
    if (params.Is8Bit()) {
      kernel_params.load_right_transposed = true;
      kernel_params.dst_slices_per_thread = 4;
      src_slices_per_wg = 4;
    }
  }
  if (gpu_info.IsNvidia()) {
    src_slices_per_wg = 4;
    kernel_params.dst_slices_per_thread = 8;
    if (params.Is8Bit()) {
      kernel_params.load_right_transposed = true;
      kernel_params.dst_slices_per_thread = 4;
    }
  }
  if (gpu_info.IsIntel()) {
    src_slices_per_wg = 4;
    kernel_params.dst_slices_per_thread = 8;
    if (gpu_info.IsApiWebGpu() &&
        params.precision == CalculationsPrecision::F32_F16) {
      kernel_params.dst_slices_per_thread = 4;
    }
    if (params.Is8Bit()) {
      kernel_params.load_right_transposed = true;
      kernel_params.dst_slices_per_thread = 4;
    }
  }
  if (gpu_info.IsApple() && gpu_info.apple_info.IsFamilyApple9()) {
    const int waves_spatial_count =
        params.different_weights_for_height
            ? DivideRoundUp(dst_shape.w * dst_shape.b,
                            kernel_params.wave_size) *
                  dst_shape.h
            : DivideRoundUp(dst_shape.w * dst_shape.b * dst_shape.h,
                            kernel_params.wave_size);
    const int total_waves_count = waves_spatial_count * (dst_slices / 2);
    const double waves_per_cu =
        total_waves_count /
        static_cast<double>(gpu_info.GetComputeUnitsCount());
    if (waves_per_cu < 64.0) {
      kernel_params.dst_slices_per_thread =
          std::min(kernel_params.dst_slices_per_thread, 2);
    } else if (waves_per_cu < 128.0) {
      kernel_params.dst_slices_per_thread =
          std::min(kernel_params.dst_slices_per_thread, 4);
    }
  }
  if (params.different_weights_for_height) {
    kernel_params.work_group_size =
        int3(kernel_params.wave_size, 1, src_slices_per_wg);
    kernel_params.work_group_launch_order = int3(0, 1, 2);
    kernel_params.linear_spatial = false;
  } else {
    const int2 optimal_2d_group_size =
        GetOptimalGroupSize(kernel_params, dst_shape);
    const int groups2d_count =
        Get2dGroupsCount(dst_shape, optimal_2d_group_size);
    const int groups1d_count = DivideRoundUp(
        dst_shape.w * dst_shape.b * dst_shape.h, kernel_params.wave_size);
    if (groups1d_count < groups2d_count) {
      kernel_params.work_group_size =
          int3(kernel_params.wave_size, src_slices_per_wg, 1);
      kernel_params.work_group_launch_order = int3(0, 1, 2);
      kernel_params.linear_spatial = true;
    } else {
      kernel_params.work_group_size = int3(
          optimal_2d_group_size.x, optimal_2d_group_size.y, src_slices_per_wg);
      kernel_params.work_group_launch_order = int3(0, 1, 2);
      kernel_params.linear_spatial = false;
    }
  }
  // ConvWaveMatrix has assumption about mapping of subgroups to work-group that
  // in general not enforced in APIs and can lead to incorrect results.
  // Assumption is that subgroups linearly mapped to linearized work-group.
  // LinearID = (local_id.z * wg_size.y + local_id.y) * wg_size.x + local_id.x;
  // SGLinearID = sg_id * sg_size + sg_local_id.
  // In ConvWaveMatrix assumed that LinearID == SGLinearID for all threads in
  // wg(that can be not true in general).
  // wg_size always divisible by sg_size in ConvWaveMatrix.
  if (gpu_info.IsApiWebGpu() || gpu_info.IsApiVulkan()) {
    // In some cases on Intel produced incorrect results without this flag in
    // Vulkan. Maybe because of wrong assumption described above.
    kernel_params.linearized_wg = true;
  }
  if (params.runtime_check.src_end_ch_index.has_value()) {
    ABSL_CHECK(params.runtime_check.GetSlicesAlignment() % src_slices_per_wg ==
               0);
  }
  if (params.weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4) {
    const int src_x4_slices = kernel_params.GetX4SlicesCount();
    const int dst_x4_slices = kernel_params.dst_slices_per_thread;
    const int i4o4_blocks = src_x4_slices * dst_x4_slices;
    ABSL_CHECK(i4o4_blocks <= kernel_params.wave_size);
  }
  return kernel_params;
}

bool SupportsConvWaveMatrix(const GpuInfo& gpu_info,
                            CalculationsPrecision precision,
                            const OHWI& weights_shape) {
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  if (gpu_info.IsApple()) {
    const auto type = DeduceDataTypeFromPrecision(precision);
    if ((precision == CalculationsPrecision::F32 ||
         precision == CalculationsPrecision::F16) &&
        gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/8,
                                   /*n_size=*/8,
                                   /*k_size=*/8,
                                   /*left_type=*/type,
                                   /*right_type=*/type,
                                   /*result_type=*/type})) {
      return src_slices % 2 == 0 && dst_slices % 4 == 0;
    }
  }
  if (gpu_info.IsMali()) {
    if (precision == CalculationsPrecision::F32 &&
        gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/4,
                                   /*n_size=*/4,
                                   /*k_size=*/4,
                                   /*left_type=*/DataType::FLOAT32,
                                   /*right_type=*/DataType::FLOAT32,
                                   /*result_type=*/DataType::FLOAT32})) {
      return dst_slices % 8 == 0;
    }
  }
  if (gpu_info.IsNvidia()) {
    if (precision == CalculationsPrecision::F16 &&
        gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/16,
                                   /*n_size=*/16,
                                   /*k_size=*/16,
                                   /*left_type=*/DataType::FLOAT16,
                                   /*right_type=*/DataType::FLOAT16,
                                   /*result_type=*/DataType::FLOAT16})) {
      return src_slices % 4 == 0 && dst_slices % 8 == 0;
    }
  }
  if (gpu_info.IsIntel()) {
    if (precision == CalculationsPrecision::F16 &&
        gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/8,
                                   /*n_size=*/16,
                                   /*k_size=*/16,
                                   /*left_type=*/DataType::FLOAT16,
                                   /*right_type=*/DataType::FLOAT16,
                                   /*result_type=*/DataType::FLOAT16})) {
      return src_slices % 4 == 0 && dst_slices % 8 == 0;
    }
    if (precision == CalculationsPrecision::F32_F16 &&
        gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/8,
                                   /*n_size=*/16,
                                   /*k_size=*/16,
                                   /*left_type=*/DataType::FLOAT16,
                                   /*right_type=*/DataType::FLOAT16,
                                   /*result_type=*/DataType::FLOAT32})) {
      return src_slices % 4 == 0 && dst_slices % 8 == 0;
    }
  }
  return false;
}

}  // namespace

ConvWaveMatrix::ConvWaveMatrix(const OperationDef& definition,
                               const GpuInfo& gpu_info,
                               const ConvParams& conv_params,
                               const KernelParams& kernel_params) {
  params_ = conv_params;
  kernel_params_ = kernel_params;
  if (kernel_params_.linearized_wg) {
    work_group_size_.x = kernel_params_.work_group_size.x *
                         kernel_params_.work_group_size.y *
                         kernel_params_.work_group_size.z;
    work_group_size_.y = 1;
    work_group_size_.z = 1;
  } else {
    work_group_size_ = kernel_params_.work_group_size;
  }
  work_group_launch_order_ = kernel_params_.work_group_launch_order;
  if (kernel_params_.linear_spatial) {
    grid_dimension_ = 2;
  } else {
    grid_dimension_ = 3;
  }
  code_ = GenerateConvolution(definition, gpu_info, params_, kernel_params_);
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  if (gpu_info.IsIntel() && gpu_info.IsApiVulkan()) {
    compiler_options_.push_back(
        WaveSizeToCompilerOption(kernel_params_.wave_size));
  }
}

void ConvWaveMatrix::UploadWeights(
    const Tensor<OHWI, DataType::FLOAT32>& weights) {
  WeightsDescription weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor buffer_desc;
  buffer_desc.element_type = params_.weights_data_type;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::GLOBAL;
  buffer_desc.size = elements_count * SizeOf(weights_desc.type);
  buffer_desc.data.resize(buffer_desc.size);
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(buffer_desc.data));
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

void ConvWaveMatrix::UploadWeights(
    const Tensor<OHWI, DataType::INT8>& weights) {
  WeightsDescription weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::INT32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::GLOBAL;
  buffer_desc.size = elements_count * SizeOf(weights_desc.type);
  buffer_desc.data.resize(buffer_desc.size);
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(buffer_desc.data));
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

int3 ConvWaveMatrix::GetGridSize() const {
  const int task_size_x = dst_[0]->Width() * dst_[0]->Batch();
  const int task_size_y = dst_[0]->Height();
  const int task_size_z = dst_[0]->Depth();
  const int task_size_s =
      DivideRoundUp(dst_[0]->Slices(), kernel_params_.dst_slices_per_thread);
  int grid_x, grid_y, grid_z;
  if (kernel_params_.linear_spatial) {
    grid_x = task_size_x * task_size_y * task_size_z;
    grid_y = task_size_s;
    grid_z = 1;
  } else {
    grid_x = task_size_x;
    grid_y = task_size_y * task_size_z;
    grid_z = task_size_s;
  }
  if (kernel_params_.linearized_wg) {
    const int workgroups_x =
        DivideRoundUp(grid_x, kernel_params_.work_group_size.x);
    const int workgroups_y =
        DivideRoundUp(grid_y, kernel_params_.work_group_size.y);
    const int workgroups_z =
        DivideRoundUp(grid_z, kernel_params_.work_group_size.z);
    return int3(workgroups_x * work_group_size_.x, workgroups_y, workgroups_z);
  } else {
    return int3(grid_x, grid_y, grid_z);
  }
}

void ConvWaveMatrix::AddConvParams(const Convolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  if (!params_.x_kernel_is_1) {
    args_.AddInt("stride_x", attr.strides.w);
    args_.AddInt("padding_x", -attr.padding.prepended.w);
    args_.AddInt("kernel_size_x", weights_shape.w);
    args_.AddInt("dilation_x", attr.dilations.w);
  }
  if (!params_.y_kernel_is_1) {
    args_.AddInt("stride_y", attr.strides.h);
    args_.AddInt("padding_y", -attr.padding.prepended.h);
    args_.AddInt("kernel_size_y", weights_shape.h);
    args_.AddInt("dilation_y", attr.dilations.h);
  }
}

ConvWaveMatrix CreateConvWaveMatrix(const OperationDef& definition,
                                    CalculationsPrecision precision,
                                    const BHWC& dst_shape,
                                    const Convolution2DAttributes& attr,
                                    const GpuInfo& gpu_info) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  ConvWaveMatrix::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.Init(attr, /*different_weight_for_height=*/false);
  params.has_bias = !attr.bias.data.empty();
  params.precision = precision;
  params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  auto kernel_params =
      InitParamsForTaskSize(gpu_info, weights_shape, dst_shape, params);
  ConvWaveMatrix desc(definition, gpu_info, params, kernel_params);
  desc.AddConvParams(attr);
  const int dst_slices_xN =
      DivideRoundUp(weights_shape.o, kernel_params.wave_matrix_n);
  desc.args_.AddInt(
      "weight_o_slices_xN",
      AlignByN(dst_slices_xN, kernel_params.dst_slices_per_thread /
                                  (kernel_params.wave_matrix_n / 4)));

  desc.UploadWeights(GetFloatWeights(attr));

  if (params.has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, params.weights_data_type, attr.bias);
    desc.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                       std::move(bias_tensor_desc)));
  }
  return desc;
}

ConvWaveMatrix CreateConvWaveMatrixExternalWeights(
    const OperationDef& definition, CalculationsPrecision precision,
    const BHWC& dst_shape, const Convolution2DAttributes& attr,
    const GpuInfo& gpu_info, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  ConvWaveMatrix::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.Init(attr, different_weights_for_height);
  params.precision = precision;
  params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  params.softmax_input_activation = src_exp != nullptr;
  params.runtime_check = runtime_check;
  params.has_bias = bias != nullptr;
  auto kernel_params =
      InitParamsForTaskSize(gpu_info, weights_shape, dst_shape, params);
  ConvWaveMatrix desc(definition, gpu_info, params, kernel_params);
  desc.AddConvParams(attr);
  const int dst_slices_xN =
      DivideRoundUp(weights_shape.o, kernel_params.wave_matrix_n);
  desc.args_.AddInt(
      "weight_o_slices_xN",
      AlignByN(dst_slices_xN, kernel_params.dst_slices_per_thread /
                                  (kernel_params.wave_matrix_n / 4)));

  BufferDescriptor weights_desc;
  weights_desc.element_type = params.weights_data_type;
  weights_desc.element_size = 1;
  weights_desc.memory_type = MemoryType::GLOBAL;
  desc.AddSrcBuffer("weights", weights_desc);

  if (bias) {
    desc.AddSrcTensor("biases", *bias);
  }

  if (src_exp) {
    desc.AddSrcTensor("src_exp", *src_exp);
  }

  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    desc.args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    desc.args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.group_sizes_offset.has_value()) {
    desc.args_.AddInt("group_sizes_offset", *runtime_check.group_sizes_offset);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = DataType::INT32;
    buffer_desc.element_size = 1;
    desc.AddSrcBuffer("params", buffer_desc);
  }

  return desc;
}

ConvWaveMatrix CreateConvWaveMatrixExternalWeights(
    const OperationDef& definition, CalculationsPrecision precision,
    const BHWC& dst_shape, const ExternalWeights& weights,
    const GpuInfo& gpu_info, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  ConvWaveMatrix::ConvParams params;
  params.weights_desc = weights.desc;
  params.scale_zp_shape = weights.scale_zp_shape;
  params.has_zero_point = weights.zero_point != nullptr;
  params.src_group_slices =
      DivideRoundUp(weights.shape.i, 4) / weights.scale_zp_shape.i;
  params.different_weights_for_height = different_weights_for_height;
  params.precision = precision;
  params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  params.softmax_input_activation = src_exp != nullptr;
  params.runtime_check = runtime_check;
  params.has_bias = bias != nullptr;
  auto kernel_params =
      InitParamsForTaskSize(gpu_info, weights.shape, dst_shape, params);
  ConvWaveMatrix desc(definition, gpu_info, params, kernel_params);
  const int dst_slices_xN =
      DivideRoundUp(weights.shape.o, kernel_params.wave_matrix_n);
  desc.args_.AddInt(
      "weight_o_slices_xN",
      AlignByN(dst_slices_xN, kernel_params.dst_slices_per_thread /
                                  (kernel_params.wave_matrix_n / 4)));

  BufferDescriptor buffer_desc;
  if (SizeInBitsOf(weights.desc.type) >= 16) {
    // float weights
    buffer_desc.element_type = weights.desc.type;
    buffer_desc.element_size = 16;
  } else {
    // quantized weights
    buffer_desc.element_type = DataType::UINT32;
    buffer_desc.element_size = SizeInBitsOf(weights.desc.type) / 2;
  }
  buffer_desc.memory_type = MemoryType::GLOBAL;
  desc.AddSrcBuffer("weights", buffer_desc);

  if (bias) {
    desc.AddSrcTensor("biases", *bias);
  }

  if (weights.scale) {
    desc.AddSrcTensor("weights_scale", *weights.scale);
  }
  if (weights.zero_point) {
    desc.AddSrcTensor("weights_zero_point", *weights.zero_point);
  }

  if (src_exp) {
    desc.AddSrcTensor("src_exp", *src_exp);
  }

  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    desc.args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    desc.args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.group_sizes_offset.has_value()) {
    desc.args_.AddInt("group_sizes_offset", *runtime_check.group_sizes_offset);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = DataType::INT32;
    buffer_desc.element_size = 1;
    desc.AddSrcBuffer("params", buffer_desc);
  }

  return desc;
}

ConvWaveMatrix CreateConvWaveMatrixInt8(
    const OperationDef& definition, const BHWC& dst_shape,
    const Tensor<OHWI, DataType::INT8>& weights, const GpuInfo& gpu_info) {
  ConvWaveMatrix::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.different_weights_for_height = false;
  params.x_kernel_is_1 = true;
  params.y_kernel_is_1 = true;
  params.has_bias = false;
  params.weights_data_type = DataType::INT8;
  auto kernel_params =
      InitParamsForTaskSize(gpu_info, weights.shape, dst_shape, params);
  ConvWaveMatrix desc(definition, gpu_info, params, kernel_params);
  const int dst_slices_xN =
      DivideRoundUp(weights.shape.o, kernel_params.wave_matrix_n);
  desc.args_.AddInt(
      "weight_o_slices_xN",
      AlignByN(dst_slices_xN, kernel_params.dst_slices_per_thread /
                                  (kernel_params.wave_matrix_n / 4)));

  desc.UploadWeights(weights);
  return desc;
}

ConvWaveMatrix CreateConvWaveMatrixInt8ExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OHWI& weights_shape, const BHWC& dst_shape) {
  ConvWaveMatrix::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.different_weights_for_height = false;
  params.x_kernel_is_1 = true;
  params.y_kernel_is_1 = true;
  params.has_bias = false;
  params.weights_data_type = DataType::INT8;
  auto kernel_params =
      InitParamsForTaskSize(gpu_info, weights_shape, dst_shape, params);
  ConvWaveMatrix desc(definition, gpu_info, params, kernel_params);
  const int dst_slices_xN =
      DivideRoundUp(weights_shape.o, kernel_params.wave_matrix_n);
  desc.args_.AddInt(
      "weight_o_slices_xN",
      AlignByN(dst_slices_xN, kernel_params.dst_slices_per_thread /
                                  (kernel_params.wave_matrix_n / 4)));

  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::INT32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::GLOBAL;
  desc.AddSrcBuffer("weights", buffer_desc);
  return desc;
}

bool SupportsConvWaveMatrix(const GpuInfo& gpu_info,
                            CalculationsPrecision precision,
                            const ExternalWeights& weights) {
  if (!gpu_info.IsApiMetal() || !gpu_info.IsApple()) {
    return false;
  }
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const bool supported_type = weights.desc.type == DataType::FLOAT32 ||
                              weights.desc.type == DataType::FLOAT16 ||
                              weights.desc.type == DataType::UINT8 ||
                              weights.desc.type == DataType::UINT4 ||
                              weights.desc.type == DataType::UINT2;
  if (weights.desc.layout != WeightsLayout::kOSpatialIOGroupI4O4 ||
      weights.desc.output_group_size != dst_slices || !supported_type) {
    return false;
  }
  return SupportsConvWaveMatrix(gpu_info, precision, weights.shape);
}

bool SupportsConvWaveMatrix(const GpuInfo& gpu_info,
                            CalculationsPrecision precision,
                            const Convolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  if (attr.groups != 1) {
    return false;
  }
  return SupportsConvWaveMatrix(gpu_info, precision, weights_shape);
}

bool SupportsConvWaveMatrixInt8(const GpuInfo& gpu_info,
                                const OHWI& weights_shape) {
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  if (src_slices % 4 != 0) {
    return false;
  }
  if (/* DISABLES CODE */ (false) && gpu_info.IsMali()) {
    if (gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/4,
                                   /*n_size=*/16,
                                   /*k_size=*/16,
                                   /*left_type=*/DataType::INT8,
                                   /*right_type=*/DataType::INT8,
                                   /*result_type=*/DataType::INT32})) {
      return src_slices % 4 == 0 && dst_slices % 4 == 0;
    }
  }
  if (gpu_info.IsNvidia()) {
    if (gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/16,
                                   /*n_size=*/16,
                                   /*k_size=*/32,
                                   /*left_type=*/DataType::INT8,
                                   /*right_type=*/DataType::INT8,
                                   /*result_type=*/DataType::INT32})) {
      return src_slices % 4 == 0 && dst_slices % 4 == 0;
    }
  }
  if (gpu_info.IsIntel()) {
    if (gpu_info.SupportsWaveMatMulOp(
            WaveMatMulOpDescriptor{/*m_size=*/8,
                                   /*n_size=*/16,
                                   /*k_size=*/32,
                                   /*left_type=*/DataType::INT8,
                                   /*right_type=*/DataType::INT8,
                                   /*result_type=*/DataType::INT32})) {
      return src_slices % 8 == 0 && dst_slices % 4 == 0;
    }
  }
  return false;
}

PackedType GetConvWaveMatrixInt8SrcType() { return PackedType::kInt8C16; }

bool IsGoodTaskSizeForAppleConvSimd(const BHWC& dst_shape,
                                    const OHWI& weights_shape,
                                    CalculationsPrecision precision,
                                    const GpuInfo& gpu_info) {
  const uint64_t task_size_spatial = dst_shape.b * dst_shape.h * dst_shape.w;
  const uint64_t wave_size = 32;
  const double useful_part = static_cast<double>(task_size_spatial) /
                             AlignByN(task_size_spatial, wave_size);
  double threshold_useful_part =
      gpu_info.apple_info.IsFamilyApple9() ? 0.75 : 0.95;
  if (precision == CalculationsPrecision::F32 &&
      gpu_info.apple_info.IsSIMDMatMulFp32Perf2x()) {
    threshold_useful_part = 0.6;
  }
  if (useful_part < threshold_useful_part) {
    return false;
  }
  const double task_size_slices = DivideRoundUp(dst_shape.c, 8);
  const double task_size = task_size_spatial * task_size_slices;
  const double task_size_per_cu = task_size / gpu_info.GetComputeUnitsCount();
  const double waves_per_cu = task_size_per_cu / wave_size;
  double threshold_waves_per_cu =
      precision == CalculationsPrecision::F32 ? 8.0 : 16.0;
  if (waves_per_cu < threshold_waves_per_cu) {
    return false;
  }
  const double total_flops = GetConvolutionFlops(dst_shape, weights_shape);
  const double flops_per_thread =
      total_flops / (waves_per_cu * gpu_info.GetComputeUnitsCount()) /
      wave_size;
  const double threshold_flops_per_thread =
      gpu_info.apple_info.IsFamilyApple9() ? 512.0 : 2048.0;
  return flops_per_thread >= threshold_flops_per_thread;
}

}  // namespace ml_drift
