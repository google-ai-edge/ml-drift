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

#include "ml_drift/common/kernels/conv_wave_memory.h"

#include <algorithm>
#include <string>
#include <variant>
#include <vector>

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
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/task/wave_memory_util.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
void AppendConditionally(const std::string& value, const std::string& delimiter,
                         std::string* result) {
  if (!result->empty()) {
    *result += delimiter;
  }
  *result += value;
}

inline bool IsConvKernelXis1(const Convolution2DAttributes& conv_attr) {
  return std::visit([](const auto& w) { return w.shape.w; },
                    conv_attr.weights) == 1 &&
         conv_attr.dilations.w == 1 && conv_attr.strides.w == 1 &&
         conv_attr.padding.prepended.w == 0 &&
         conv_attr.padding.appended.w == 0;
}

inline bool IsConvKernelYis1(const Convolution2DAttributes& conv_attr) {
  return std::visit([](const auto& w) { return w.shape.h; },
                    conv_attr.weights) == 1 &&
         conv_attr.dilations.h == 1 && conv_attr.strides.h == 1 &&
         conv_attr.padding.prepended.h == 0 &&
         conv_attr.padding.appended.h == 0;
}

std::string GenerateDstCoords(bool precise_xy, bool slices_first,
                              bool need_batch) {
  std::string c;
  if (slices_first) {
    if (precise_xy) {
      c += "  int linear_spatial = ucl::GetGroupId<1>() * "
           "ucl::GetGroupSize<0>() + "
           "ucl::GetLocalId<0>();\n";
      if (need_batch) {
        c += "  int B = linear_spatial % args.dst_tensor.Batch();\n";
        c += "  linear_spatial = linear_spatial / args.dst_tensor.Batch();\n";
      }
      c += "  int X = linear_spatial % args.dst_tensor.Width();\n";
      c += "  int Y = linear_spatial / args.dst_tensor.Width();\n";
      c += "  int S = ucl::GetGroupId<0>() * ucl::GetGroupSize<2>() + "
           "ucl::GetLocalId<2>();\n";
    } else {
      c += "  int X = ucl::GetGroupId<1>() * ucl::GetGroupSize<0>() + "
           "ucl::GetLocalId<0>();\n";
      c += "  int Y = ucl::GetGroupId<2>() * ucl::GetGroupSize<1>() + "
           "ucl::GetLocalId<1>();\n";
      c += "  int S = ucl::GetGroupId<0>() * ucl::GetGroupSize<2>() + "
           "ucl::GetLocalId<2>();\n";
      if (need_batch) {
        c += "  int B = X % args.dst_tensor.Batch();\n";
        c += "  X = X / args.dst_tensor.Batch();\n";
      }
    }
  } else {
    if (precise_xy) {
      c += "  int linear_spatial = ucl::GetGlobalId<0>();\n";
      if (need_batch) {
        c += "  int B = linear_spatial % args.dst_tensor.Batch();\n";
        c += "  linear_spatial = linear_spatial / args.dst_tensor.Batch();\n";
      }
      c += "  int X = linear_spatial % args.dst_tensor.Width();\n";
      c += "  int Y = linear_spatial / args.dst_tensor.Width();\n";
    } else {
      c += "  int X = ucl::GetGlobalId<0>();\n";
      c += "  int Y = ucl::GetGlobalId<1>();\n";
      if (need_batch) {
        c += "  int B = X % args.dst_tensor.Batch();\n";
        c += "  X = X / args.dst_tensor.Batch();\n";
      }
    }
    c += "  int S = ucl::GetGlobalId<2>();\n";
  }
  return c;
}

std::string GenerateConvImg(const ConvWaveMemory::ConvParams& conv_params,
                            const std::string& dst, const std::string& src,
                            int index) {
  std::string c;
  if (conv_params.Is8Bit()) {
    c = R"(
  $0.x = __builtin_PXL_dot_u8_x16($1, weights_cache_wave_var0, $2, $0.x);
  $0.y = __builtin_PXL_dot_u8_x16($1, weights_cache_wave_var0, $3, $0.y);
  $0.z = __builtin_PXL_dot_u8_x16($1, weights_cache_wave_var0, $4, $0.z);
  $0.w = __builtin_PXL_dot_u8_x16($1, weights_cache_wave_var0, $5, $0.w);
)";
    return absl::Substitute(c, dst, src, index * 4 + 0, index * 4 + 1,
                            index * 4 + 2, index * 4 + 3);
  } else if (conv_params.precision == CalculationsPrecision::F32) {
    c += "  $0.x = __builtin_PXL_dot_f32_x4($1, weights_cache_wave_var0, $2, "
         "$0.x);\n";
    c += "  $0.y = __builtin_PXL_dot_f32_x4($1, weights_cache_wave_var0, $3, "
         "$0.y);\n";
    c += "  $0.z = __builtin_PXL_dot_f32_x4($1, weights_cache_wave_var0, $4, "
         "$0.z);\n";
    c += "  $0.w = __builtin_PXL_dot_f32_x4($1, weights_cache_wave_var0, $5, "
         "$0.w);\n";
    return absl::Substitute(c, dst, src, index * 4 + 0, index * 4 + 1,
                            index * 4 + 2, index * 4 + 3);
  } else if (conv_params.precision == CalculationsPrecision::F16) {
    c += "  $0a = pixel_dot4_f16_x2($1, weights_cache_wave_var0, "
         "$2, $0a);\n";
    c += "  $0b = pixel_dot4_f16_x2($1, weights_cache_wave_var0, "
         "$3, $0b);\n";
    return absl::Substitute(c, dst, src, index * 2 + 0, index * 2 + 1);
  }
  return c;
}

std::string GenerateConv(const ConvWaveMemory::ConvParams& conv_params,
                         const std::string& dst, const std::string& src,
                         int index) {
  std::string code;
  if (conv_params.weights_data_type == DataType::INT8) {
    code += "  $0.x = qcom_dot8_acc(weights_cache[$2].x, $1_uint, $0.x);\n";
    code += "  $0.y = qcom_dot8_acc(weights_cache[$2].y, $1_uint, $0.y);\n";
    code += "  $0.z = qcom_dot8_acc(weights_cache[$2].z, $1_uint, $0.z);\n";
    code += "  $0.w = qcom_dot8_acc(weights_cache[$2].w, $1_uint, $0.w);\n";
    return absl::Substitute(code, dst, src, index);
  }
  switch (conv_params.precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      code += "  $0 += $1.x * weights_cache[$2];\n";
      code += "  $0 += $1.y * weights_cache[$3];\n";
      code += "  $0 += $1.z * weights_cache[$4];\n";
      code += "  $0 += $1.w * weights_cache[$5];\n";
      break;
    case CalculationsPrecision::F32_F16:
      code +=
          "  $0 += ucl::Convert<float4>($1.x * weights_cache[$2] + $1.y * "
          "weights_cache[$3] + $1.z * "
          "weights_cache[$4] + $1.w * weights_cache[$5]);\n";
      break;
  }
  return absl::Substitute(code, dst, src, index * 4 + 0, index * 4 + 1,
                          index * 4 + 2, index * 4 + 3);
}

DataType GetAccumulatorType(const ConvWaveMemory::ConvParams& conv_params) {
  if (IsFloatType(conv_params.weights_data_type)) {
    return conv_params.precision == CalculationsPrecision::F16
               ? DataType::FLOAT16
               : DataType::FLOAT32;
  } else {
    return IsSigned(conv_params.weights_data_type) ? DataType::INT32
                                                   : DataType::UINT32;
  }
}

std::string GenerateConvolutionGeneric(
    const GpuInfo& gpu_info, const ConvWaveMemory::ConvParams& conv_params,
    const ConvWaveMemory::KernelParams& kernel_params) {
  const bool late_xy_check = !gpu_info.IsAdreno();
  const auto& src_tensor = conv_params.src_desc;
  const DataType src_type = src_tensor.GetDataType();
  std::string c;
  c += "#pragma OPENCL EXTENSION ucl_wave_memory: enable\n";
  if (gpu_info.IsAdreno() && conv_params.Is8Bit()) {
    c += "#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable\n";
  }
  if (kernel_params.img_wave_dot) {
    if (conv_params.Is8Bit()) {
      c += "uint __builtin_PXL_dot_u8_x16(uint4 a, uint4 b, uint index, uint "
           "acc);\n";
    } else if (conv_params.precision == CalculationsPrecision::F32) {
      c += "float __builtin_PXL_dot_f32_x4(float4 a, float4 b, uint index, "
           "float acc);\n";
    } else if (conv_params.precision == CalculationsPrecision::F16) {
      c +=
          "uint __builtin_PXL_dot_x2_f16_x4(uint2 a, uint4 b, uint sglid, uint "
          "acc);\n";
      c += "half2 pixel_dot4_f16_x2(half4 a, half8 b, uint sglid, half2 acc) "
           "{\n";
      c += "  return as_half2(__builtin_PXL_dot_x2_f16_x4(as_uint2(a), "
           "as_uint4(b), sglid, as_uint(acc)));\n";
      c += "}\n";
    }
  }
  c += "MAIN_FUNCTION($0) {\n";
  c += GenerateDstCoords(kernel_params.precise_spatial,
                         kernel_params.slices_first,
                         src_tensor.HasAxis(Axis::BATCH));
  if (src_tensor.HasAxis(Axis::BATCH)) {
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  }
  if (conv_params.runtime_check.packed_groups.has_value()) {
    c += "  int w_batch_id = Y;\n";
    c += "  Y = 0;\n";
    c += "  int w_group_size = args.params.Read(args.packed_params_offset + "
         "w_batch_id);\n";
    c += "  int w_group_offset = args.params.Read(args.packed_params_offset + "
         "w_batch_id + " +
         std::to_string(conv_params.runtime_check.packed_groups->num_groups) +
         ");\n";
    c += "  int wave_first_w = (X / WAVE_SIZE) * WAVE_SIZE;\n";
    c += "  if (wave_first_w >= w_group_size) return;\n";
    c += "  X = w_group_offset + X;\n";
  }
  if (!late_xy_check) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
         "return;\n";
  }
  c += "  if (S * " + std::to_string(kernel_params.slices_out) +
       " >= args.dst_tensor.Slices()) return;\n\n";
  if (conv_params.runtime_check.dst_end_ch_index.has_value()) {
    c += "  int dst_end_slice_runtime = " +
         conv_params.runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.dst_end_ch_index)",
             "args.dst_tensor.Slices()") +
         ";\n";
    c += "  if (S * " + std::to_string(kernel_params.slices_out) +
         " >= dst_end_slice_runtime) return;\n";
  }
  const DataType acc_type = GetAccumulatorType(conv_params);
  const std::string zero_value = GetZeroValue(acc_type);
  const bool accum_as_vec2 =
      kernel_params.img_wave_dot &&
      conv_params.precision == CalculationsPrecision::F16;
  const int acc_vec_size = accum_as_vec2 ? 2 : 4;
  const std::string acc_type_ucl = ToUclDataType(acc_type, acc_vec_size);
  for (int s_out = 0; s_out < kernel_params.slices_out; ++s_out) {
    std::string val_name = "r" + std::to_string(s_out);
    if (accum_as_vec2) {
      c += "  " + acc_type_ucl + " " + val_name + "a = ucl::Init<" +
           acc_type_ucl + ">(" + zero_value + ");\n";
      c += "  " + acc_type_ucl + " " + val_name + "b = ucl::Init<" +
           acc_type_ucl + ">(" + zero_value + ");\n";
    } else {
      c += "  " + acc_type_ucl + " " + val_name + " = ucl::Init<" +
           acc_type_ucl + ">(" + zero_value + ");\n";
    }
  }
  c += "\n";
  std::string src_start_slice = "0";
  if (conv_params.groups_count != 1) {
    c += "      int group_id = (S * " +
         std::to_string(kernel_params.slices_out) +
         ") / args.dst_group_size;\n";
    c += "      int src_start_slice = group_id * args.src_group_size;\n";
    c += "      int src_end_slice = src_start_slice + args.src_group_size;\n";
    src_start_slice = "src_start_slice";
  }
  std::string src_end_slice = conv_params.groups_count != 1
                                  ? "src_end_slice"
                                  : "args.src_tensor.Slices()";
  if (conv_params.runtime_check.src_end_ch_index.has_value()) {
    c += "  int src_slices_dynamic = " +
         conv_params.runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.src_end_ch_index)",
             "args.src_tensor.Slices()") +
         ";\n";
    src_end_slice = "src_slices_dynamic";
  }
  c += "  int x_coord = mad24(X, args.stride_x, args.padding_x);\n";
  c += "  int y_coord = mad24(Y, args.stride_y, args.padding_y);\n";
  c += "  int coord_x, coord_y, coord_s;\n";
  std::string f_offset = "S";
  if (conv_params.different_weights_for_height) {
    if (conv_params.runtime_check.packed_groups.has_value()) {
      f_offset =
          "(S * " +
          std::to_string(conv_params.runtime_check.packed_groups->num_groups) +
          " + w_batch_id)";
    } else {
      f_offset = "(S * args.src_tensor.Height() + Y)";
    }
  }
  f_offset += " * args.i_slices";
  int wave_cache_size = kernel_params.slices_out * kernel_params.slices_in * 4;
  std::string wave_cache_type = "Type";
  if (conv_params.Is8Bit()) {
    // 8 bit conv uses 4 uint32 values instead of 4 uint8 values.
    wave_cache_size /= 4;
    wave_cache_type = "uint4";
  } else if (kernel_params.img_wave_dot &&
             conv_params.precision == CalculationsPrecision::F16) {
    // f16 conv uses 8 x float16 (128 bit per thread) instead of 4 x float16
    // values.
    wave_cache_size /= 2;
    wave_cache_type = "half8";
  }
  if (kernel_params.unroll_x_loop) {
    wave_cache_size *= conv_params.weights_shape.w;
  }
  c += "  __wave " + wave_cache_type + " weights_cache[" +
       std::to_string(wave_cache_size) + "];\n";
  f_offset += " * " + std::to_string(wave_cache_size / kernel_params.slices_in);
  if (!kernel_params.unroll_x_loop && !conv_params.x_kernel_is_1) {
    f_offset += " * args.kernel_size_x";
  }
  if (!conv_params.y_kernel_is_1) {
    f_offset += " * args.kernel_size_y";
  }
  c += "  int f_offset = " + f_offset + ";\n\n";
  if (!kernel_params.slices_loop_first) {
    c += "  coord_s = " + src_start_slice + ";\n";
    c += "  do {\n";
  }
  std::string oob_check;
  if (!conv_params.y_kernel_is_1) {
    c += "  int y = 0;\n";
    c += "  do {\n";
    c += "    coord_y = mad24(y, args.dilation_y, y_coord);\n";
    if (!src_tensor.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      AppendConditionally("in_y", " && ", &oob_check);
      c += "    bool in_y = coord_y >= 0 && coord_y < "
           "args.src_tensor.Height();\n";
      c += "    coord_y = clamp(coord_y, 0, args.src_tensor.Height() - 1);\n";
    }
  } else {
    c += "  coord_y = Y;\n";
  }
  if (!kernel_params.unroll_x_loop && !conv_params.x_kernel_is_1) {
    c += "    int x = 0;\n";
    c += "    do {\n";
    c += "      coord_x = mad24(x, args.dilation_x, x_coord);\n";
    if (!src_tensor.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      AppendConditionally("in_x", " && ", &oob_check);
      c += "      bool in_x = coord_x >= 0 && coord_x < "
           "args.src_tensor.Width();\n";
      c += "      coord_x = clamp(coord_x, 0, args.src_tensor.Width() - 1);\n";
    }
  } else {
    c += "  coord_x = X;\n";
  }
  if (src_tensor.IsLinear()) {
    c += "      int addr = args.src_tensor.GetAddress(coord_x, coord_y, " +
         src_start_slice + ");\n";
    c += "      int dz = args.src_tensor.SliceStride();\n";
    if (src_tensor.ReturnsZeroForNegOneRead(gpu_info) && !oob_check.empty()) {
      c += "      addr = select(-1, addr, " + oob_check + ");\n";
      c += "      dz = select(0, dz, " + oob_check + ");\n";
    }
  }
  if (conv_params.softmax_input_activation) {
    c += "  SType2 exp_val = args.src_exp.Read(coord_x, coord_y, 0).xy;\n";
  }
  const std::string oob_multiplier =
      oob_check.empty() ? "" : " * ucl::Convert<SType>(" + oob_check + ")";
  if (kernel_params.slices_loop_first) {
    c += "      coord_s = " + src_start_slice + ";\n";
  }
  if (conv_params.blocked_quantization) {
    c += "  for (int quant_i_group = 0; quant_i_group < args.quant_i_groups; "
         "quant_i_group += 1) {\n";
    c += "    int block_slice = 0;\n";
  }
  if (kernel_params.slices_loop_first) {
    c += "      do {\n";
  }
  auto read_src = [&]() {
    if (src_tensor.IsLinear()) {
      if (src_tensor.ReturnsZeroForNegOneRead(gpu_info)) {
        return std::string("args.src_tensor.Read(addr); addr += dz;\n");
      } else {
        return std::string("args.src_tensor.Read(addr)" + oob_multiplier +
                           "; addr += dz;\n");
      }
    } else {
      return std::string("args.src_tensor.Read(coord_x, coord_y, coord_s)" +
                         oob_multiplier + ";\n");
    }
  };
  if (kernel_params.unroll_x_loop) {
    for (int kx = 0; kx < conv_params.weights_shape.w; ++kx) {
      c += "    coord_x = mad24(" + std::to_string(kx) +
           ", args.dilation_x, x_coord);\n";
      c += "    Type src" + std::to_string(kx) + " = " + read_src();
    }
  } else if (src_type == DataType::UINT32) {
    for (int s_in = 0; s_in < kernel_params.slices_in / 4; ++s_in) {
      const std::string val_name = "src" + std::to_string(s_in);
      c += "  uint4 " + val_name + " = " + read_src();
      c += "  coord_s++;\n";
      if (!kernel_params.img_wave_dot) {
        const std::string coords[4] = {"x", "y", "z", "w"};
        for (int i = 0; i < 4; ++i) {
          c += "  uint src" + std::to_string(s_in * 4 + i) +
               "_uint = " + val_name + "." + coords[i] + ";\n";
        }
      }
    }
  } else {
    const int src_slices = DivideRoundUp(conv_params.weights_shape.i, 4);
    for (int s_in = 0; s_in < kernel_params.slices_in; ++s_in) {
      const std::string val = "src" + std::to_string(s_in);
      if (src_type == DataType::UINT8) {
        c += "        uchar4 " + val + ";\n";
        if (kernel_params.img_wave_dot &&
            src_slices % kernel_params.slices_in != 0) {
          c += "        if (coord_s < args.src_tensor.Slices()) {\n";
          c += "          " + val + " = " + read_src();
          c += "        } else {\n";
          c += "          " + val + " = (uchar4)(0);\n";
          c += "        }\n";
        } else {
          c += "        " + val + " = " + read_src();
        }
        c += "        uint " + val + "_uint = as_uint(" + val + ");\n";
        c += "        coord_s++;\n";
      } else {
        c += "        Type " + val + " = " + read_src();
        c += "        coord_s++;\n";
      }
      if (conv_params.softmax_input_activation) {
        c += "  " + val + " = ucl::Exp<Type>(" + val +
             " - exp_val.y) * exp_val.x;\n";
      }
    }
    if (kernel_params.img_wave_dot && conv_params.Is8Bit() &&
        src_type == DataType::UINT8) {
      c += "        uint4 src_packed = (uint4)(src0_uint, src1_uint, "
           "src2_uint, src3_uint);\n";
    }
    if (!kernel_params.slices_loop_first) {
      c += "        coord_s -= " + std::to_string(kernel_params.slices_in) +
           ";\n";
    }
  }
  if (!gpu_info.IsAdreno()) {
    c += "        ucl::SyncThreads<WaveLoad>();\n";
  }
  std::string weights_base_ptr = "args.weights.GetPtr()";
  if (kernel_params.img_wave_dot) {
    if (conv_params.Is8Bit()) {
      weights_base_ptr = "((__global uint4*)(" + weights_base_ptr + "))";
    } else if (conv_params.precision == CalculationsPrecision::F16) {
      weights_base_ptr = "((__global half8*)(" + weights_base_ptr + "))";
    }
  }
  c += "        ucl::WaveLoad(weights_cache, " + weights_base_ptr +
       ", f_offset, " + std::to_string(wave_cache_size) + ");\n";
  c += "        f_offset += " + std::to_string(wave_cache_size) + ";\n";
  c += "        ucl::SyncThreads<WaveLoad>();\n";
  int in_slices = kernel_params.slices_in;
  if (kernel_params.img_wave_dot && conv_params.Is8Bit()) {
    in_slices /= 4;
  }
  if (kernel_params.unroll_x_loop) {
    for (int kx = 0; kx < conv_params.weights_shape.w; ++kx) {
      std::string src_name = "src" + std::to_string(kx);
      for (int s_out = 0; s_out < kernel_params.slices_out; ++s_out) {
        const std::string dst_name = "r" + std::to_string(s_out);
        if (kernel_params.img_wave_dot) {
          c += GenerateConvImg(conv_params, dst_name, src_name,
                               kx * kernel_params.slices_out + s_out);
        } else {
          c += GenerateConv(conv_params, dst_name, src_name,
                            kx * kernel_params.slices_out + s_out);
        }
      }
    }
  } else {
    for (int s_in = 0; s_in < in_slices; ++s_in) {
      std::string src_name = "src" + std::to_string(s_in);
      if (kernel_params.img_wave_dot && conv_params.Is8Bit() &&
          src_type == DataType::UINT8) {
        src_name = "src_packed";
      }
      for (int s_out = 0; s_out < kernel_params.slices_out; ++s_out) {
        const std::string dst_name = "r" + std::to_string(s_out);
        if (kernel_params.img_wave_dot) {
          c += GenerateConvImg(conv_params, dst_name, src_name,
                               s_in * kernel_params.slices_out + s_out);
        } else {
          c += GenerateConv(conv_params, dst_name, src_name,
                            s_in * kernel_params.slices_out + s_out);
        }
      }
    }
  }
  if (conv_params.blocked_quantization) {
    c +=
        "    block_slice += " + std::to_string(kernel_params.slices_in) + ";\n";
    c += "  } while (block_slice < args.quant_i_group_slices);\n";
    c += "    float4 scale = args.src_params.Read<float>(X, Y, "
         "quant_i_group);\n";
    for (int s_out = 0; s_out < kernel_params.slices_out; ++s_out) {
      std::string val_name = "r" + std::to_string(s_out);
      c += "    " + val_name + " = ucl::Convert<int4>(ucl::Convert<float4>(" +
           val_name + ") * scale.w);\n";
    }
    c += "  }\n";
  } else {
    if (kernel_params.slices_loop_first) {
      c += "      } while (coord_s < " + src_end_slice + ");\n";
    }
  }
  if (!kernel_params.unroll_x_loop && !conv_params.x_kernel_is_1) {
    c += "      x++;\n";
    c += "    } while (x < args.kernel_size_x);\n";
  }
  if (!conv_params.y_kernel_is_1) {
    c += "    y++;\n";
    c += "  } while (y < args.kernel_size_y);\n";
  }
  if (!kernel_params.slices_loop_first) {
    c += "    coord_s += " + std::to_string(kernel_params.slices_in) + ";\n";
    c += "  } while (coord_s < " + src_end_slice + ");\n";
  }
  if (late_xy_check) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
         "return;\n";
  }
  if (conv_params.runtime_check.packed_groups.has_value()) {
    c += "  if (X >= w_group_offset + w_group_size) return;\n\n";
  }
  c += "  coord_s = mul24(S, " + std::to_string(kernel_params.slices_out) +
       ");\n";
  c += "  coord_x = X;\n";
  c += "  coord_y = Y;\n";
  for (int s_out = 0; s_out < kernel_params.slices_out; ++s_out) {
    std::string val_name = "r" + std::to_string(s_out);
    if (accum_as_vec2) {
      const std::string acc_type_ucl = ToUclDataType(acc_type, 4);
      val_name = "(" + acc_type_ucl + ")(" + val_name + "a, " + val_name + "b)";
    }
    c += "  if (coord_s < args.dst_tensor.Slices()) {\n";
    c += "    args.dst_tensor::type res = "
         "ucl::Convert<args.dst_tensor::type>(" +
         val_name + ");\n";
    if (conv_params.has_bias) {
      c += "    res += args.biases.Read(coord_s);\n";
    } else if (!conv_params.Is8Bit()) {
      // fake if, without it we have worse performance
      c += "    if (coord_s < 0) {\n";
      c += "      res += args.src_tensor.Read(0, 0, 0);\n";
      c += "    }\n";
    }
    c += "    args.dst_tensor.Write(res, coord_x, coord_y, coord_s);\n";
    c += "    coord_s++;\n";
    c += "  }\n";
  }
  c += "}\n";
  absl::StrReplaceAll({{"SType", ToUclDataType(src_type, 1)},
                       {"Type", ToUclDataType(src_type, 4)},
                       {"WAVE_SIZE", std::to_string(kernel_params.wave_size)}},
                      &c);

  return c;
}

std::vector<int2> Get2DWorkgroupsEqualTo128() {
  return {{128, 1}, {64, 2}, {32, 4}, {16, 8}};
}

ConvWaveMemory::KernelParams GetKernelParamsAdreno(
    const ConvWaveMemory::ConvParams& params, const GpuInfo& gpu_info,
    int src_slices, int dst_slices, const BHWC* dst_shape) {
  ConvWaveMemory::KernelParams kernel_params;
  kernel_params.wave_size = 128;
  kernel_params.slices_in = 1;
  if (src_slices % 2 == 0 && params.precision != CalculationsPrecision::F32) {
    kernel_params.slices_in = 2;
  }
  const AdrenoInfo& adreno_info = gpu_info.adreno_info;
  if (gpu_info.adreno_info.adreno_gpu == AdrenoGpu::kAdreno650 ||
      gpu_info.adreno_info.adreno_gpu == AdrenoGpu::kAdreno660 ||
      gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen7) {
    kernel_params.slices_first = true;
  }

  if (dst_slices % 8 == 0 && params.precision == CalculationsPrecision::F16 &&
      (adreno_info.adreno_gpu == AdrenoGpu::kAdreno642 ||
       adreno_info.adreno_gpu == AdrenoGpu::kAdreno650 ||
       adreno_info.adreno_gpu == AdrenoGpu::kAdreno660 ||
       gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen7)) {
    kernel_params.slices_out = 8;
  } else if (dst_slices % 4 == 0 || dst_slices >= 7) {
    kernel_params.slices_out = 4;
  } else if (dst_slices % 2 == 0) {
    kernel_params.slices_out = 2;
  } else {
    kernel_params.slices_out = 1;
  }

  if (params.weights_data_type == DataType::INT8) {
    if (gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen7) {
      kernel_params.slices_out = 16;
    } else {
      kernel_params.slices_out = 8;
    }
    kernel_params.slices_in = src_slices % 2 == 0 ? 2 : 1;
    if (params.src_desc.GetDataType() == DataType::UINT32) {
      kernel_params.slices_in = 4;
      kernel_params.slices_out = 8;
    }
  }

  if (!params.src_desc.IsLinear() &&
      (!params.x_kernel_is_1 || !params.y_kernel_is_1)) {
    kernel_params.slices_in = 1;
    kernel_params.slices_loop_first = false;
  }

  if (dst_slices == 3 || dst_slices == 6 || dst_slices == 9) {
    kernel_params.slices_out = 3;
  }

  if (dst_slices == 5 || dst_slices == 10) {
    kernel_params.slices_out = 5;
  }

  kernel_params.precise_spatial = !params.different_weights_for_height;
  if (dst_shape) {
    int min_1d_work_groups =
        DivideRoundUp(dst_shape->w * dst_shape->b * dst_shape->h, 128);
    const auto base_work_groups = Get2DWorkgroupsEqualTo128();
    int min_2d_work_groups = min_1d_work_groups * 10;
    for (const auto& work_group : base_work_groups) {
      int x_groups = DivideRoundUp(dst_shape->w * dst_shape->b, work_group.x);
      int y_groups = DivideRoundUp(dst_shape->h, work_group.y);
      int xy_groups = x_groups * y_groups;
      min_2d_work_groups = std::min(min_2d_work_groups, xy_groups);
    }
    float min_waves;
    if (min_1d_work_groups < min_2d_work_groups &&
        !params.different_weights_for_height) {
      kernel_params.precise_spatial = true;
      min_waves = min_1d_work_groups * dst_slices;
    } else {
      kernel_params.precise_spatial = false;
      min_waves = min_2d_work_groups * dst_slices;
    }
    const float waves_per_cu = min_waves / gpu_info.GetComputeUnitsCount();
    if (waves_per_cu < 256.0 && kernel_params.slices_out >= 8) {
      kernel_params.slices_out = 4;
    }
    if (waves_per_cu < 32.0 && kernel_params.slices_out >= 4) {
      kernel_params.slices_out = 2;
    }
    if (waves_per_cu < 16.0 && kernel_params.slices_out >= 2) {
      kernel_params.slices_out = 1;
    }
  }

  return kernel_params;
}

ConvWaveMemory::KernelParams GetKernelParamsPowerVR(
    const ConvWaveMemory::ConvParams& params, const GpuInfo& gpu_info,
    int src_slices, int dst_slices, const BHWC* dst_shape) {
  ConvWaveMemory::KernelParams kernel_params;
  kernel_params.precise_spatial = !params.different_weights_for_height;
  kernel_params.slices_first = true;
  kernel_params.slices_out = 4;
  kernel_params.slices_in = src_slices % 2 == 0 ? 2 : 1;
  kernel_params.wave_size = 128;

  if (gpu_info.SupportsExtension("cl_img_pixel_subgroup_dot")) {
    if (params.Is8Bit()) {
      kernel_params.img_wave_dot = true;
      kernel_params.slices_out = 4;
      kernel_params.slices_in = 4;
      if (params.src_desc.GetDataType() == DataType::UINT32) {
        kernel_params.slices_in = src_slices % 8 == 0 ? 8 : 4;
      }
    } else if (params.precision == CalculationsPrecision::F32 ||
               params.precision == CalculationsPrecision::F16) {
      kernel_params.img_wave_dot = true;
      kernel_params.slices_in = 1;
      kernel_params.slices_out =
          params.precision == CalculationsPrecision::F16 ? 8 : 4;

      if (!params.src_desc.IsLinear() &&
          (!params.x_kernel_is_1 || !params.y_kernel_is_1)) {
        kernel_params.slices_loop_first = false;
        kernel_params.slices_in = 1;
        kernel_params.slices_out = 4;
        kernel_params.unroll_x_loop = true;
        kernel_params.precise_spatial = false;
      }

      if (dst_shape) {
        int min_1d_work_groups =
            DivideRoundUp(dst_shape->w * dst_shape->b * dst_shape->h, 128);
        const auto base_work_groups = Get2DWorkgroupsEqualTo128();
        int min_2d_work_groups = min_1d_work_groups * 10;
        for (const auto& work_group : base_work_groups) {
          int x_groups =
              DivideRoundUp(dst_shape->w * dst_shape->b, work_group.x);
          int y_groups = DivideRoundUp(dst_shape->h, work_group.y);
          int xy_groups = x_groups * y_groups;
          min_2d_work_groups = std::min(min_2d_work_groups, xy_groups);
        }
        float min_waves;
        if (min_1d_work_groups < min_2d_work_groups &&
            !params.different_weights_for_height) {
          kernel_params.precise_spatial = true;
          min_waves = min_1d_work_groups * dst_slices;
        } else {
          kernel_params.precise_spatial = false;
          min_waves = min_2d_work_groups * dst_slices;
        }
        const float waves_per_cu = min_waves / gpu_info.GetComputeUnitsCount();
        if (waves_per_cu < 256.0 && kernel_params.slices_out >= 8) {
          kernel_params.slices_out = 4;
        }
        if (waves_per_cu < 128.0 && kernel_params.slices_out >= 4) {
          kernel_params.slices_out = 2;
        }
        if (waves_per_cu < 64.0 && kernel_params.slices_out >= 2) {
          kernel_params.slices_out = 1;
        }
      }
      if (dst_slices < 4) {
        kernel_params.slices_out =
            std::min(kernel_params.slices_out, dst_slices);
      }
      if (kernel_params.slices_loop_first) {
        int max_slices_in = src_slices % 2 == 0 ? 2 : 1;
        max_slices_in = src_slices % 4 == 0 ? 4 : max_slices_in;
        if (kernel_params.slices_out <= 2) {
          kernel_params.slices_in = std::min(max_slices_in, 4);
        } else {
          kernel_params.slices_in = std::min(max_slices_in, 2);
        }
      }
    }
  }

  return kernel_params;
}

ConvWaveMemory::KernelParams GetKernelParams(
    const ConvWaveMemory::ConvParams& params, const GpuInfo& gpu_info,
    int src_slices, int dst_slices, const BHWC* dst_shape) {
  ConvWaveMemory::KernelParams kernel_params;
  if (gpu_info.IsAdreno()) {
    kernel_params = GetKernelParamsAdreno(params, gpu_info, src_slices,
                                          dst_slices, dst_shape);
  } else if (gpu_info.IsPowerVR()) {
    kernel_params = GetKernelParamsPowerVR(params, gpu_info, src_slices,
                                           dst_slices, dst_shape);
  } else if (gpu_info.IsIntel()) {
    kernel_params.precise_spatial = !params.different_weights_for_height;
    kernel_params.slices_first = true;
    kernel_params.slices_out = 4;
    kernel_params.slices_in = src_slices % 2 == 0 ? 2 : 1;
    kernel_params.wave_size = 32;
  } else if (gpu_info.IsApple()) {
    kernel_params.precise_spatial = !params.different_weights_for_height;
    kernel_params.slices_first = true;
    kernel_params.slices_out = 4;
    kernel_params.slices_in = src_slices % 2 == 0 ? 2 : 1;
    kernel_params.wave_size = 32;
  }
  if (params.runtime_check.src_end_ch_index.has_value() &&
      params.runtime_check.GetSlicesAlignment() % kernel_params.slices_in !=
          0) {
    kernel_params.slices_in = 1;
  }

  if (params.groups_count != 1) {
    const int dst_group_slices = dst_slices / params.groups_count;
    if (dst_group_slices % kernel_params.slices_out != 0) {
      if (kernel_params.slices_out >= 4 && dst_group_slices % 2 == 0) {
        kernel_params.slices_out = 2;
      } else {
        kernel_params.slices_out = 1;
      }
    }
  }
  return kernel_params;
}

}  // namespace

ConvWaveMemory::ConvWaveMemory(const ConvWaveMemory::ConvParams& conv_params,
                               const GpuInfo& gpu_info,
                               const OHWI& weights_shape,
                               const BHWC* dst_shape) {
  conv_params_ = conv_params;
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  if (conv_params_.groups_count != 1) {
    args_.AddInt("src_group_size", src_slices);
    args_.AddInt("dst_group_size", dst_slices / conv_params_.groups_count);
  }
  kernel_params_ = GetKernelParams(conv_params_, gpu_info, src_slices,
                                   dst_slices, dst_shape);
  if (kernel_params_.precise_spatial ||
      conv_params_.different_weights_for_height) {
    work_group_size_ = int3(kernel_params_.wave_size, 1, 1);
  } else {
    if (kernel_params_.wave_size == 128) {
      work_group_size_ = int3(16, 8, 1);
    } else if (kernel_params_.wave_size == 64) {
      work_group_size_ = int3(8, 8, 1);
    } else if (kernel_params_.wave_size == 32) {
      work_group_size_ = int3(8, 4, 1);
    } else {
      work_group_size_ = int3(kernel_params_.wave_size, 1, 1);
    }
  }

  if (kernel_params_.img_wave_dot) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
    compiler_options_.push_back(
        CompilerOptions::kClPixelDisableKernelBlobCache);
    compiler_options_.push_back(CompilerOptions::kClPixelDisableRecompile);
  }

  args_.AddInt("i_slices", src_slices);
  if (!conv_params.x_kernel_is_1) {
    args_.AddInt("kernel_size_x", weights_shape.w);
  }
  if (!conv_params.y_kernel_is_1) {
    args_.AddInt("kernel_size_y", weights_shape.h);
  }

  conv_params_.weights_shape = weights_shape;
}

void ConvWaveMemory::GenerateCode(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const int2& stride, const int2& padding,
                                  const int2& dilation) {
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  if (conv_params_.different_weights_for_height) {
    args_.AddInt("y_offset");
  }
  args_.AddInt("stride_x", stride.x);
  args_.AddInt("stride_y", stride.y);
  args_.AddInt("padding_x", padding.x);
  args_.AddInt("padding_y", padding.y);
  args_.AddInt("dilation_x", dilation.x);
  args_.AddInt("dilation_y", dilation.y);

  code_ = GenerateConvolutionGeneric(gpu_info, conv_params_, kernel_params_);
  work_group_launch_order_ = int3(0, 1, 2);
  if (kernel_params_.slices_first) {
    work_group_launch_order_ = int3(2, 0, 1);
  }
  if (gpu_info.IsApiOpenCl() && gpu_info.IsAdreno() &&
      gpu_info.adreno_info.IsAdreno8xx() &&
      conv_params_.precision == CalculationsPrecision::F16) {
    compiler_options_.push_back(CompilerOptions::kClAdrenoFixBinary);
  }
}

int3 ConvWaveMemory::GetGridSize() const {
  if (kernel_params_.precise_spatial) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Height() * dst_[0]->Batch();
    const int grid_z =
        DivideRoundUp(dst_[0]->Slices(), kernel_params_.slices_out);
    return int3(grid_x, 1, grid_z);
  } else {
    int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    int grid_y = dst_[0]->Height();
    if (conv_params_.runtime_check.packed_groups.has_value()) {
      grid_x = conv_params_.runtime_check.packed_groups->max_group_size;
      grid_y = conv_params_.runtime_check.packed_groups->num_groups;
    }
    const int grid_z =
        DivideRoundUp(dst_[0]->Slices(), kernel_params_.slices_out);
    return int3(grid_x, grid_y, grid_z);
  }
}

std::vector<int3> ConvWaveMemory::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (conv_params_.runtime_check.HasValues()) {
    tuning_type = TuningType::kFast;
  }
  if (conv_params_.different_weights_for_height) {
    switch (tuning_type) {
      case TuningType::kExhaustive:
        return GetWorkGroupsXMultipleOf(kernel_params_.wave_size, gpu_info,
                                        kernel_info, grid_size_);
      case TuningType::kFast:
      default: {
        const int spatial_size = dst_[0]->Width() * dst_[0]->Batch();
        if (spatial_size >= 1024 && dst_[0]->Slices() >= 256) {
          return {{kernel_params_.wave_size * 4, 1, 2}};
        } else {
          return {{kernel_params_.wave_size, 1, 8}};
        }
      }
    }
  }
  if (kernel_params_.precise_spatial) {
    switch (tuning_type) {
      case TuningType::kExhaustive:
        return GetWorkGroupsXMultipleOf(kernel_params_.wave_size, gpu_info,
                                        kernel_info, grid_size_);
      case TuningType::kFast:
      default:
        return {GetConvWorkGroupXisMultipleKYis1(kernel_params_.wave_size,
                                                 grid_size_)};
    }
  } else {
    switch (tuning_type) {
      case TuningType::kExhaustive:
        return GetWorkGroupsXYMultipleOf(kernel_params_.wave_size, gpu_info,
                                         kernel_info, grid_size_);
      case TuningType::kFast:
      default:
        if (conv_params_.weights_data_type == DataType::INT8 &&
            gpu_info.IsAdreno()) {
          if (gpu_info.adreno_info.IsAdreno8xx()) {
            int src_slices = src_[0]->Slices();
            if (conv_params_.src_desc.GetDataType() == DataType::UINT32) {
              src_slices *= 4;
            }
            const int dst_slices = dst_[0]->Slices();
            if (src_slices >= dst_slices * 2 && dst_slices % 32 == 0) {
              return {{128, 1, 4}};
            }
            int max_z_size = dst_slices <= src_slices ? 8 : 1;
            return {GetConvWorkGroupXYisMultipleK(
                kernel_params_.wave_size, grid_size_, 1024, max_z_size)};
          }
          if (grid_size_.x % 512 == 0) {
            if (gpu_info.adreno_info.generation >=
                AdrenoInfo::Generation::kGen7) {
              return {{512, 1, 1}};
            } else {
              return {{256, 1, 1}};
            }
          }
        }
        return {GetConvWorkGroupXYisMultipleK(kernel_params_.wave_size,
                                              grid_size_)};
    }
  }
}

bool IsConvWaveMemorySupported(const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno() && gpu_info.SupportsExtension("ucl_wave_memory")) {
    return true;
  }
  if (gpu_info.IsIntel() && gpu_info.SupportsExtension("ucl_wave_memory")) {
    return true;
  }
  if (gpu_info.IsPowerVR() && gpu_info.SupportsExtension("ucl_wave_memory")) {
    return true;
  }
  return false;
}

ConvWaveMemory CreateConvWaveMemory(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    CalculationsPrecision precision,
                                    const Convolution2DAttributes& attr,
                                    const BHWC* dst_shape) {
  ConvWaveMemory::ConvParams conv_params;
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.precision = precision;
  conv_params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  conv_params.x_kernel_is_1 = IsConvKernelXis1(attr);
  conv_params.y_kernel_is_1 = IsConvKernelYis1(attr);
  conv_params.groups_count = attr.groups;
  const auto& weights = GetFloatWeights(attr);
  ConvWaveMemory result(conv_params, gpu_info, weights.shape, dst_shape);
  const int2 stride = int2(attr.strides.w, attr.strides.h);
  const int2 padding =
      int2(-attr.padding.prepended.w, -attr.padding.prepended.h);
  const int2 dilation = int2(attr.dilations.w, attr.dilations.h);
  result.conv_params_.has_bias = !attr.bias.data.empty();
  result.GenerateCode(gpu_info, definition, stride, padding, dilation);
  result.UploadWeights(gpu_info, weights);
  result.UploadBias(gpu_info, attr.bias);
  return result;
}

ConvWaveMemory CreateConvWaveMemory(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    CalculationsPrecision precision,
                                    const FullyConnectedAttributes& attr,
                                    const BHWC* dst_shape) {
  ConvWaveMemory::ConvParams conv_params;
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.precision = precision;
  conv_params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  conv_params.x_kernel_is_1 = true;
  conv_params.y_kernel_is_1 = true;
  ConvWaveMemory result(conv_params, gpu_info, attr.weights.shape, dst_shape);
  result.conv_params_.has_bias = !attr.bias.data.empty();
  result.GenerateCode(gpu_info, definition);
  result.UploadWeights(gpu_info, attr.weights);
  result.UploadBias(gpu_info, attr.bias);
  return result;
}

ConvWaveMemory CreateConvWaveMemoryExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const Convolution2DAttributes& attr,
    const TensorDescriptor* bias, const BHWC* dst_shape,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  ConvWaveMemory::ConvParams conv_params;
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.precision = precision;
  conv_params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  conv_params.x_kernel_is_1 = IsConvKernelXis1(attr);
  conv_params.y_kernel_is_1 = IsConvKernelYis1(attr);
  conv_params.groups_count = attr.groups;
  conv_params.different_weights_for_height = different_weights_for_height;
  OHWI weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  if (different_weights_for_height) {
    conv_params.x_kernel_is_1 = true;
    conv_params.y_kernel_is_1 = true;
    weights_shape.w = 1;
    weights_shape.h = 1;
  }
  conv_params.runtime_check = runtime_check;
  conv_params.softmax_input_activation = src_exp != nullptr;
  conv_params.has_bias = bias != nullptr;
  ConvWaveMemory result(conv_params, gpu_info, weights_shape, dst_shape);
  const int2 stride = int2(attr.strides.w, attr.strides.h);
  const int2 padding =
      int2(-attr.padding.prepended.w, -attr.padding.prepended.h);
  const int2 dilation = int2(attr.dilations.w, attr.dilations.h);
  result.GenerateCode(gpu_info, definition, stride, padding, dilation);
  result.AddSrcBuffer("weights", GetBufferDescForWaveMemoryUpload(
                                     gpu_info, result.GetWeightsDescription()));
  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }
  if (src_exp) {
    result.AddSrcTensor("src_exp", *src_exp);
  }
  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    result.args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    result.args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.packed_groups.has_value()) {
    result.args_.AddInt("packed_params_offset",
                        runtime_check.packed_groups->params_offset);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = DataType::INT32;
    buffer_desc.element_size = 1;
    buffer_desc.memory_type = MemoryType::CONSTANT;
    result.AddSrcBuffer("params", buffer_desc);
  }
  return result;
}

bool SupportsConvWaveMemoryInt8(const GpuInfo& gpu_info) {
  if (!IsConvWaveMemorySupported(gpu_info)) {
    return false;
  }
  if (gpu_info.IsAdreno()) {
    if (!gpu_info.SupportsExtension("cl_qcom_dot_product8")) {
      return false;
    }
    // Adreno650 has very bad performance in int8, worse than fp16
    return gpu_info.adreno_info.adreno_gpu != AdrenoGpu::kAdreno650;
  }
  if (gpu_info.IsPowerVR() &&
      gpu_info.SupportsExtension("cl_img_pixel_subgroup_dot")) {
    return true;
  }
  return false;
}

PackedType GetConvWaveMemoryInt8SrcType(const GpuInfo& gpu_info,
                                        const BHWC& src_shape) {
  const int src_slices = DivideRoundUp(src_shape.c, 4);
  if (src_slices % 4 == 0 &&
      (gpu_info.adreno_info.IsAdreno8xx() || gpu_info.IsPowerVR())) {
    return PackedType::kUint8C16;
  } else {
    return PackedType::kUint8C4;
  }
}

ConvWaveMemory CreateConvWaveMemoryInt8(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Tensor<OHWI, DataType::INT8>& weights, const BHWC* dst_shape) {
  ConvWaveMemory::ConvParams conv_params;
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.weights_data_type = DataType::INT8;
  conv_params.x_kernel_is_1 = true;
  conv_params.y_kernel_is_1 = true;
  conv_params.has_bias = false;
  ConvWaveMemory result(conv_params, gpu_info, weights.shape, dst_shape);
  result.GenerateCode(gpu_info, definition);
  result.UploadWeights(gpu_info, weights);
  return result;
}

ConvWaveMemory CreateConvWaveMemoryInt8ExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OHWI& weights_shape, const BHWC* dst_shape) {
  ConvWaveMemory::ConvParams conv_params;
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.weights_data_type = DataType::INT8;
  conv_params.x_kernel_is_1 = true;
  conv_params.y_kernel_is_1 = true;
  conv_params.has_bias = false;
  ConvWaveMemory result(conv_params, gpu_info, weights_shape, dst_shape);
  result.GenerateCode(gpu_info, definition);
  result.AddSrcBuffer("weights", GetBufferDescForWaveMemoryUpload(
                                     gpu_info, result.GetWeightsDescription()));
  return result;
}

ConvWaveMemory CreateConvWaveMemoryInt8Grouped(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Tensor<OHWI, DataType::INT8>& weights, int group_size,
    const TensorDescriptor& src_params, const BHWC* dst_shape) {
  ConvWaveMemory::ConvParams conv_params;
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.weights_data_type = DataType::INT8;
  conv_params.x_kernel_is_1 = true;
  conv_params.y_kernel_is_1 = true;
  conv_params.has_bias = false;
  conv_params.blocked_quantization = true;
  ConvWaveMemory result(conv_params, gpu_info, weights.shape, dst_shape);
  result.args_.AddInt("quant_i_groups", weights.shape.i / group_size);
  result.args_.AddInt("quant_i_group_slices", group_size / 4);
  result.GenerateCode(gpu_info, definition);
  result.AddSrcTensor("src_params", src_params);
  result.UploadWeights(gpu_info, weights);
  return result;
}

}  // namespace ml_drift
