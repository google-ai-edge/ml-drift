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

#include "ml_drift/common/kernels/quantize_and_dequantize.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
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
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

GPUOperation CreateQuantizeAndOrDequantize(
    const OperationDef& definition, const QuantizeAndDequantizeAttributes& attr,
    bool is_quantize_only) {
  QuantizeAndDequantizeAttributes adjusted_attr = attr;
  const DataType src_type = definition.src_tensors[0].GetDataType();
  const bool is_fp16 = src_type == DataType::kFloat16;
  if (is_fp16 && attr.scale < 0.000062f) {
    // The smallest positive normal number for Half-precision floating-point
    // format is 2^-14 ~ 0.000062f. Therefore, if the scale is lesser than this
    // number, we just reset it accordingly.
    adjusted_attr.scale = 0.000062f;
  }

  ElementwiseDescriptor op_desc;
  op_desc.args.AddFloat("min", adjusted_attr.min, src_type);
  op_desc.args.AddFloat("max", adjusted_attr.max, src_type);
  op_desc.args.AddFloat("scale", adjusted_attr.scale, src_type);
  const std::string in_out_type = ToUclDataType(src_type, 4);
  std::string op_code;
  if (adjusted_attr.scale == 0.0f) {
    // scale == 0 means min == max, so every input maps to code 0, which
    // dequantizes to min. Emit those constants rather than dividing by scale,
    // which would produce NaN.
    if (is_quantize_only) {
      op_code = R"(
out_value = ucl::Init<$0>(0.0f);)";
    } else {
      op_code = R"(
out_value = ucl::Init<$0>(args.min);)";
    }
  } else {
    // Quantize around the zero point zp = -min / scale rather than computing
    // (x - min) / scale directly: the latter lands in [0, 2^bits - 1], where
    // F16 can only resolve 0.125-0.25, so ~5% of codes would round differently
    // than in F32. zp is split on the host into an integer part (added back
    // exactly at the end) and a fractional part in [-0.5, 0.5], so the
    // rounded value stays around [-2^(bits-1), 2^(bits-1)] with F16 spacing
    // <= 0.0625. Rounding uses floor(t + 0.5) (round half up) because t can be
    // negative, and round() would round negative ties away from zero, unlike
    // the direct form whose argument is never negative.
    const float inverse_scale = 1.0f / adjusted_attr.scale;
    const float zero_point = -adjusted_attr.min * inverse_scale;
    const float zero_point_int = std::round(zero_point);
    const float zero_point_frac = zero_point - zero_point_int;
    op_desc.args.AddFloat("inverse_scale", inverse_scale, src_type);
    op_desc.args.AddFloat("zero_point_frac_plus_half", zero_point_frac + 0.5f,
                          src_type);
    op_desc.args.AddFloat(
        "q_min",
        std::floor(adjusted_attr.min * inverse_scale + zero_point_frac + 0.5f),
        src_type);
    op_desc.args.AddFloat(
        "q_max",
        std::floor(adjusted_attr.max * inverse_scale + zero_point_frac + 0.5f),
        src_type);
    op_code = R"(
$0 clamped_value = min(ucl::Init<$0>(args.max), max(ucl::Init<$0>(args.min), in_value));
$0 centered_q = floor(clamped_value * ucl::Init<$0>(args.inverse_scale) + ucl::Init<$0>(args.zero_point_frac_plus_half));
centered_q = min(ucl::Init<$0>(args.q_max), max(ucl::Init<$0>(args.q_min), centered_q));)";
    op_code += '\n';
    if (is_quantize_only) {
      op_desc.args.AddFloat("zero_point_int", zero_point_int, src_type);
      op_code += R"(
out_value = centered_q + ucl::Init<$0>(args.zero_point_int);)";
    } else {
      // (q - zp) * scale = (centered_q - zero_point_frac) * scale.
      op_desc.args.AddFloat("dequant_offset",
                            -zero_point_frac * adjusted_attr.scale, src_type);
      op_code += R"(
out_value = centered_q * ucl::Init<$0>(args.scale) + ucl::Init<$0>(args.dequant_offset);)";
    }
  }
  op_desc.code = absl::Substitute(op_code, in_out_type);

  GPUOperation op = CreateGpuOperation(definition, std::move(op_desc));
  // Quantize: min, max, sub, mul/div = 4; dequantize adds mul, add (+2 = 6).
  op.flops_ = definition.dst_tensors[0].GetBHWDCShape().DimensionsProduct() *
              (is_quantize_only ? 4 : 6);
  return op;
}

GPUOperation CreateQuantizeAndDequantize(
    const OperationDef& definition,
    const QuantizeAndDequantizeAttributes& attr) {
  return CreateQuantizeAndOrDequantize(definition, attr, false);
}

GPUOperation CreateStaticRangeQuantization(
    const OperationDef& definition,
    const QuantizeAndDequantizeAttributes& attr) {
  return CreateQuantizeAndOrDequantize(definition, attr, true);
}

std::string ReduceCodeX2(const std::string& memory,
                         const std::string& reduction_id,
                         const std::string& reduce_value,
                         const std::string& calculate_func,
                         int reduction_size) {
  std::string c;
  c += "  {\n";
  c += "    int reduction_size = " + std::to_string(reduction_size) + ";\n";
  c += "    " + absl::Substitute(memory, reduction_id) + " = " + reduce_value +
       ";\n";
  c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  c += "    while (reduction_size > 1) {\n";
  c += "      int active_thread_limit = reduction_size / 2;\n";
  c += "      int offset = (reduction_size + 1) / 2;\n";
  c += "      if (local_id_s < active_thread_limit) {\n";
  c += "        " +
       absl::Substitute(calculate_func, reduction_id + " + offset") + "\n";
  c += "        " + absl::Substitute(memory, reduction_id) + " = " +
       reduce_value + ";\n";
  c += "      }\n";
  c += "      ucl::SyncThreads<WorkGroup, Local>();\n";
  c += "      reduction_size = offset;\n";
  c += "    }\n";
  c += "    " + reduce_value + " = " + absl::Substitute(memory, 0) + ";\n";
  c += "  }\n";
  return c;
}

std::string ReduceCodeX4(const std::string& memory,
                         const std::string& reduction_id,
                         const std::string& reduce_value,
                         const std::string& calculate_func,
                         int reduction_size) {
  std::string c;
  c += "  {\n";
  c += "    int reduction_size = " + std::to_string(reduction_size) + ";\n";
  c += "    " + absl::Substitute(memory, reduction_id) + " = " + reduce_value +
       ";\n";
  c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
  c += "    while (reduction_size >= 16) {\n";
  c += "      reduction_size /= 4;\n";
  c += "      if (local_id_s < reduction_size) {\n";
  std::string val_id;
  val_id = reduction_id + " + reduction_size";
  c += "        " + absl::Substitute(calculate_func, val_id) + "\n";
  val_id = reduction_id + " + reduction_size * 2";
  c += "        " + absl::Substitute(calculate_func, val_id) + "\n";
  val_id = reduction_id + " + reduction_size * 3";
  c += "        " + absl::Substitute(calculate_func, val_id) + "\n";
  c += "        " + absl::Substitute(memory, reduction_id) + " = " +
       reduce_value + ";\n";
  c += "      }\n";
  c += "      ucl::SyncThreads<WorkGroup, Local>();\n";
  c += "    }\n";
  c += "    " + reduce_value + " = " + absl::Substitute(memory, 0) + ";\n";
  c += "    for (int i = 1; i < reduction_size; ++i) {\n";
  c += "      " + absl::Substitute(calculate_func, "i") + "\n";
  c += "    }\n";
  c += "  }\n";
  return c;
}

bool IsPowerOfTwo(unsigned int x) { return (x & (x - 1)) == 0; }

// reduction, all threads inside workgroup must execute this code
std::string ReduceCode(const std::string& memory,
                       const std::string& reduction_id,
                       const std::string& reduce_value,
                       const std::string& calculate_func, int reduction_size) {
  if (IsPowerOfTwo(reduction_size) && reduction_size >= 16) {
    return ReduceCodeX4(memory, reduction_id, reduce_value, calculate_func,
                        reduction_size);
  } else {
    return ReduceCodeX2(memory, reduction_id, reduce_value, calculate_func,
                        reduction_size);
  }
}

std::string GetCode(const int3& work_group_size,
                    const TensorDescriptor& src_desc, bool slices_first,
                    const PackedType dst_type, int channels, bool calculate_sum,
                    bool is_params_fp16, bool symmetric,
                    bool cap_symmetric_scale) {
  const int wg_reduction_size =
      slices_first ? work_group_size.x : work_group_size.z;
  const int wg_spatial_size = slices_first
                                  ? work_group_size.y * work_group_size.z
                                  : work_group_size.x * work_group_size.y;
  const bool work_group_reduction = wg_reduction_size != 1;
  const int slices = DivideRoundUp(channels, 4);
  const int reduction_groups = DivideRoundUp(slices, wg_reduction_size);
  // test version ,intentionally disabled
  bool register_cache = slices == 0;
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (work_group_reduction) {
    if (symmetric) {
      c += "__local SrcType shared_mem[WG_REDUCTION_SIZE][WG_SPATIAL_SIZE];\n";
    } else {
      c += "__local SrcType2 shared_mem[WG_REDUCTION_SIZE][WG_SPATIAL_SIZE];\n";
    }
  }
  if (dst_type == PackedType::kInt8W4C4 || dst_type == PackedType::kInt8C16 ||
      dst_type == PackedType::kInt4C32) {
    c += "__local char4 outputs[WG_REDUCTION_SIZE][WG_SPATIAL_SIZE];\n";
  } else if (dst_type == PackedType::kUint8W4C4 ||
             dst_type == PackedType::kUint8C16 ||
             dst_type == PackedType::kUint4C32) {
    c += "__local uchar4 outputs[WG_REDUCTION_SIZE][WG_SPATIAL_SIZE];\n";
  }
  if (slices_first) {
    c += "  int linear_wb = ucl::GetGlobalId<1>();\n";
    c += "  int linear_h = ucl::GetGlobalId<2>();\n";
    if (work_group_reduction) {
      c += "  int local_id_s = ucl::GetLocalId<0>();\n";
      c += "  int local_id = ucl::GetLocalId<2>() * ucl::GetGroupSize<1>() + "
           "ucl::GetLocalId<1>();\n";
    }
  } else {
    c += "  int linear_wb = ucl::GetGlobalId<0>();\n";
    c += "  int linear_h = ucl::GetGlobalId<1>();\n";
    if (work_group_reduction) {
      c += "  int local_id_s = ucl::GetLocalId<2>();\n";
      c += "  int local_id = ucl::GetLocalId<1>() * ucl::GetGroupSize<0>() + "
           "ucl::GetLocalId<0>();\n";
    }
  }
  if (!work_group_reduction) {
    c += "  int local_id_s = 0;\n";
  }
  if (dst_type == PackedType::kInt8W4C4 || dst_type == PackedType::kUint8W4C4) {
    c += "  int sub_x = linear_wb % 4;\n";
    c += "  linear_wb = linear_wb / 4;\n";
  }
  if (src_desc.HasAxis(Axis::kBatch)) {
    c += "  int X = linear_wb / args.src_tensor.Batch();\n";
    c += "  int B = linear_wb % args.src_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.dst_params.SetBatchRef(B);\n";
  } else {
    c += "  int X = linear_wb;\n";
  }
  if (dst_type == PackedType::kInt8W4C4 || dst_type == PackedType::kUint8W4C4) {
    c += "  X = X * 4 + sub_x;\n";
  }
  c += "  int Y = linear_h;\n";
  if (!work_group_reduction) {
    c += "  if (X >= args.src_tensor.Width()) { return; }\n";
    c += "  if (Y >= args.src_tensor.Height()) { return; }\n";
  }
  if (symmetric) {
    c += "  SrcType abs_max = ucl::Init<SrcType>(0.0f);\n";
  } else {
    c += "  SrcType min_value = args.src_tensor.Read(0, 0, 0).x;\n";
    c += "  SrcType max_value = min_value;\n";
  }
  c += "  int x_clamped = min(X, args.src_tensor.Width() - 1);\n";
  c += "  int y_clamped = min(Y, args.src_tensor.Height() - 1);\n";
  if (register_cache) {
    c += "  SrcType4 src_s[REDUCTION_GROUPS];\n";
    c += "  for (int i = 0; i < REDUCTION_GROUPS; ++i) {\n";
    c += "    int S = local_id_s + i * WG_REDUCTION_SIZE;\n";
    c += "    int s_clamped = min(S, args.src_tensor.Slices() - 1);\n";
    c += "    src_s[i] = args.src_tensor.Read(x_clamped, y_clamped, "
         "s_clamped);\n";
    c += "    SrcType4 t = src_s[i];\n";
  } else {
    c += "  for (int S = local_id_s; S < args.src_tensor.Slices(); S += "
         "WG_REDUCTION_SIZE) {\n";
    c += "    SrcType4 t = args.src_tensor.Read(x_clamped, y_clamped, S);\n";
  }
  if (symmetric) {
    c += "    SrcType4 abs_t = fabs(t);\n";
    if (channels % 4 == 0) {
      c += "    abs_max = max(abs_max, max(max(abs_t.x, abs_t.y), "
           "max(abs_t.z, abs_t.w)));\n";
    } else {
      c += R"(
    abs_max = max(abs_max, abs_t.x);
    if (S * 4 + 1 < args.src_tensor.Channels()) {
      abs_max = max(abs_max, abs_t.y);
    }
    if (S * 4 + 2 < args.src_tensor.Channels()) {
      abs_max = max(abs_max, abs_t.z);
    }
    if (S * 4 + 3 < args.src_tensor.Channels()) {
      abs_max = max(abs_max, abs_t.w);
    }
)";
    }
  } else if (channels % 4 == 0) {
    c += R"(
    min_value = min(min_value, t.x);
    max_value = max(max_value, t.x);
    min_value = min(min_value, t.y);
    max_value = max(max_value, t.y);
    min_value = min(min_value, t.z);
    max_value = max(max_value, t.z);
    min_value = min(min_value, t.w);
    max_value = max(max_value, t.w);
)";
  } else {
    c += R"(
    min_value = min(min_value, t.x);
    max_value = max(max_value, t.x);
    if (S * 4 + 1 < args.src_tensor.Channels()) {
      min_value = min(min_value, t.y);
      max_value = max(max_value, t.y);
    }
    if (S * 4 + 2 < args.src_tensor.Channels()) {
      min_value = min(min_value, t.z);
      max_value = max(max_value, t.z);
    }
    if (S * 4 + 3 < args.src_tensor.Channels()) {
      min_value = min(min_value, t.w);
      max_value = max(max_value, t.w);
    }
)";
  }
  c += "  }\n";
  if (symmetric) {
    if (work_group_reduction) {
      const std::string calculate_func =
          "abs_max = max(abs_max, shared_mem[$0][local_id]);";
      c += ReduceCode(
          /*memory=*/"shared_mem[$0][local_id]", /*reduction_id=*/"local_id_s",
          /*reduce_value=*/"abs_max", calculate_func, wg_reduction_size);
    }
    c += R"(  float abs_max_f32 = ucl::Convert<float>(abs_max);
  float inv_scale = 0.0f;
  float scale = 0.0f;
  float offset = 0.0f;
  if (abs_max_f32 > 0.0f) {
    inv_scale = max(args.min_scale, abs_max_f32 / args.symmetric_half_range);
)";
    if (cap_symmetric_scale) {
      c += "    inv_scale = min(inv_scale, args.max_scale);\n";
    }
    c += "    scale = 1.0f / inv_scale;\n";
    if (!IsSigned(dst_type)) {
      c += "    offset = -inv_scale * args.zero_point;\n";
    }
    c += "  }\n";
  } else {
    if (work_group_reduction) {
      c += "  SrcType2 min_max_value = ucl::Init<SrcType2>(min_value, "
           "max_value);\n";
      const std::string calculate_func =
          R"({SrcType2 temp_value = shared_mem[$0][local_id];
  min_max_value.x = min(min_max_value.x, temp_value.x);
  min_max_value.y = max(min_max_value.y, temp_value.y);})";
      c += ReduceCode(
          /*memory=*/"shared_mem[$0][local_id]", /*reduction_id=*/"local_id_s",
          /*reduce_value=*/"min_max_value", calculate_func, wg_reduction_size);
      c += "  min_value = min_max_value.x;\n";
      c += "  max_value = min_max_value.y;\n";
    }
    c += "  float src_range = ucl::Convert<float>(max_value - min_value);\n";
    c += "  float offset = ucl::Convert<float>(min_value);\n";
    c += "  float scale = ucl::Convert<float>(args.dst_range) / src_range;\n";
    c += "  float inv_scale = src_range / "
         "ucl::Convert<float>(args.dst_range);\n";
    if (IsSigned(dst_type)) {
      c += "  offset += src_range * args.src_range_middle;\n";
    }
  }
  c += "  int sum = 0;\n";
  if (dst_type == PackedType::kInt8W4C4 || dst_type == PackedType::kUint8W4C4 ||
      dst_type == PackedType::kInt8C16 || dst_type == PackedType::kUint8C16 ||
      dst_type == PackedType::kInt4C32 || dst_type == PackedType::kUint4C32) {
    c += "  for (int i = 0; i < REDUCTION_GROUPS; ++i) {\n";
    c += "    int S = local_id_s + i * WG_REDUCTION_SIZE;\n";
    c += "    int s_clamped = min(S, args.src_tensor.Slices() - 1);\n";
    c += "    float4 src = args.src_tensor.Read<float>(x_clamped, y_clamped, "
         "s_clamped);\n";
  } else {
    if (register_cache) {
      c += "  for (int i = 0; i < REDUCTION_GROUPS; ++i) {\n";
      c += "    int S = local_id_s + i * WG_REDUCTION_SIZE;\n";
      c += "    int s_clamped = min(S, args.src_tensor.Slices() - 1);\n";
      c += "    float4 src = ucl::Convert<float4>(src_s[i]);\n";
    } else {
      c += "  for (int S = local_id_s; S < args.src_tensor.Slices(); S += "
           "WG_REDUCTION_SIZE) {\n";
      c += "    float4 src = args.src_tensor.Read<float>(x_clamped, y_clamped, "
           "S);\n";
    }
  }
  if (symmetric) {
    c += R"(
    src = floor(src * scale + ucl::Init<float4>(args.zero_point_plus_half));
    src = min(ucl::Init<float4>(ucl::Convert<float>(args.dst_range)),
              max(ucl::Init<float4>(0.0f), src));
    int4 src_quantized = ucl::Convert<int4>(src);
)";
  } else {
    c += R"(
    src = (src - ucl::Convert<float>(min_value)) * scale;
    src += ucl::Init<float4>(0.5f);  // ro replicate round() that used on CPU
    int4 src_quantized = ucl::Convert<int4>(src);
    src_quantized = min(ucl::Init<int4>(args.dst_range), src_quantized);
    src_quantized = max(ucl::Init<int4>(0), src_quantized);
)";
  }
  if (IsSigned(dst_type)) {
    c += "    src_quantized -= ucl::Init<int4>((args.dst_range + 1) / 2);\n";
  }
  if (calculate_sum) {
    c += "    if (S < args.src_tensor.Slices()) {\n";
    if (channels % 4 == 0) {
      c += R"(
      sum += src_quantized.x;
      sum += src_quantized.y;
      sum += src_quantized.z;
      sum += src_quantized.w;
)";
    } else {
      c += R"(
      sum += src_quantized.x;
      sum += S * 4 + 1 < args.src_tensor.Channels() ? src_quantized.y : 0;
      sum += S * 4 + 2 < args.src_tensor.Channels() ? src_quantized.z : 0;
      sum += S * 4 + 3 < args.src_tensor.Channels() ? src_quantized.w : 0;
)";
    }
    c += "    }\n";
  }
  if (dst_type == PackedType::kInt8W4C4) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    outputs[local_id_s][local_id] = "
         "ucl::Convert<char4>(src_quantized);\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    if (local_id % 4 == 0) {\n";
    c += "      char4 p0 = outputs[local_id_s][local_id + 0];\n";
    c += "      char4 p1 = outputs[local_id_s][local_id + 1];\n";
    c += "      char4 p2 = outputs[local_id_s][local_id + 2];\n";
    c += "      char4 p3 = outputs[local_id_s][local_id + 3];\n";
    c += "      int4 result;\n";
    c += "      result.x = ucl::Reinterpret<char4, int>(p0);\n";
    c += "      result.y = ucl::Reinterpret<char4, int>(p1);\n";
    c += "      result.z = ucl::Reinterpret<char4, int>(p2);\n";
    c += "      result.w = ucl::Reinterpret<char4, int>(p3);\n";
    c += "      if (X / 4 < args.dst_tensor.Width() && Y < "
         "args.dst_tensor.Height() && S < args.dst_tensor.Slices()) {\n";
    c += "        args.dst_tensor.Write(result, X / 4, Y, S);\n";
    c += "      }\n";
    c += "    }\n";
  } else if (dst_type == PackedType::kUint8W4C4) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    outputs[local_id_s][local_id] = "
         "ucl::Convert<uchar4>(src_quantized);\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    if (local_id % 4 == 0) {\n";
    c += "      uchar4 p0 = outputs[local_id_s][local_id + 0];\n";
    c += "      uchar4 p1 = outputs[local_id_s][local_id + 1];\n";
    c += "      uchar4 p2 = outputs[local_id_s][local_id + 2];\n";
    c += "      uchar4 p3 = outputs[local_id_s][local_id + 3];\n";
    c += "      uint4 result;\n";
    c += "      result.x = ucl::Reinterpret<uchar4, uint>(p0);\n";
    c += "      result.y = ucl::Reinterpret<uchar4, uint>(p1);\n";
    c += "      result.z = ucl::Reinterpret<uchar4, uint>(p2);\n";
    c += "      result.w = ucl::Reinterpret<uchar4, uint>(p3);\n";
    c += "      if (X / 4 < args.dst_tensor.Width() && Y < "
         "args.dst_tensor.Height() && S < args.dst_tensor.Slices()) {\n";
    c += "        args.dst_tensor.Write(result, X / 4, Y, S);\n";
    c += "      }\n";
    c += "    }\n";
  } else if (dst_type == PackedType::kInt8C16) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    outputs[local_id_s][local_id] = "
         "ucl::Convert<char4>(src_quantized);\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    if (local_id_s % 4 == 0) {\n";
    c += "      char4 p0 = outputs[local_id_s + 0][local_id];\n";
    c += "      char4 p1 = outputs[local_id_s + 1][local_id];\n";
    c += "      char4 p2 = outputs[local_id_s + 2][local_id];\n";
    c += "      char4 p3 = outputs[local_id_s + 3][local_id];\n";
    c += "      int4 result;\n";
    c += "      result.x = ucl::Reinterpret<char4, int>(p0);\n";
    c += "      result.y = ucl::Reinterpret<char4, int>(p1);\n";
    c += "      result.z = ucl::Reinterpret<char4, int>(p2);\n";
    c += "      result.w = ucl::Reinterpret<char4, int>(p3);\n";
    c += "      if (X < args.dst_tensor.Width() && Y < "
         "args.dst_tensor.Height() && S / 4 < args.dst_tensor.Slices()) {\n";
    c += "        args.dst_tensor.Write(result, X, Y, S / 4);\n";
    c += "      }\n";
    c += "    }\n";
  } else if (dst_type == PackedType::kUint8C16) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    outputs[local_id_s][local_id] = "
         "ucl::Convert<uchar4>(src_quantized);\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    if (local_id_s % 4 == 0) {\n";
    c += "      uchar4 p0 = outputs[local_id_s + 0][local_id];\n";
    c += "      uchar4 p1 = outputs[local_id_s + 1][local_id];\n";
    c += "      uchar4 p2 = outputs[local_id_s + 2][local_id];\n";
    c += "      uchar4 p3 = outputs[local_id_s + 3][local_id];\n";
    c += "      uint4 result;\n";
    c += "      result.x = ucl::Reinterpret<uchar4, uint>(p0);\n";
    c += "      result.y = ucl::Reinterpret<uchar4, uint>(p1);\n";
    c += "      result.z = ucl::Reinterpret<uchar4, uint>(p2);\n";
    c += "      result.w = ucl::Reinterpret<uchar4, uint>(p3);\n";
    c += "      if (X < args.dst_tensor.Width() && Y < "
         "args.dst_tensor.Height() && S / 4 < args.dst_tensor.Slices()) {\n";
    c += "        args.dst_tensor.Write(result, X, Y, S / 4);\n";
    c += "      }\n";
    c += "    }\n";
  } else if (dst_type == PackedType::kInt4C32) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    outputs[local_id_s][local_id] = "
         "ucl::Convert<char4>(src_quantized);\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    if (local_id_s % 8 == 0) {\n";
    c += "      uint4 p0 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "0][local_id]));\n";
    c += "      uint4 p1 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "1][local_id]));\n";
    c += "      uint4 p2 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "2][local_id]));\n";
    c += "      uint4 p3 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "3][local_id]));\n";
    c += "      uint4 p4 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "4][local_id]));\n";
    c += "      uint4 p5 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "5][local_id]));\n";
    c += "      uint4 p6 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "6][local_id]));\n";
    c += "      uint4 p7 = "
         "ucl::Convert<uint4>(ucl::Convert<int4>(outputs[local_id_s + "
         "7][local_id]));\n";
    c += "      uint4 uresult;\n";
    c += "      uresult.x = ((p1.w & 15u) << 28u) | ((p1.z & 15u) << 24u) | "
         "((p1.y "
         "& 15u) << 20u) | ((p1.x & 15u) << 16u) | ((p0.w & 15u) << 12u) | "
         "((p0.z & 15u) << 8u) | ((p0.y "
         "& 15u) << 4u) | (p0.x & 15u);\n";
    c += "      uresult.y = ((p3.w & 15u) << 28u) | ((p3.z & 15u) << 24u) | "
         "((p3.y "
         "& 15u) << 20u) | ((p3.x & 15u) << 16u) | ((p2.w & 15u) << 12u) | "
         "((p2.z & 15u) << 8u) | ((p2.y "
         "& 15u) << 4u) | (p2.x & 15u);\n";
    c += "      uresult.z = ((p5.w & 15u) << 28u) | ((p5.z & 15u) << 24u) | "
         "((p5.y "
         "& 15u) << 20u) | ((p5.x & 15u) << 16u) | ((p4.w & 15u) << 12u) | "
         "((p4.z & 15u) << 8u) | ((p4.y "
         "& 15u) << 4u) | (p4.x & 15u);\n";
    c += "      uresult.w = ((p7.w & 15u) << 28u) | ((p7.z & 15u) << 24u) | "
         "((p7.y "
         "& 15u) << 20u) | ((p7.x & 15u) << 16u) | ((p6.w & 15u) << 12u) | "
         "((p6.z & 15u) << 8u) | ((p6.y "
         "& 15u) << 4u) | (p6.x & 15u);\n";
    c += "      int4 result = ucl::Convert<int4>(uresult);\n";
    c += "      if (X < args.dst_tensor.Width() && Y < "
         "args.dst_tensor.Height() && S / 8 < args.dst_tensor.Slices()) {\n";
    c += "        args.dst_tensor.Write(result, X, Y, S / 8);\n";
    c += "      }\n";
    c += "    }\n";
  } else if (dst_type == PackedType::kUint4C32) {
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    outputs[local_id_s][local_id] = "
         "ucl::Convert<char4>(src_quantized);\n";
    c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "    if (local_id_s % 8 == 0) {\n";
    c += "      uint4 p0 = ucl::Convert<uint4>(outputs[local_id_s + "
         "0][local_id]);\n";
    c += "      uint4 p1 = ucl::Convert<uint4>(outputs[local_id_s + "
         "1][local_id]);\n";
    c += "      uint4 p2 = ucl::Convert<uint4>(outputs[local_id_s + "
         "2][local_id]);\n";
    c += "      uint4 p3 = ucl::Convert<uint4>(outputs[local_id_s + "
         "3][local_id]);\n";
    c += "      uint4 p4 = ucl::Convert<uint4>(outputs[local_id_s + "
         "4][local_id]);\n";
    c += "      uint4 p5 = ucl::Convert<uint4>(outputs[local_id_s + "
         "5][local_id]);\n";
    c += "      uint4 p6 = ucl::Convert<uint4>(outputs[local_id_s + "
         "6][local_id]);\n";
    c += "      uint4 p7 = ucl::Convert<uint4>(outputs[local_id_s + "
         "7][local_id]);\n";
    c += "      uint4 uresult;\n";
    c += "      uresult.x = ((p1.w & 15u) << 28u) | ((p1.z & 15u) << 24u) | "
         "((p1.y "
         "& 15u) << 20u) | ((p1.x & 15u) << 16u) | ((p0.w & 15u) << 12u) | "
         "((p0.z & 15u) << 8u) | ((p0.y "
         "& 15u) << 4u) | (p0.x & 15u);\n";
    c += "      uresult.y = ((p3.w & 15u) << 28u) | ((p3.z & 15u) << 24u) | "
         "((p3.y "
         "& 15u) << 20u) | ((p3.x & 15u) << 16u) | ((p2.w & 15u) << 12u) | "
         "((p2.z & 15u) << 8u) | ((p2.y "
         "& 15u) << 4u) | (p2.x & 15u);\n";
    c += "      uresult.z = ((p5.w & 15u) << 28u) | ((p5.z & 15u) << 24u) | "
         "((p5.y "
         "& 15u) << 20u) | ((p5.x & 15u) << 16u) | ((p4.w & 15u) << 12u) | "
         "((p4.z & 15u) << 8u) | ((p4.y "
         "& 15u) << 4u) | (p4.x & 15u);\n";
    c += "      uresult.w = ((p7.w & 15u) << 28u) | ((p7.z & 15u) << 24u) | "
         "((p7.y "
         "& 15u) << 20u) | ((p7.x & 15u) << 16u) | ((p6.w & 15u) << 12u) | "
         "((p6.z & 15u) << 8u) | ((p6.y "
         "& 15u) << 4u) | (p6.x & 15u);\n";
    c += "      if (X < args.dst_tensor.Width() && Y < "
         "args.dst_tensor.Height() && S / 8 < args.dst_tensor.Slices()) {\n";
    c += "        args.dst_tensor.Write(uresult, X, Y, S / 8);\n";
    c += "      }\n";
    c += "    }\n";
  } else {
    c += R"(
    if (X < args.dst_tensor.Width() && Y < args.dst_tensor.Height()) {
      args.dst_tensor::type dst_val = ucl::Convert<args.dst_tensor::type>(src_quantized);
      args.dst_tensor.Write(dst_val, X, Y, S);
    }
)";
  }
  c += "  }\n";
  if (calculate_sum && work_group_reduction) {
    c += "__local int shared_sum[WG_REDUCTION_SIZE][WG_SPATIAL_SIZE];\n";
    c += ReduceCode(
        /*memory=*/"shared_sum[$0][local_id]", /*reduction_id=*/"local_id_s",
        /*reduce_value=*/"sum",
        /*calculate_func=*/"sum += shared_sum[$0][local_id];",
        wg_reduction_size);
  }
  c += R"(
  if (X >= args.dst_params.Width()) { return; }
  if (Y >= args.dst_params.Height()) { return; }
  if (local_id_s == 0) {
    args.dst_params::type final_value;
    final_value.x = ucl::Convert<args.dst_params::scalar_type>(inv_scale);
    final_value.y = ucl::Convert<args.dst_params::scalar_type>(offset);
    )";
  if (calculate_sum) {
    if (is_params_fp16) {
      c += "sum = min(sum, args.max_f16);";
      c += "sum = max(sum, -args.max_f16);";
    }
    c += R"(
    final_value.z = ucl::Convert<args.dst_params::scalar_type>(sum);
)";
  }
  c += R"(
    args.dst_params.Write(final_value, X, Y, 0);
  }
}
)";
  c = absl::StrReplaceAll(
      c, {{"SrcType", ToUclDataType(src_desc.GetDataType(), 1)},
          {"WG_REDUCTION_SIZE", std::to_string(wg_reduction_size)},
          {"WG_SPATIAL_SIZE", std::to_string(wg_spatial_size)},
          {"REDUCTION_GROUPS", std::to_string(reduction_groups)}});
  return c;
}

class Quantization : public GPUOperation {
 public:
  Quantization() = default;
  explicit Quantization(const OperationDef& definition, PackedType dst_type,
                        const GpuInfo& gpu_info, const BHWC& shape,
                        bool calculate_sum, bool symmetric)
      : dst_type_(dst_type) {
    work_group_reduction_ = true;
    int wg_reduction_size = 1;
    int wg_spatial_size = 1;
    if (work_group_reduction_) {
      if (shape.w >= 512 && !gpu_info.IsIntel()) {
        wg_reduction_size = 8;
        wg_spatial_size = 32;
      } else if (shape.w >= 256 && !gpu_info.IsIntel()) {
        wg_reduction_size = 16;
        wg_spatial_size = 16;
      } else if (shape.w >= 128) {
        wg_reduction_size = 32;
        wg_spatial_size = 8;
      } else {
        wg_reduction_size = 32;
        wg_spatial_size = 4;
      }
      if (gpu_info.IsApple() &&
          gpu_info.apple_info.gpu_family >= AppleInfo::Family::kApple10) {
        wg_reduction_size = 64;
        wg_spatial_size = 4;
      }
    }
    if (dst_type == PackedType::kInt8W4C4 ||
        dst_type == PackedType::kUint8W4C4) {
      work_group_reduction_ = true;
      if (wg_spatial_size % 4 != 0) {
        wg_spatial_size = AlignByN(wg_spatial_size, 4);
      }
    }
    if (dst_type == PackedType::kInt8C16 || dst_type == PackedType::kUint8C16) {
      work_group_reduction_ = true;
      if (wg_reduction_size % 4 != 0) {
        wg_reduction_size = AlignByN(wg_reduction_size, 4);
      }
    }
    if (dst_type == PackedType::kInt4C32 || dst_type == PackedType::kUint4C32) {
      work_group_reduction_ = true;
      if (wg_reduction_size % 8 != 0) {
        wg_reduction_size = AlignByN(wg_reduction_size, 4);
      }
    }
    if (work_group_reduction_) {
      if (slices_first_) {
        work_group_size_ = int3(wg_reduction_size, wg_spatial_size, 1);
      } else {
        work_group_size_ = int3(wg_spatial_size, 1, wg_reduction_size);
      }
    } else {
      work_group_size_ = int3(128, 1, 1);
    }
    if (gpu_info.IsPowerVR() || gpu_info.IsMali()) {
      compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
    }
    AddSrcTensor("src_tensor", definition.src_tensors[0]);
    AddDstTensor("dst_tensor", definition.dst_tensors[0]);
    AddDstTensor("dst_params", definition.dst_tensors[1]);
    const int range = 1 << GetTypeSizeInBits(dst_type);
    args_.AddInt("dst_range", range - 1);
    const bool is_params_fp16 =
        definition.dst_tensors[1].GetDataType() == DataType::kFloat16;
    bool cap_symmetric_scale = false;
    if (symmetric) {
      // Clamp params to normal FP16 range if stored as FP16, or if src is FP16
      // (downstream ops may cast params to the source type).
      const bool scale_may_be_fp16 =
          definition.src_tensors[0].GetDataType() == DataType::kFloat16 ||
          is_params_fp16;
      // Avoid FP16 subnormals (which GPUs may flush to zero) and 1 / inv_scale
      // overflow in FP32.
      const float min_scale = scale_may_be_fp16 ? kMinNormalHalf : 1.0e-30f;
      args_.AddFloat("min_scale", min_scale);
      args_.AddFloat("symmetric_half_range", static_cast<float>(range / 2 - 1));
      if (!IsSigned(dst_type)) {
        const float zero_point = static_cast<float>(range / 2);
        args_.AddFloat("zero_point", zero_point);
        if (scale_may_be_fp16) {
          // Cap inv_scale so offset = -inv_scale * zero_point doesn't overflow
          // FP16 when max(|x|) is near kMaxHalf.
          args_.AddFloat("max_scale", kMaxHalf / zero_point);
          cap_symmetric_scale = true;
        }
      }
      args_.AddFloat("zero_point_plus_half",
                     static_cast<float>(range / 2) + 0.5f);
    } else {
      args_.AddFloat("src_range_middle",
                     static_cast<float>(range / 2) / (range - 1));
    }
    if (is_params_fp16) {
      // sum is int, so type match.
      args_.AddInt("max_f16", static_cast<int>(kMaxHalf));
    }
    code_ = GetCode(work_group_size_, definition.src_tensors[0], slices_first_,
                    dst_type, shape.c, calculate_sum, is_params_fp16, symmetric,
                    cap_symmetric_scale);
  }

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override {
    int wb_size = src_[0]->Width() * src_[0]->Batch();
    if (dst_type_ == PackedType::kInt8W4C4 ||
        dst_type_ == PackedType::kUint8W4C4) {
      wb_size = AlignByN(src_[0]->Width(), 4) * src_[0]->Batch();
    }
    const int h_size = src_[0]->Height();
    if (slices_first_) {
      const int s_size = work_group_reduction_ ? work_group_size_.x : 1;
      return int3(s_size, wb_size, h_size);
    } else {
      const int s_size = work_group_reduction_ ? work_group_size_.z : 1;
      return int3(wb_size, h_size, s_size);
    }
  }

  // Move only
  Quantization(Quantization&& kernel) = default;
  Quantization& operator=(Quantization&& kernel) = default;
  Quantization(const Quantization&) = delete;
  Quantization& operator=(const Quantization&) = delete;

 private:
  bool work_group_reduction_ = true;
  bool slices_first_ = false;
  PackedType dst_type_;
};

std::unique_ptr<GPUOperation> CreateQuantization(const OperationDef& definition,
                                                 PackedType dst_type,
                                                 const GpuInfo& gpu_info,
                                                 const BHWC& shape,
                                                 bool calculate_sum,
                                                 bool symmetric) {
  auto op = std::make_unique<Quantization>(Quantization(
      definition, dst_type, gpu_info, shape, calculate_sum, symmetric));
  // Min/max or abs-max reduction (2) + quantize scale/zero_point/clamp (3) = 5.
  op->flops_ = shape.DimensionsProduct() * 5;
  return op;
}

// Wzp == 0:
//   Si = Us * Ssc + Szp
//   Wi = Iw * Wsc
// sum(Si * Wi) =
// = sum((Us * Ssc + Szp) * (Iw * Wsc))
// = Wsc * sum((Us * Ssc + Szp) * Iw)
// = Wsc * sum(Us * Ssc * Iw + Szp * Iw)
// = Wsc * sum(Us * Ssc * Iw + Szp * Iw)
// = Wsc * sum(Us * Ssc * Iw) + Wsc * sum(Szp * Iw)
// = Wsc * Ssc * sum(Us * Iw) + Wsc * Szp * sum(Iw)
// sum(Us * Iw) is conv result (in_value)
//
// Wzp != 0:
//   Si = Us * Ssc + Szp
//   Wi = Iw * Wsc + Wzp
// sum(Si * Wi) =
// = sum((Us * Ssc + Szp) * (Iw * Wsc + Wzp))
// = sum(Us * Ssc * Iw * Wsc + Szp * Iw * Wsc + Us * Ssc * Wzp + Szp * Wzp)
// = Wsc * Ssc * sum(Us * Iw) + Wsc * Szp * sum(Iw)
//   + Ssc * Wzp * sum(Us) + Szp * Wzp * N
// sum(Us * Iw) is conv result (in_value)
// Wsc * Ssc * sum(Us * Iw) + Wsc * Szp * sum(Iw) - the same for Wzp == 0
// + Ssc * Wzp * sum(Us) + Szp * Wzp * N - additional part
GPUOperation CreateDequantization(const OHWI& weights_shape,
                                  const GpuInfo& gpu_info,
                                  const TensorDescriptor& src,
                                  const TensorDescriptor& dst,
                                  const TensorDescriptor& src_scale_zp_sum,
                                  const TensorDescriptor& weights_sum,
                                  const TensorDescriptor& weights_scale,
                                  const TensorDescriptor* weights_zero_point) {
  std::string c;
  c += "  float4 src_params = args.params_tensor.Read<float>(X_COORD, Y_COORD, "
       "0";
  if (src_scale_zp_sum.HasAxis(Axis::kBatch)) {
    c += ", B_COORD";
  }
  c += ");\n";
  // Using extra checks to reduce register usage on PowerVR.
  const bool pvr_checks =
      gpu_info.IsPowerVR() &&
      gpu_info.SupportsExtension("cl_img_pixel_subgroup_dot");
  if (src.GetDataType() == DataType::kUint32) {
    // Adjusting sum(Su * Wu) to sum(Su * Wi)
    // Uint8MathForInt8 used only with src as uint(Su)
    //   sum(Su * Wi) = sum(Su * (Wu - 128)) =
    //   sum(Su * Wu) - sum(Su * 128) = sum(Su * Wu) - 128 * sum(Su)
    //   sum(Su * Wu) is rid
    c += "  int4 ri32 = ucl::Convert<int4>(in_value);\n";
    if (pvr_checks) {
      c += "  if (S_COORD > -1) {\n";
      c += "    ri32 = ri32 - 128 * ucl::Convert<int>(src_params.z);\n";
      c += "  }\n";
      c += "  float4 resf32 = ucl::Convert<float4>(ri32);\n";
    } else {
      c += "  float4 resf32 = ucl::Convert<float4>(ri32 - 128 * "
           "ucl::Convert<int>(src_params.z));\n";
    }
  } else {
    c += "  float4 resf32 = ucl::Convert<float4>(in_value);\n";
  }
  c += R"(
  float4 weights_scale = args.weights_scale.Read<float>(S_COORD);
  float src_scale = src_params.x;
  float src_zero_point = src_params.y;
  resf32 *= weights_scale * src_scale;
)";
  if (pvr_checks) {
    c += "  if (S_COORD > -2) {\n";
  }
  c += R"(
  float4 weights_sum = ucl::Convert<float4>(args.weights_sum.Read(S_COORD));
  resf32 += weights_scale * src_zero_point * weights_sum;
)";
  if (pvr_checks) {
    c += "  }\n";
  }
  if (weights_zero_point != nullptr) {
    if (pvr_checks) {
      c += "  if (S_COORD > -3) {\n";
    }
    c += R"(
  float4 weights_zero_point = args.weights_zero_point.Read<float>(S_COORD);
  float src_sum = src_params.z;
  resf32 += src_scale * weights_zero_point * src_sum;
  resf32 += src_zero_point * weights_zero_point * args.in_channels;
)";
    if (pvr_checks) {
      c += "  }\n";
    }
  }
  c += "  out_value = ucl::Convert<" + ToUclDataType(dst.GetDataType(), 4) +
       ">(resf32);\n";

  ElementwiseDescriptor op_desc;
  op_desc.args.AddFloat("in_channels", weights_shape.i);
  op_desc.code = c;

  GPUOperation op = CreateGpuOperation(src, dst, std::move(op_desc));
  op.AddSrcTensor("params_tensor", src_scale_zp_sum);
  // The default case cause compilation failure in combination with complex main
  // kernel(convolution) in Adreno with driver major version lower than 45.
  if (gpu_info.IsAdreno() &&
      weights_sum.GetStorageType() == TensorStorageType::kBuffer) {
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = weights_sum.GetDataType();
    buffer_desc.element_size = 4;
    buffer_desc.memory_type = MemoryType::kConstant;
    op.AddSrcBuffer("weights_sum", buffer_desc);
  } else {
    op.AddSrcTensor("weights_sum", weights_sum);
  }
  op.AddSrcTensor("weights_scale", weights_scale);
  if (weights_zero_point) {
    op.AddSrcTensor("weights_zero_point", *weights_zero_point);
  }
  // Scale mul + weights_sum FMA (~5 FLOPs); zero_point adds 2 FMAs (+4 = 9).
  op.flops_ = dst.GetBHWDCShape().DimensionsProduct() *
              (weights_zero_point != nullptr ? 9 : 5);
  return op;
}

}  // namespace ml_drift
