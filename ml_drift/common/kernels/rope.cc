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

#include "ml_drift/common/kernels/rope.h"

#include <string>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
namespace ml_drift {
GPUOperation CreateRoPE(const GpuInfo& gpu_info, const OperationDef& definition,
                        const RoPEAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_l", definition.src_tensors[0]);
  op.AddSrcTensor("src_r", definition.src_tensors[1]);
  op.AddSrcTensor("position", definition.src_tensors[2]);
  op.AddDstTensor("dst_l", definition.dst_tensors[0]);
  op.AddDstTensor("dst_r", definition.dst_tensors[1]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.args_.AddFloat("min_timescale", attr.min_timescale);
  op.args_.AddFloat("max_timescale", attr.max_timescale);
  op.args_.AddFloat("proportion", attr.proportion);

  std::string pow_func_name = "pow";
  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    pow_func_name = "native_powr";
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }
  std::string code;
  code += "MAIN_FUNCTION($0) {\n";
  if (definition.dst_tensors[0].HasAxis(Axis::BATCH)) {
    code += "  int linear_id = ucl::GetGlobalId<0>();\n";
    code += "  int X = linear_id / args.dst_l.Batch();\n";
    code += "  int B = linear_id % args.dst_l.Batch();\n";
    code += "  args.src_l.SetBatchRef(B);\n";
    code += "  args.src_r.SetBatchRef(B);\n";
    code += "  args.position.SetBatchRef(B);\n";
    code += "  args.dst_l.SetBatchRef(B);\n";
    code += "  args.dst_r.SetBatchRef(B);\n";
  } else {
    code += "  int X = ucl::GetGlobalId<0>();\n";
  }
  code += absl::Substitute(R"(
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_l.Width() || Y >= args.dst_l.Height() || S >= args.dst_l.Slices()) {
    return;
  }
  float inv_dst_ch = 1.0f / ucl::Convert<float>(args.dst_l.Channels());
  float4 fraction = (ucl::Init<float4>(0.0f, 1.0f, 2.0f, 3.0f) + ucl::Convert<float>(S * 4)) * inv_dst_ch;
  float4 min_timescale = ucl::Init<float4>(args.min_timescale);
  float4 max_timescale = ucl::Init<float4>(args.max_timescale);
  float4 timescale = min_timescale * $0(max_timescale / min_timescale, fraction);
  float pos_scalar;
  if (args.position.Channels() > 1) {
    args.position.ReadPerChannel<float>(pos_scalar, 0, 0, X % args.position.Channels());
  } else {
    pos_scalar = args.position.Read<float>(X % args.position.Width(), 0, 0).x;
  }
  float4 pos_val = ucl::Init<float4>(pos_scalar);
  float4 sinusoid_inp = pos_val / timescale;
  Type sin_val = ucl::Convert<Type>($1(sinusoid_inp));
  Type cos_val = ucl::Convert<Type>($2(sinusoid_inp));
  Type val0 = args.src_l.Read(X, Y, S);
  Type val1 = args.src_r.Read(X, Y, S);
  Type out0;
  Type out1;
  if (fraction.w < ucl::Init<float>(args.proportion)) {
    out0 = val0 * cos_val - val1 * sin_val;
    out1 = val1 * cos_val + val0 * sin_val;
  } else {
    out0 = val0;
    out1 = val1;
  }
  args.dst_l.Write(out0, X, Y, S);
  args.dst_r.Write(out1, X, Y, S);
})",
                           pow_func_name, sin_func_name, cos_func_name);
  absl::StrReplaceAll(
      {{"Type", ToUclDataType(definition.dst_tensors[0].GetDataType(), 4)}},
      &code);
  op.code_ = code;
  return op;
}

GPUOperation CreateSplitRoPEConcat(const GpuInfo& gpu_info,
                                   const OperationDef& definition,
                                   const RoPEAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("position", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.args_.AddFloat("min_timescale", attr.min_timescale);
  op.args_.AddFloat("max_timescale", attr.max_timescale);
  op.args_.AddFloat("proportion", attr.proportion);
  op.args_.AddInt("kernel_type", static_cast<int>(attr.kernel_type));

  std::string pow_func_name = "pow";
  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    pow_func_name = "native_powr";
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }
  std::string code;
  code += "MAIN_FUNCTION($0) {\n";
  if (definition.dst_tensors[0].HasAxis(Axis::BATCH)) {
    code += "  int linear_id = ucl::GetGlobalId<0>();\n";
    code += "  int X = linear_id / args.dst_tensor.Batch();\n";
    code += "  int B = linear_id % args.dst_tensor.Batch();\n";
    code += "  args.src_tensor.SetBatchRef(B);\n";
    code += "  args.position.SetBatchRef(B);\n";
    code += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    code += "  int X = ucl::GetGlobalId<0>();\n";
  }
  // kernel_type 0=PLANAR_1D, 1=INTERLEAVED_2D.
  code += absl::Substitute(
      R"(
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  int slice_count = args.dst_tensor.Slices();
  int half_slices_count = slice_count / 2;
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) {
    return;
  }
  if (args.kernel_type == 0 && S >= half_slices_count) {
    return;
  }
  if (args.kernel_type == 1 && (S % 2 == 1 || S >= slice_count)) {
    return;
  }
  float inv_dst_ch = 1.0f / ucl::Convert<float>(args.dst_tensor.Channels());
  float pos_scalar;
  if (args.position.Channels() > 1) {
    args.position.ReadPerChannel<float>(pos_scalar, 0, 0, X % args.position.Channels());
  } else {
    pos_scalar = args.position.Read<float>(X % args.position.Width(), 0, 0).x;
  }
  float4 pos_val = ucl::Init<float4>(pos_scalar);
  int s_mult = args.kernel_type == 1 ? 2 : 4;
  int4 p = S * s_mult + ucl::Init<int4>(0, 1, 2, 3);
  if (args.kernel_type == 1) {)"
      // INTERLEAVED_2D: Y % Width() is correct. args.position is 1D and it is
      // assumed that args.src_tensor.Width() == args.src_tensor.Height().
      // Note: channels is a multiple of 8, so slice_count is even and p.xy and
      // p.zw are each guaranteed to be in the same interval [0, slice_count) or
      // [slice_count, 2*slice_count). Hence, only 2 ifs instead of 4.
      R"(
    float pos_y_scalar;
    if (args.position.Channels() > 1) {
      args.position.ReadPerChannel<float>(pos_y_scalar, 0, 0, Y % args.position.Channels());
    } else {
      pos_y_scalar = args.position.Read<float>(Y % args.position.Width(), 0, 0).x;
    }
    float4 pos_y = ucl::Init<float4>(pos_y_scalar);
    if (p.x >= slice_count) { p.x -= slice_count; p.y -= slice_count; pos_val.x = pos_y.x; pos_val.y = pos_y.y; }
    if (p.z >= slice_count) { p.z -= slice_count; p.w -= slice_count; pos_val.z = pos_y.z; pos_val.w = pos_y.w; }
  }
  float fraction_mult = args.kernel_type == 1 ? 4.0f : 2.0f;
  float4 fraction;
  fraction.x = fraction_mult * ucl::Convert<float>(p.x) * inv_dst_ch;
  fraction.y = fraction_mult * ucl::Convert<float>(p.y) * inv_dst_ch;
  fraction.z = fraction_mult * ucl::Convert<float>(p.z) * inv_dst_ch;
  fraction.w = fraction_mult * ucl::Convert<float>(p.w) * inv_dst_ch;

  float4 min_timescale = ucl::Init<float4>(args.min_timescale);
  float4 max_timescale = ucl::Init<float4>(args.max_timescale);
  float4 timescale = min_timescale * $0(max_timescale / min_timescale, fraction);
  float4 sinusoid_inp = pos_val / timescale;
  Type sin_val = ucl::Convert<Type>($1(sinusoid_inp));
  Type cos_val = ucl::Convert<Type>($2(sinusoid_inp));

  int slice0 = S;
  int slice1 = S + half_slices_count;
  if (args.kernel_type == 1) {
    slice0 = S;
    slice1 = S + 1;
  }
  Type val0 = args.src_tensor.Read(X, Y, slice0);
  Type val1 = args.src_tensor.Read(X, Y, slice1);
  if (args.kernel_type == 1) {
    Type t0 = ucl::Init<Type>(val0.x, val0.z, val1.x, val1.z);
    Type t1 = ucl::Init<Type>(val0.y, val0.w, val1.y, val1.w);
    val0 = t0;
    val1 = t1;
  }
  Type out0;
  Type out1;
  if (fraction.w < ucl::Init<float>(args.proportion)) {
    out0 = val0 * cos_val - val1 * sin_val;
    out1 = val1 * cos_val + val0 * sin_val;
  } else {
    out0 = val0;
    out1 = val1;
  }
  if (args.kernel_type == 1) {
    Type t0 = ucl::Init<Type>(out0.x, out1.x, out0.y, out1.y);
    Type t1 = ucl::Init<Type>(out0.z, out1.z, out0.w, out1.w);
    out0 = t0;
    out1 = t1;
  }
  args.dst_tensor.Write(out0, X, Y, slice0);
  args.dst_tensor.Write(out1, X, Y, slice1);
})",
      pow_func_name, sin_func_name, cos_func_name);
  absl::StrReplaceAll(
      {{"Type", ToUclDataType(definition.dst_tensors[0].GetDataType(), 4)}},
      &code);
  op.code_ = code;
  op.AllowFuseInputReorder(true);
  return op;
}
}  // namespace ml_drift
