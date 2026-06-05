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
  int pos_x = X % args.position.Width();
  float4 pos_val = ucl::Init<float4>(args.position.Read<float>(pos_x, 0, 0).x);
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
  code += absl::Substitute(R"(
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  int half_slices_count = args.dst_tensor.Slices() / 2;
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || S >= half_slices_count) {
    return;
  }
  int slice0 = S;
  int slice1 = S + half_slices_count;
  float4 fraction;
  float inv_dst_ch = 1.0f / ucl::Convert<float>(args.dst_tensor.Channels());
  fraction.x = 2.0f * ucl::Convert<float>(S * 4 + 0) * inv_dst_ch;
  fraction.y = 2.0f * ucl::Convert<float>(S * 4 + 1) * inv_dst_ch;
  fraction.z = 2.0f * ucl::Convert<float>(S * 4 + 2) * inv_dst_ch;
  fraction.w = 2.0f * ucl::Convert<float>(S * 4 + 3) * inv_dst_ch;
  float4 min_timescale = ucl::Init<float4>(args.min_timescale);
  float4 max_timescale = ucl::Init<float4>(args.max_timescale);
  float4 timescale = min_timescale * $0(max_timescale / min_timescale, fraction);
  int pos_x = X % args.position.Width();
  float4 pos_val = ucl::Init<float4>(args.position.Read<float>(pos_x, 0, 0).x);
  float4 sinusoid_inp = pos_val / timescale;
  Type sin_val = ucl::Convert<Type>($1(sinusoid_inp));
  Type cos_val = ucl::Convert<Type>($2(sinusoid_inp));
  Type val0 = args.src_tensor.Read(X, Y, slice0);
  Type val1 = args.src_tensor.Read(X, Y, slice1);
  Type out0;
  Type out1;
  if (fraction.w < ucl::Init<float>(args.proportion)) {
    out0 = val0 * cos_val - val1 * sin_val;
    out1 = val1 * cos_val + val0 * sin_val;
  } else {
    out0 = val0;
    out1 = val1;
  }
  args.dst_tensor.Write(out0, X, Y, slice0);
  args.dst_tensor.Write(out1, X, Y, slice1);
})",
                           pow_func_name, sin_func_name, cos_func_name);
  absl::StrReplaceAll(
      {{"Type", ToUclDataType(definition.dst_tensors[0].GetDataType(), 4)}},
      &code);
  op.code_ = code;
  return op;
}
}  // namespace ml_drift
