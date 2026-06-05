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

#include "ml_drift/common/kernels/positional_embedding.h"

#include <string>
#include <utility>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

GPUOperation CreatePositionalEmbedding(const GpuInfo& gpu_info,
                                       const OperationDef& definition) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("position", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }
  std::string code = absl::Substitute(R"(MAIN_FUNCTION($$0) {
  int linear_xb = ucl::GetGlobalId<0>();
  int X = linear_xb / args.dst_tensor.Batch();
  int B = linear_xb % args.dst_tensor.Batch();
  args.src_tensor.SetBatchRef(B);
  args.dst_tensor.SetBatchRef(B);
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || S >= args.dst_tensor.Slices()) {
    return;
  }
  int sub_slice = S;
  if (S >= args.dst_tensor.Slices() / 2) {
    sub_slice = S - args.dst_tensor.Slices() / 2;
  }
  float4 indexes;
  indexes.x = ucl::Convert<float>(sub_slice * 4 + 0);
  indexes.y = ucl::Convert<float>(sub_slice * 4 + 1);
  indexes.z = ucl::Convert<float>(sub_slice * 4 + 2);
  indexes.w = ucl::Convert<float>(sub_slice * 4 + 3);

  float4 min_timescale = ucl::Init<float4>(1.0f);
  float4 max_timescale = ucl::Init<float4>(10000.0f);
  float4 num_timescales = ucl::Init<float4>(ucl::Convert<float>(args.dst_tensor.Channels() / 2));
  float4 log_timescale_increment = log(max_timescale / min_timescale) / max(num_timescales - ucl::Init<float4>(1.0f), ucl::Init<float4>(1.0f));
  float4 inv_timescale = min_timescale * exp(indexes * -log_timescale_increment);
  float4 pos_val = ucl::Init<float4>(ucl::Convert<float>(args.position.Read(X, 0, 0).x));
  float4 scaled_time = pos_val * inv_timescale;
  Type res_value;
  if (S < args.dst_tensor.Slices() / 2) {
    res_value = ucl::Convert<Type>($0(scaled_time));
  } else {
    res_value = ucl::Convert<Type>($1(scaled_time));
  }
  res_value += args.src_tensor.Read(X, Y, S);
  args.dst_tensor.Write(res_value, X, Y, S);
})",
                                      sin_func_name, cos_func_name);

  absl::StrReplaceAll(
      {{"Type", ToUclDataType(definition.dst_tensors[0].GetDataType(), 4)}},
      &code);
  op.code_ = code;
  return op;
}

}  // namespace ml_drift
