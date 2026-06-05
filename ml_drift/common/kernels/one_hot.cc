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

#include "ml_drift/common/kernels/one_hot.h"

#include <string>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

std::string GetOneHotCode(const OperationDef& op_def,
                          const OneHotAttributes& attr, GPUOperation* op) {
  op->AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  std::string c = R"(
MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int X = linear_id;
  int B = 0;
  if (args.dst_tensor.Batch() != 1) {
    X = linear_id / args.dst_tensor.Batch();
    B = linear_id % args.dst_tensor.Batch();
  }
  args.dst_tensor.SetBatchRef(B);
  args.src_tensor.SetBatchRef(B);
  int Y = ucl::GetGlobalId<1>();
  int Z = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) {
    return;
  }
  int idx = Z * 4;
  int hot_idx;
  args.src_tensor.ReadPerChannel(hot_idx, 0, Y, X);
  float4 res = ucl::Init<float4>(args.off_value);
  if ((hot_idx >= idx) && (hot_idx < (idx + 4))) {
    if (idx + 0 == hot_idx) { res.x = args.on_value;}
    if (idx + 1 == hot_idx) { res.y = args.on_value;}
    if (idx + 2 == hot_idx) { res.z = args.on_value;}
    if (idx + 3 == hot_idx) { res.w = args.on_value;}
  }
  args.dst_tensor::type res_final = ucl::Convert<args.dst_tensor::type>(res);
  args.dst_tensor.Write(res_final, X, Y, Z);
}
)";
  return c;
}

GPUOperation CreateOneHot(const OperationDef& definition,
                          const OneHotAttributes& attr) {
  GPUOperation op;
  op.code_ = GetOneHotCode(definition, attr, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.args_.AddFloat("on_value", attr.on_value);
  op.args_.AddFloat("off_value", attr.off_value);
  return op;
}

}  // namespace ml_drift
