// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/select_v2.h"

#include <string>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

std::string GetSelectV2Code(const OperationDef& op_def) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += R"(
  int linear_id = ucl::GetGlobalId<0>();
  int X = linear_id / args.dst_tensor.Batch();
  int B = linear_id % args.dst_tensor.Batch();
  args.cond_tensor.SetBatchRef(B);
  args.dst_tensor.SetBatchRef(B);
  int true_b = args.true_tensor.Batch() == 1 ? 0 : B;
  args.true_tensor.SetBatchRef(true_b);
  int false_b = args.false_tensor.Batch() == 1 ? 0 : B;
  args.false_tensor.SetBatchRef(false_b);
  int cond_b = args.cond_tensor.Batch() == 1 ? 0 : B;
  args.cond_tensor.SetBatchRef(cond_b);
)";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += R"(
  int Y = ucl::GetGlobalId<1>();
  int Z = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) {
    return;
  }

  // Read with broadcast true_val.
  int true_x = args.true_tensor.Width() == 1 ? 0 : X;
  int true_y = args.true_tensor.Height() == 1 ? 0 : Y;
  int true_z = args.true_tensor.Channels() == 1 ? 0 : Z;
  args.dst_tensor::type true_val = args.true_tensor.Read(true_x, true_y, true_z);
  if (args.true_tensor.Channels() == 1) {
    true_val.y = true_val.x;
    true_val.z = true_val.x;
    true_val.w = true_val.x;
  }

  // Read with broadcast false_val.
  int false_x = args.false_tensor.Width() == 1 ? 0 : X;
  int false_y = args.false_tensor.Height() == 1 ? 0 : Y;
  int false_z = args.false_tensor.Channels() == 1 ? 0 : Z;
  args.dst_tensor::type false_val = args.false_tensor.Read(false_x, false_y, false_z);
  if (args.false_tensor.Channels() == 1) {
    false_val.y = false_val.x;
    false_val.z = false_val.x;
    false_val.w = false_val.x;
  }

  // Read with broadcast cond_val.
  int cond_x = args.cond_tensor.Width() == 1 ? 0 : X;
  int cond_y = args.cond_tensor.Height() == 1 ? 0 : Y;
  int cond_z = args.cond_tensor.Channels() == 1 ? 0 : Z;
  bool4 cond_val = args.cond_tensor.Read<bool>(cond_x, cond_y, cond_z);
  if (args.cond_tensor.Channels() == 1) {
    cond_val.y = cond_val.x;
    cond_val.z = cond_val.x;
    cond_val.w = cond_val.x;
  }

  args.dst_tensor::type res;
  res.x = cond_val.x ? true_val.x : false_val.x;
  res.y = cond_val.y ? true_val.y : false_val.y;
  res.z = cond_val.z ? true_val.z : false_val.z;
  res.w = cond_val.w ? true_val.w : false_val.w;
  args.dst_tensor.Write(res, X, Y, Z);
}
)";
  return c;
}

GPUOperation CreateSelectV2(const OperationDef& definition,
                            const SelectV2Attributes& attr) {
  GPUOperation op;
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.AddSrcTensor("cond_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("true_tensor", definition.src_tensors[1]);
  op.AddSrcTensor("false_tensor", definition.src_tensors[2]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetSelectV2Code(definition);
  return op;
}

}  // namespace ml_drift
