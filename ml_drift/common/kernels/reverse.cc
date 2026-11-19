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

#include "ml_drift/common/kernels/reverse.h"

#include <string>
#include <vector>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetReverseCode(const OperationDef& op_def,
                           const ReverseAttributes& attr) {
  std::string write_coords;
  if (attr.axes.find(Axis::kWidth) != attr.axes.end()) {
    write_coords += "args.dst_tensor.Width() - X - 1, ";
  } else {
    write_coords += "X, ";
  }
  if (attr.axes.find(Axis::kHeight) != attr.axes.end()) {
    write_coords += "args.dst_tensor.Height() - Y - 1, ";
  } else {
    write_coords += "Y, ";
  }
  if (attr.axes.find(Axis::kChannels) != attr.axes.end()) {
    write_coords += "args.dst_tensor.Slices() - S - 1";
  } else {
    write_coords += "S";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    write_coords += ", ";
    if (attr.axes.find(Axis::kBatch) != attr.axes.end()) {
      write_coords += "args.dst_tensor.Batch() - B - 1";
    } else {
      write_coords += "B";
    }
  }
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::type result = args.src_tensor.Read(X, Y, S);\n";
  if (attr.axes.find(Axis::kChannels) != attr.axes.end()) {
    c += "  if ((S + 1) * 4 > args.dst_tensor.Channels()) {\n";
    c += "    int extra = args.dst_tensor.Channels() % 4;\n";
    c += "    if (extra == 2) {\n";
    c += "      result = result.yxzw;\n";
    c += "    } else if (extra == 3) {\n";
    c += "      result = result.zyxw;\n";
    c += "    }\n";
    c += "  } else {\n";
    c += "    result = result.wzyx;\n";
    c += "  }\n";
  }
  c += "  args.dst_tensor.Write(result, " + write_coords + ");\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateReverse(const OperationDef& op_def,
                           const ReverseAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.code_ = GetReverseCode(op_def, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
