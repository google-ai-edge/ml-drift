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

#include "ml_drift/common/kernels/add.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
namespace {
GPUOperation CreateUnequalAdd(const OperationDef& op_def) {
  GPUOperation op;
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_tensor_", i);
    op.AddSrcTensor(tensor_name, op_def.src_tensors[i]);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    for (int i = 0; i < op_def.src_tensors.size(); ++i) {
      const std::string tensor_name = absl::StrCat("src_tensor_", i);
      c += "  args." + tensor_name + ".SetBatchRef(B);\n";
    }
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_y = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_y / args.dst_tensor.Depth();\n";
    c += "  int D = linear_y % args.dst_tensor.Depth();\n";
    coords += ", D";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) return; \n";
  coords += ", S";

  c += "  args.src_tensor_0::type src = args.src_tensor_0::zero_value;\n";
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_tensor_", i);
    c += "  if (S < args." + tensor_name + ".Slices()) {\n";
    c += "    src += args." + tensor_name + ".Read(" + coords + ");\n";
    c += "  }\n";
  }
  c += "  args.dst_tensor.Write(src, " + coords + ");\n";
  c += "} \n";
  op.code_ = std::move(c);
  return op;
}
}  // namespace

GPUOperation CreateAdd(const OperationDef& definition,
                       const std::vector<int>& channels, int dst_channels) {
  if (dst_channels != channels[0]) {
    return CreateUnequalAdd(definition);
  }
  ElementwiseDescriptor op_desc;
  op_desc.code = "  out_value = in_value;\n";
  for (int i = 1; i < definition.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_tensor_", i);
    std::string coords = "X_COORD, Y_COORD";
    if (definition.src_tensors[i].HasAxis(Axis::DEPTH)) {
      coords += ", Z_COORD";
    }
    coords += ", S_COORD";
    if (definition.src_tensors[i].HasAxis(Axis::BATCH)) {
      coords += ", B_COORD";
    }
    op_desc.code += "if (S_COORD < args." + tensor_name + ".Slices()) {\n";
    op_desc.code +=
        "  out_value += args." + tensor_name + ".Read(" + coords + ");\n";
    op_desc.code += "}\n";
  }
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace ml_drift
