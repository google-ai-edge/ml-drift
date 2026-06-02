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

#include "ml_drift/common/kernels/transpose.h"

#include <string>
#include <utility>
#include <vector>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetTransposeCode(const OperationDef& op_def,
                             const std::vector<int>& remap, bool is_5d) {
  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  const std::string depth_id =
      op_def.dst_tensors[0].HasAxis(Axis::DEPTH) ? "D" : "0";
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id_0 = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id_0 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_0 % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::scalar_type temps[4];\n";
  c += "  temps[0] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[1] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[2] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[3] = args.src_tensor::scalar_zero_value;\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_channel = S * 4 + i;\n";
  c += "    if (dst_channel < args.dst_tensor.Channels()) {\n";
  const std::string bhwdc[] = {batch_id, "Y", "X", depth_id, "dst_channel"};
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "      args.src_tensor.SetBatchRef(" + bhwdc[remap[0]] + ");\n";
  }
  c += "      int s_y = " + bhwdc[remap[1]] + ";\n";
  c += "      int s_x = " + bhwdc[remap[2]] + ";\n";
  std::string read_coords = "s_x, s_y";
  if (op_def.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "      int s_d = " + bhwdc[remap[3]] + ";\n";
    read_coords += ", s_d";
  }
  const int src_c_index = is_5d ? 4 : 3;
  c += "      int s_c = " + bhwdc[remap[src_c_index]] + ";\n";
  read_coords += ", s_c";
  c += "      args.src_tensor.ReadPerChannel(temps[i], " + read_coords + ");\n";
  c += "    }\n";
  c += "  }\n";
  c += "  args.src_tensor::type result;\n";
  c += "  result.x = temps[0];\n";
  c += "  result.y = temps[1];\n";
  c += "  result.z = temps[2];\n";
  c += "  result.w = temps[3];\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  args.dst_tensor.Write(result, X, Y, D, S);\n";
  } else {
    c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  }
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateTranspose(const OperationDef& definition,
                             const TransposeAttributes& attr) {
  if (attr.perm.c == 3) {
    // optimized version when no channels permutation
    int remap[4];
    remap[attr.perm.b] = 0;
    remap[attr.perm.h] = 1;
    remap[attr.perm.w] = 2;
    remap[attr.perm.c] = 3;
    const std::string batch_id =
        definition.dst_tensors[0].HasAxis(Axis::BATCH) ? "DST_B" : "0";
    const std::string bhw[] = {batch_id, "DST_Y", "DST_X"};
    std::string code = "  SRC_S = DST_S;\n";
    code += "  SRC_X = " + bhw[remap[2]] + ";\n";
    code += "  SRC_Y = " + bhw[remap[1]] + ";\n";
    if (definition.src_tensors[0].HasAxis(Axis::BATCH)) {
      code += "  SRC_B = " + bhw[remap[0]] + ";\n";
    }
    return CreateReorderGpuOperation(definition, std::move(code));
  }
  GPUOperation op;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  std::vector<int> remap(5);
  remap[attr.perm.b] = 0;
  remap[attr.perm.h] = 1;
  remap[attr.perm.w] = 2;
  remap[attr.perm.c] = 4;
  op.code_ = GetTransposeCode(definition, remap, false);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateTranspose(const OperationDef& definition,
                             const Transpose3DAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  std::vector<int> remap(5);
  remap[attr.perm.b] = 0;
  remap[attr.perm.h] = 1;
  remap[attr.perm.w] = 2;
  remap[attr.perm.d] = 3;
  remap[attr.perm.c] = 4;
  op.code_ = GetTransposeCode(definition, remap, true);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
