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

#include "ml_drift/common/kernels/reshape.h"

#include <string>

#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetReshapeCode(const OperationDef& op_def) {
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
  c += "  int Z = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::scalar_type temps[4];\n";
  c += "  temps[0] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[1] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[2] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[3] = args.src_tensor::scalar_zero_value;\n";
  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  c += "  int base = " + batch_id + ";\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  base = (((base * args.dst_tensor.Height() + Y) * "
         "args.dst_tensor.Width() + X) * args.dst_tensor.Depth() + D) * "
         "args.dst_tensor.Channels() + Z * 4;\n";
  } else {
    c += "  base = ((base * args.dst_tensor.Height() + Y) * "
         "args.dst_tensor.Width() + X) * args.dst_tensor.Channels() + Z * 4;\n";
  }
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_channel = Z * 4 + i;\n";
  c += "    if (dst_channel < args.dst_tensor.Channels()) {\n";
  c += "      int p = base + i;\n";
  c += "      int src_c = p % args.src_tensor.Channels();\n";
  c += "      p = p / args.src_tensor.Channels();\n";
  std::string coords = "src_x, src_y, src_c";
  if (op_def.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "      int src_d = p % args.src_tensor.Depth();\n";
    c += "      p = p / args.src_tensor.Depth();\n";
    coords = "src_x, src_y, src_d, src_c";
  }
  c += "      int src_x = p % args.src_tensor.Width();\n";
  c += "      p = p / args.src_tensor.Width();\n";
  c += "      int src_y = p % args.src_tensor.Height();\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "      int src_b = p / args.src_tensor.Height();\n";
    coords += ", src_b";
  }
  c += "      args.src_tensor.ReadPerChannel(temps[i], " + coords + ");\n";
  c += "    }\n";
  c += "  }\n";
  c += "  args.src_tensor::type result;\n";
  c += "  result.x = temps[0];\n";
  c += "  result.y = temps[1];\n";
  c += "  result.z = temps[2];\n";
  c += "  result.w = temps[3];\n";
  std::string dst_coords =
      op_def.dst_tensors[0].HasAxis(Axis::DEPTH) ? "X, Y, D, Z" : "X, Y, Z";
  c += "  args.dst_tensor.Write(result, " + dst_coords + ");\n";
  c += "}\n";
  return c;
}

}  // namespace

GPUOperation CreateReshape(const OperationDef& definition) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetReshapeCode(definition);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
