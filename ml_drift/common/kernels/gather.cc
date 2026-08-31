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

#include "ml_drift/common/kernels/gather.h"

#include <string>
#include <utility>
#include <vector>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetGatherCode(const OperationDef& op_def, Axis gather_axis) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int linear_id = ucl::GetGlobalId<0>();\n";
  c += "  int X = linear_id / args.dst_tensor.Batch();\n";
  c += "  int B = linear_id % args.dst_tensor.Batch();\n";
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  if (gather_axis == Axis::CHANNELS) {
    c += R"(
  args.src_tensor::scalar_type src_array[4];
  int4 gather_indexes = args.indices.Read<int>(0, 0, S);
  int gather_indexes_array[4];
  gather_indexes_array[0] = gather_indexes.x;
  gather_indexes_array[1] = gather_indexes.y;
  gather_indexes_array[2] = gather_indexes.z;
  gather_indexes_array[3] = gather_indexes.w;
  for (int i = 0; i < 4; ++i) {
    src_array[i] = args.src_tensor::scalar_zero_value;
    int ch_index = S * 4 + i;
    if (ch_index < args.dst_tensor.Channels()) {
      args.src_tensor.ReadPerChannel(src_array[i], X, Y, gather_indexes_array[i], B);
    }
  }
  args.dst_tensor::type result;
  result.x = ucl::Convert<args.dst_tensor::scalar_type>(src_array[0]);
  result.y = ucl::Convert<args.dst_tensor::scalar_type>(src_array[1]);
  result.z = ucl::Convert<args.dst_tensor::scalar_type>(src_array[2]);
  result.w = ucl::Convert<args.dst_tensor::scalar_type>(src_array[3]);
)";
  } else {
    const std::vector<std::pair<Axis, std::string>> coords = {
        {Axis::WIDTH, "X"},
        {Axis::HEIGHT, "Y"},
        {Axis::CHANNELS, "S"},
        {Axis::BATCH, "B"},
    };
    std::string src_coords;
    std::string gather_coord;
    for (const auto& [axis, name] : coords) {
      if (!src_coords.empty()) {
        src_coords += ", ";
      }
      if (axis == gather_axis) {
        gather_coord = name;
        src_coords += "gather_index";
      } else {
        src_coords += name;
      }
    }
    c += "  int gather_index;\n";
    c += "  args.indices.ReadPerChannel<int>(gather_index, 0, 0, " +
         gather_coord + ");\n";
    // Key the result off the destination, not the source: a folded fp16
    // weight can feed an fp32 destination, and WGSL has no implicit numeric
    // conversions.
    c += "  args.dst_tensor::type result = "
         "ucl::Convert<args.dst_tensor::type>(args.src_tensor.Read(" +
         src_coords + "));\n";
  }
  c += "  args.dst_tensor.Write(result, X, Y, S, B);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateGather(const OperationDef& op_def,
                          const GatherAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddSrcTensor("indices", op_def.src_tensors[1]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.code_ = GetGatherCode(op_def, attr.axis);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
