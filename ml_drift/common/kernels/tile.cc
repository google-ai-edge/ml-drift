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

#include "ml_drift/common/kernels/tile.h"

#include <string>

#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

namespace {
std::string GetTileCode(const TensorDescriptor& src_desc,
                        const TensorDescriptor& dst_desc,
                        bool src_channels_x4) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (dst_desc.HasAxis(Axis::BATCH)) {
    c += "  int linear_id_0 = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id_0 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_0 % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  if (dst_desc.HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  std::string dst_coords = "X, Y";
  if (dst_desc.HasAxis(Axis::DEPTH)) {
    dst_coords += ", Z";
  }
  dst_coords += ", S";
  if (dst_desc.HasAxis(Axis::BATCH)) {
    dst_coords += ", B";
  }
  std::string src_coords = "src_x, src_y";
  if (src_desc.HasAxis(Axis::DEPTH)) {
    src_coords += ", src_z";
  }
  src_coords += ", src_s";
  if (src_desc.HasAxis(Axis::BATCH)) {
    src_coords += ", src_b";
  }
  c += "  int src_x = X % args.src_tensor.Width();\n";
  c += "  int src_y = Y % args.src_tensor.Height();\n";
  if (src_desc.HasAxis(Axis::DEPTH)) {
    c += "  int src_z = Z % args.src_tensor.Depth();\n";
  }
  if (src_desc.HasAxis(Axis::BATCH)) {
    c += "  int src_b = B % args.src_tensor.Batch();\n";
  }
  if (src_channels_x4) {
    c += "  int src_s = S % args.src_tensor.Slices();\n";
    c += "  args.src_tensor::type result = args.src_tensor.Read(" + src_coords +
         ");\n";
  } else {
    c += "  args.src_tensor::scalar_type tmp[4];\n";
    c += "  tmp[0] = args.src_tensor::scalar_zero_value;\n";
    c += "  tmp[1] = args.src_tensor::scalar_zero_value;\n";
    c += "  tmp[2] = args.src_tensor::scalar_zero_value;\n";
    c += "  tmp[3] = args.src_tensor::scalar_zero_value;\n";
    c += "  for (int i = 0; i < 4; ++i) {\n";
    c += "    int dst_c = 4 * S + i;\n";
    c += "    int src_s = dst_c % args.src_tensor.Channels();\n";
    c += "    args.src_tensor.ReadPerChannel(tmp[i], " + src_coords + ");\n";
    c += "  }\n";
    c += "  args.src_tensor::type result;\n";
    c += "  result.x = tmp[0];\n";
    c += "  result.y = tmp[1];\n";
    c += "  result.z = tmp[2];\n";
    c += "  result.w = tmp[3];\n";
  }
  c += "  args.dst_tensor.Write(result, " + dst_coords + ");\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateTile(const TensorDescriptor& src_desc,
                        const TensorDescriptor& dst_desc,
                        bool is_src_channels_x4) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", src_desc);
  op.AddDstTensor("dst_tensor", dst_desc);
  op.code_ = GetTileCode(src_desc, dst_desc, is_src_channels_x4);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
