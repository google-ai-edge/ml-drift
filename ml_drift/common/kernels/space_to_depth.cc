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

#include "ml_drift/common/kernels/space_to_depth.h"

#include <string>
#include <vector>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetSpaceToDepthCode(const OperationDef& op_def) {
  std::string c = "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += R"("
  int linear_id = ucl::GetGlobalId<0>();\n";
  int X = linear_id / args.dst_tensor.Batch();
  int B = linear_id % args.dst_tensor.Batch();
  args.dst_tensor.SetBatchRef(B);
  args.src_tensor.SetBatchRef(B);
)";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += R"(
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || S >= args.dst_tensor.Slices()) {
    return;
  }
  args.src_tensor::scalar_type tmp[4];
  tmp[0] = args.src_tensor::scalar_zero_value;
  tmp[1] = args.src_tensor::scalar_zero_value;
  tmp[2] = args.src_tensor::scalar_zero_value;
  tmp[3] = args.src_tensor::scalar_zero_value;
  for (int i = 0; i < 4; ++i) {
    int dst_c = 4 * S + i;
    if (dst_c >= args.dst_tensor.Channels()) {
      break;
    }
    int block_id = dst_c / args.src_tensor.Channels();
    int src_x = X * args.block_size + block_id % args.block_size;
    int src_y = Y * args.block_size + block_id / args.block_size;
    int src_c = dst_c % args.src_tensor.Channels();
    args.src_tensor.ReadPerChannel(tmp[i], src_x, src_y, src_c);
  }
  args.src_tensor::type result;
  result.x = tmp[0];
  result.y = tmp[1];
  result.z = tmp[2];
  result.w = tmp[3];
  args.dst_tensor.Write(result, X, Y, S);
}
)";
  return c;
}

std::string GetDepthToSpaceCode(const OperationDef& op_def) {
  std::string c = "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += R"("
  int linear_id = ucl::GetGlobalId<0>();\n";
  int X = linear_id / args.dst_tensor.Batch();
  int B = linear_id % args.dst_tensor.Batch();
  args.dst_tensor.SetBatchRef(B);
  args.src_tensor.SetBatchRef(B);
)";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += R"(
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || S >= args.dst_tensor.Slices()) {
    return;
  }
  int block_x = X % args.block_size;
  int src_x = X / args.block_size;
  int block_y = Y % args.block_size;
  int src_y = Y / args.block_size;
  int block_id = block_y * args.block_size + block_x;
  args.dst_tensor::type result;
  if (args.dst_tensor.Channels() % 4 == 0) {
    int src_s = block_id * args.dst_tensor.Slices() + S;
    src_s = min(src_s, args.src_tensor.Slices() - 1);
    result = args.src_tensor.Read(src_x, src_y, src_s);
  } else {
    args.dst_tensor::scalar_type tmp[4];
    tmp[0] = ucl::Init<args.dst_tensor::scalar_type>(0.0f);
    tmp[1] = ucl::Init<args.dst_tensor::scalar_type>(0.0f);
    tmp[2] = ucl::Init<args.dst_tensor::scalar_type>(0.0f);
    tmp[3] = ucl::Init<args.dst_tensor::scalar_type>(0.0f);
    for (int i = 0; i < 4; ++i) {
      int dst_c = 4 * S + i;
      if (dst_c >= args.dst_tensor.Channels()) {
        break;
      }
      int src_c = block_id * args.dst_tensor.Channels() + dst_c;
      src_c = min(src_c, args.src_tensor.Channels() - 1);
      args.src_tensor.ReadPerChannel(tmp[i], src_x, src_y, src_c);
    }
    result.x = tmp[0];
    result.y = tmp[1];
    result.z = tmp[2];
    result.w = tmp[3];
  }
  args.dst_tensor.Write(result, X, Y, S);
}
)";
  return c;
}
}  // namespace

GPUOperation CreateSpaceToDepth(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.args_.AddInt("block_size", attr.block_size);
  op.code_ = GetSpaceToDepthCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateDepthToSpace(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.args_.AddInt("block_size", attr.block_size);
  op.code_ = GetDepthToSpaceCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
