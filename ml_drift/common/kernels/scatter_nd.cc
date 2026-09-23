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

#include "ml_drift/common/kernels/scatter_nd.h"

#include <string>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {

std::string GetScatterNdCode(const OperationDef& op_def) {
  const bool accumulate = IsFloatType(op_def.src_tensors[1].GetDataType());
  const std::string combine = accumulate
                                  ? "acc += args.updates.Read(iw, ih, S, ib);"
                                  : "acc = args.updates.Read(iw, ih, S, ib);";
  std::string c = R"(MAIN_FUNCTION($0) {
  int linear_xb = ucl::GetGlobalId<0>();
  int X = linear_xb / args.dst_tensor.Batch();
  int B = linear_xb % args.dst_tensor.Batch();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() ||
      S >= args.dst_tensor.Slices()) {
    return;
  }

  args.updates::type acc = args.updates::zero_value;

  for (int ib = 0; ib < args.indices.Batch(); ++ib) {
    for (int ih = 0; ih < args.indices.Height(); ++ih) {
      for (int iw = 0; iw < args.indices.Width(); ++iw) {
        int4 coord = args.indices.Read<int>(iw, ih, 0, ib);
        // Index depth 3: coordinates map to output (batch, height, width).
        if (coord.x == B && coord.y == Y && coord.z == X) {
          )";
  c += combine;
  c += R"(
        }
      }
    }
  }

  args.dst_tensor::type result;
  result.x = ucl::Convert<args.dst_tensor::scalar_type>(acc.x);
  result.y = ucl::Convert<args.dst_tensor::scalar_type>(acc.y);
  result.z = ucl::Convert<args.dst_tensor::scalar_type>(acc.z);
  result.w = ucl::Convert<args.dst_tensor::scalar_type>(acc.w);
  args.dst_tensor.Write(result, X, Y, S, B);
}
)";
  return c;
}

}  // namespace

GPUOperation CreateScatterNd(const OperationDef& op_def) {
  GPUOperation op;
  op.AddSrcTensor("indices", op_def.src_tensors[0]);
  op.AddSrcTensor("updates", op_def.src_tensors[1]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.code_ = GetScatterNdCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
