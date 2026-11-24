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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "ml_drift/common/op_api.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor_handle.h"

namespace ml_drift {

class FillPositionIds : public OpBase<FillPositionIds> {
 public:
  // clang-format off
  MLD_OP_ATTRS(
      MLD_OP_ATTR(int, param_index, 0);
      MLD_OP_ATTR_NO_BIND(int, batch_size, 1);
      MLD_OP_ATTR_NO_BIND(int, sequence_size, 1);
  )
  // clang-format on

  absl::Status Build(GpuModelBuilder& graph,
                     const std::vector<TensorHandle>& inputs, Attrs& attrs,
                     std::vector<TensorHandle>& outputs) const {
    if (inputs.size() != 1) {
      return absl::InvalidArgumentError("FillPositionIds expects 1 input");
    }
    if (attrs.batch_size <= 0 || attrs.sequence_size <= 0) {
      return absl::InvalidArgumentError(
          "FillPositionIds requires batch_size > 0 and sequence_size > 0");
    }
    if (attrs.param_index < 0 ||
        attrs.param_index >=
            inputs[0].tensor_desc.GetBHWCShape().DimensionsProduct()) {
      return absl::InvalidArgumentError(
          "FillPositionIds param_index is out of range for src");
    }

    outputs = {graph.AddTensor(attrs.batch_size, 1, attrs.sequence_size, 1,
                               DataType::kInt32)};

    auto* op = AppendNewOp(graph, attrs);

    AddSrcBuffer(graph, op, "src", inputs[0]);
    AddDstTensor(graph, op, "dst", outputs[0]);

    const auto& dst_shape = outputs[0].tensor_desc.GetBHWCShape();
    op->SetGridSize(
        int3(dst_shape.w * dst_shape.b, dst_shape.h, (dst_shape.c + 3) / 4));
    op->flops_ = dst_shape.DimensionsProduct();

    op->code_ = R"(MAIN_FUNCTION($0) {
      int linear_xb = ucl::GetGlobalId<0>();
      int X = linear_xb / args.dst.Batch();
      int B = linear_xb % args.dst.Batch();
      args.dst.SetBatchRef(B);
      int Y = ucl::GetGlobalId<1>();
      int S = ucl::GetGlobalId<2>();
      if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
      args.dst::type res_value;
      int offset = args.src.Read(args.param_index);
      res_value.x = ucl::Convert<args.dst::scalar_type>(offset + X);
      res_value.y = ucl::Convert<args.dst::scalar_type>(offset + X);
      res_value.z = ucl::Convert<args.dst::scalar_type>(offset + X);
      res_value.w = ucl::Convert<args.dst::scalar_type>(offset + X);
      args.dst.Write(res_value, X, Y, S);
    })";
    return absl::OkStatus();
  }
};

MLD_REGISTER_OP(FillPositionIds);

}  // namespace ml_drift
