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

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "ml_drift/common/op_api.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

std::string GetTembGenerationCode() {
  std::string c = R"(MAIN_FUNCTION($0) {
    int X = ucl::GetGlobalId<0>();
    int Y = ucl::GetGlobalId<1>();
    int S = ucl::GetGlobalId<2>();
    if (X >= args.dst.Width() || Y >= args.dst.Height() ||
        S >= args.dst.Slices())
      return;
  float index_val = args.index_val.Read(0);
  if (S < args.half_slices) {
    args.dst::type result = ucl::Convert<args.dst::type>(
        cos(args.coeffs.Read(S) * index_val));
    args.dst.Write(result, X, Y, S);
  } else {
    args.dst::type result = ucl::Convert<args.dst::type>(
        sin(args.coeffs.Read(S - args.half_slices) * index_val));
    args.dst.Write(result, X, Y, S);
  }
})";
  return c;
}

class TembGeneration : public OpBase<TembGeneration> {
 public:
  // clang-format off
  MLD_OP_ATTRS(
      MLD_OP_ATTR_NO_BIND(int, channels, 320);
  )
  // clang-format on

  absl::Status Build(
      GpuModelBuilder& graph,
      const std::vector<GpuModelBuilder::TensorHandle>& inputs, Attrs& attrs,
      std::vector<GpuModelBuilder::TensorHandle>& outputs) const {
    if (inputs.size() != 1) {
      return absl::InvalidArgumentError("TembGeneration expects 1 input");
    }

    int half_channels = attrs.channels / 2;
    int half_slices = half_channels / 4;

    outputs = {graph.AddTensor(1, 1, 1, attrs.channels, DataType::kFloat16)};

    auto* op = AppendNewOp(graph, attrs);
    if (!op) return absl::InternalError("Failed to append new op");

    Tensor<Linear, DataType::kFloat32> coeffs_tensor;
    coeffs_tensor.shape = Linear(half_channels);
    coeffs_tensor.data.resize(coeffs_tensor.shape.DimensionsProduct());
    for (int i = 0; i < coeffs_tensor.shape.v; ++i) {
      float value = std::exp(-std::log(10000.0) *
                             (i / static_cast<float>(half_channels)));
      coeffs_tensor.data[i] = value;
    }
    TensorDescriptor coeffs_tensor_desc = CreateConstantLinearTensorDescriptor(
        graph.gpu_info(), DataType::kFloat32, coeffs_tensor);
    op->args_.AddObject("coeffs", std::make_unique<TensorDescriptor>(
                                      std::move(coeffs_tensor_desc)));
    op->args_.AddInt("half_slices", half_slices);

    AddSrcBuffer(graph, op, "index_val", inputs[0]);
    AddDstTensor(graph, op, "dst", outputs[0]);

    const auto& dst_shape = outputs[0].tensor_desc.GetBHWCShape();
    op->SetGridSize(
        int3(dst_shape.w * dst_shape.b, dst_shape.h, (dst_shape.c + 3) / 4));
    op->code_ = GetTembGenerationCode();
    return absl::OkStatus();
  }
};

MLD_REGISTER_OP(TembGeneration);

}  // namespace ml_drift
