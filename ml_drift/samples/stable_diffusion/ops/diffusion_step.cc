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
#include "absl/strings/str_replace.h"
#include "ml_drift/common/op_api.h"

namespace ml_drift {

std::string GetDiffusionStepCode(DataType dst_type) {
  std::string c = R"(MAIN_FUNCTION($0) {
    int X = ucl::GetGlobalId<0>();
    int Y = ucl::GetGlobalId<1>();
    int S = ucl::GetGlobalId<2>();
    if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
    Type eta_cond = args.etaCondIn.Read(X, Y, S);
    Type eta_uncond = args.etaUncondIn.Read(X, Y, S);
    Type x_in = args.xIn.Read(X, Y, S);

    Type delta_cond = (eta_cond - eta_uncond) * args.guidance_scale;
    Type eta = eta_uncond + delta_cond;

    Type deltaX0 = eta * args.sqrt_one_minus_alpha;
    Type predX0Unscaled = x_in - deltaX0;
    Type predX0 = predX0Unscaled / args.sqrt_alpha;
    Type dirX = eta * args.sqrt_one_minus_alpha_prev;
    Type xPrevBase = predX0 * args.sqrt_alpha_prev;
    Type result = xPrevBase + dirX;
    args.dst.Write(result, X, Y, S);
})";
  absl::StrReplaceAll({{"Type", ToUclDataType(dst_type, 4)}}, &c);
  absl::StrReplaceAll({{"Type", ToUclDataType(dst_type, 4)}}, &c);
  return c;
}

class DiffusionStep : public OpBase {
 public:
  // clang-format off
  MLD_DECLARE_OP_ATTRS(
      MLD_OP_ATTR(half, guidance_scale, half(1.0f)),
      MLD_OP_ATTR(half, sqrt_alpha, half(1.0f)),
      MLD_OP_ATTR(half, sqrt_alpha_prev, half(1.0f)),
      MLD_OP_ATTR(half, sqrt_one_minus_alpha, half(1.0f)),
      MLD_OP_ATTR(half, sqrt_one_minus_alpha_prev, half(1.0f))
  )
  // clang-format on

  absl::Status Build(
      GpuModelBuilder& graph,
      const std::vector<GpuModelBuilder::TensorHandle>& inputs,
      const OpAttrs& attrs,
      std::vector<GpuModelBuilder::TensorHandle>& outputs) const override {
    if (inputs.size() != 3) {
      return absl::InvalidArgumentError("DiffusionStep expects 3 inputs");
    }

    // Output shape matches input shape
    outputs = graph.AddTensors({inputs[0].tensor_desc});

    auto* op = AppendNewOp(graph, attrs);
    if (!op) return absl::InternalError("Failed to append new op");

    AddSrcTensor(graph, op, "xIn", inputs[0]);
    AddSrcTensor(graph, op, "etaUncondIn", inputs[1]);
    AddSrcTensor(graph, op, "etaCondIn", inputs[2]);
    AddDstTensor(graph, op, "dst", outputs[0]);

    const auto& dst_shape = outputs[0].tensor_desc.GetBHWCShape();
    op->SetGridSize(
        int3(dst_shape.w * dst_shape.b, dst_shape.h, (dst_shape.c + 3) / 4));

    op->code_ = GetDiffusionStepCode(outputs[0].tensor_desc.GetDataType());
    return absl::OkStatus();
  }
};

MLD_REGISTER_OP(DiffusionStep);

}  // namespace ml_drift
