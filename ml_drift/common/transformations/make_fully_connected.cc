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

#include "ml_drift/common/transformations/make_fully_connected.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "ml_drift/common/model.h"
#include "ml_drift/common/model_transformer.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

bool IsConvEquivalentToFullyConnected(const Convolution2DAttributes& attr) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return weights_shape.w == 1 &&                //
         weights_shape.h == 1 &&                //
         attr.strides == HW(1, 1) &&            //
         attr.dilations == HW(1, 1) &&          //
         attr.padding.prepended == HW(0, 0) &&  //
         attr.padding.appended == HW(0, 0);
}

class MakeFullyConnectedFromConvolution : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::CONVOLUTION_2D)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto inputs = graph->FindInputs(node->id);
    if (inputs.size() != 1) {
      return {TransformStatus::SKIPPED, ""};
    }

    const auto& input_shape = inputs[0]->tensor.shape;
    if (input_shape.w != 1 || input_shape.h != 1) {
      return {TransformStatus::SKIPPED, ""};
    }

    const auto& conv_attr = std::any_cast<const Convolution2DAttributes&>(
        node->operation.attributes);
    if (!IsConvEquivalentToFullyConnected(conv_attr)) {
      return {TransformStatus::SKIPPED, ""};
    }

    FullyConnectedAttributes fc_attr;
    fc_attr.weights = GetFloatWeights(conv_attr);
    fc_attr.bias = conv_attr.bias;

    node->operation.attributes = fc_attr;
    node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    return {TransformStatus::APPLIED,
            "Replaced convolution with fully connected."};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewMakeFullyConnectedFromConvolution() {
  return std::make_unique<MakeFullyConnectedFromConvolution>();
}

}  // namespace ml_drift
