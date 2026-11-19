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

#include "ml_drift/common/transformations/global_pooling_to_reduce_op.h"

#include <any>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ml_drift/common/model.h"
#include "ml_drift/common/model_transformer.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

bool IsGlobalPooling(const Pooling2DAttributes& attr, const BHWC& src_shape,
                     const BHWC& dst_shape) {
  return dst_shape.w == 1 && dst_shape.h == 1 && attr.kernel.w == src_shape.w &&
         attr.kernel.h == src_shape.h && attr.padding.appended.w == 0 &&
         attr.padding.appended.h == 0 && attr.padding.prepended.w == 0 &&
         attr.padding.prepended.h == 0;
}

bool IsGlobalAveragePooling(const Pooling2DAttributes& attr,
                            const BHWC& src_shape, const BHWC& dst_shape) {
  return attr.type == PoolingType::kAverage && attr.output_indices == false &&
         IsGlobalPooling(attr, src_shape, dst_shape);
}

class GlobalPoolingToReduceOp : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::kPooling2D)) {
      return {TransformStatus::kSkipped, ""};
    }

    auto inputs = graph->FindInputs(node->id);
    auto outputs = graph->FindOutputs(node->id);
    const auto& pool_attr =
        std::any_cast<const Pooling2DAttributes&>(node->operation.attributes);
    if (!IsGlobalAveragePooling(pool_attr, inputs[0]->tensor.shape,
                                outputs[0]->tensor.shape)) {
      return {TransformStatus::kSkipped, ""};
    }

    ReduceAttributes mean_attr;
    mean_attr.dims = {Axis::kWidth, Axis::kHeight};

    node->operation.attributes = mean_attr;
    node->operation.type = ToString(OperationType::kMean);
    return {TransformStatus::kApplied,
            "Replaced global average pooling with mean."};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewGlobalPoolingToReduceOp() {
  return std::make_unique<GlobalPoolingToReduceOp>();
}

}  // namespace ml_drift
