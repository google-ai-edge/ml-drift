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

#include "ml_drift/common/transformations/remove_noop.h"

#include <algorithm>
#include <any>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_transformer.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

using ShouldRemoveOperation = std::function<bool(GraphFloat32* graph, Node*)>;

class RemoveOperation : public SequenceTransformation {
 public:
  explicit RemoveOperation(ShouldRemoveOperation remove_predicate)
      : remove_predicate_(std::move(remove_predicate)) {}

  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    Node* prev_op_node = sequence.front();
    Node* op_node = sequence.back();
    if (!remove_predicate_(graph, op_node)) {
      return {TransformStatus::kSkipped, ""};
    }
    absl::Status status = RemoveFollowingNode(graph, op_node, prev_op_node);
    if (!status.ok()) {
      return {TransformStatus::kInvalid,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::kApplied, ""};
  }

 private:
  ShouldRemoveOperation remove_predicate_;
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputConcat() {
  // Using SequenceTransformation implies that CONCAT has a single input.
  auto type = ToString(OperationType::kConcat);
  return std::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        return type == node->operation.type;
      });
}

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputAdd() {
  // Using SequenceTransformation implies that ADD has a single input.
  auto type = ToString(OperationType::kAdd);
  return std::make_unique<RemoveOperation>([type](GraphFloat32* graph,
                                                  Node* node) {
    if (node->operation.type != type) {
      return false;
    }
    auto& attr =
        std::any_cast<const ElementwiseAttributes&>(node->operation.attributes);
    return std::holds_alternative<std::monostate>(attr.param);
  });
}

std::unique_ptr<SequenceTransformation> NewRemoveDegenerateUpsampling() {
  auto type = ToString(OperationType::kResize);
  return std::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        if (node->operation.type != type) {
          return false;
        }
        auto inputs = graph->FindInputs(node->id);
        auto outputs = graph->FindOutputs(node->id);
        return inputs.size() == 1 && outputs.size() == 1 &&
               inputs[0]->tensor.shape == outputs[0]->tensor.shape;
      });
}

class RemoveIdentityReshape : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::kReshape)) {
      return {TransformStatus::kSkipped, ""};
    }
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    const auto& reshape_attr =
        std::any_cast<const ReshapeAttributes&>(node->operation.attributes);
    if (input_shape != reshape_attr.new_shape) {
      return {TransformStatus::kSkipped, ""};
    }
    {
      // Check if any of the consumers of the reshape node is already using the
      // input of the reshape node.
      const auto input_id = graph->FindInputs(node->id)[0]->id;
      const auto output_id = graph->FindOutputs(node->id)[0]->id;
      const auto consumers = graph->FindConsumers(output_id);
      for (auto& consumer : consumers) {
        const auto consumer_inputs = graph->FindInputs(consumer->id);
        if (std::find_if(consumer_inputs.begin(), consumer_inputs.end(),
                         [&input_id](const Value* value) {
                           return value != nullptr && value->id == input_id;
                         }) != consumer_inputs.end()) {
          // The reshape node might double as a broadcast op,
          // e.g. x is a tensor with shape [2]
          //   mul(x, reshape(x, [1, 2]))
          // Not skipping it will crash RemoveSimpleNodeKeepInput as it assumes
          // the consumer is not using the input of the reshape node.
          return {
              TransformStatus::kSkipped,
              "One of the consumers also uses the input of the reshape node."};
        }
      }
    }
    auto output = graph->FindOutputs(node->id)[0];
    const auto& graph_outputs = graph->outputs();
    if (std::find(graph_outputs.begin(), graph_outputs.end(), output) !=
        graph_outputs.end()) {
      return {TransformStatus::kSkipped,
              "Can not apply transformation when node output is graph output"};
    }
    absl::Status status = RemoveSimpleNodeKeepInput(graph, node);
    if (!status.ok()) {
      return {TransformStatus::kInvalid,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::kApplied,
            "Removed reshape with input_shape == output_shape."};
  }
};

std::unique_ptr<NodeTransformation> NewRemoveIdentityReshape() {
  return std::make_unique<RemoveIdentityReshape>();
}

class MergeConsecutiveReshapes : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
    Node* first_node = sequence.front();
    Node* second_node = sequence.back();
    auto reshape_type = ToString(OperationType::kReshape);
    if (first_node->operation.type != reshape_type ||
        second_node->operation.type != reshape_type) {
      return {TransformStatus::kSkipped, ""};
    }

    auto first_inputs = graph->FindInputs(first_node->id);
    auto first_outputs = graph->FindOutputs(first_node->id);
    auto second_inputs = graph->FindInputs(second_node->id);
    auto second_outputs = graph->FindOutputs(second_node->id);

    if (first_inputs.size() != 1 || first_outputs.size() != 1 ||
        second_inputs.size() != 1 || second_outputs.size() != 1) {
      return {TransformStatus::kSkipped, ""};
    }

    Value* intermediate_value = first_outputs[0];
    if (second_inputs[0]->id != intermediate_value->id) {
      return {TransformStatus::kSkipped, ""};
    }

    // Make sure the intermediate value has no other consumers.
    if (graph->FindConsumers(intermediate_value->id).size() != 1) {
      return {TransformStatus::kSkipped,
              "Intermediate value between consecutive reshapes has multiple "
              "consumers."};
    }

    // Intermediate value must not be a graph output.
    if (graph->IsGraphOutput(intermediate_value->id)) {
      return {TransformStatus::kSkipped,
              "Intermediate value between consecutive reshapes is a graph "
              "output."};
    }

    // Update first_node's target shape to second_node's target shape.
    const auto& second_attr = std::any_cast<const ReshapeAttributes&>(
        second_node->operation.attributes);
    first_node->operation.attributes = second_attr;

    // Remove second_node and rewire its outputs to be produced by first_node.
    absl::Status status = RemoveFollowingNode(graph, second_node, first_node);
    if (!status.ok()) {
      return {TransformStatus::kInvalid,
              "Unable to remove following reshape node: " +
                  std::string(status.message())};
    }
    return {TransformStatus::kApplied, "Merged consecutive reshapes into one."};
  }
};

std::unique_ptr<SequenceTransformation> NewMergeConsecutiveReshapes() {
  return std::make_unique<MergeConsecutiveReshapes>();
}

class RemoveIdentityStridedSlice : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::kSlice)) {
      return {TransformStatus::kSkipped, ""};
    }
    auto input = graph->FindInputs(node->id)[0];
    auto output = graph->FindOutputs(node->id)[0];
    const auto& slice_attr =
        std::any_cast<const SliceAttributes&>(node->operation.attributes);
    if (input->tensor.shape != output->tensor.shape) {
      return {TransformStatus::kSkipped, ""};
    }
    if (slice_attr.starts != BHWC(0, 0, 0, 0)) {
      return {TransformStatus::kSkipped, ""};
    }
    if (slice_attr.strides != BHWC(1, 1, 1, 1)) {
      return {TransformStatus::kSkipped, ""};
    }
    if (slice_attr.ends != output->tensor.shape) {
      return {TransformStatus::kSkipped, ""};
    }
    const auto& graph_outputs = graph->outputs();
    const auto& graph_inputs = graph->inputs();
    const bool input_is_graph_input =
        std::find(graph_inputs.begin(), graph_inputs.end(), input) !=
        graph_inputs.end();
    const bool output_is_graph_output =
        std::find(graph_outputs.begin(), graph_outputs.end(), output) !=
        graph_outputs.end();
    if (input_is_graph_input && output_is_graph_output) {
      return {TransformStatus::kSkipped,
              "Can not apply transformation when node input is graph input and "
              "node output is graph output"};
    }
    if (output_is_graph_output) {
      if (graph->FindConsumers(input->id).size() != 1) {
        return {TransformStatus::kSkipped,
                "Can not apply transformation when node output is graph output "
                "and input consumed by other nodes."};
      }
      absl::Status status = RemoveSimpleNodeKeepOutput(graph, node);
      if (!status.ok()) {
        return {TransformStatus::kInvalid,
                "Unable to remove a node: " + std::string(status.message())};
      }
      return {TransformStatus::kApplied, "Removed identity strided slice."};
    }
    absl::Status status = RemoveSimpleNodeKeepInput(graph, node);
    if (!status.ok()) {
      return {TransformStatus::kInvalid,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::kApplied, "Removed identity strided slice."};
  }
};

std::unique_ptr<NodeTransformation> NewRemoveIdentityStridedSlice() {
  return std::make_unique<RemoveIdentityStridedSlice>();
}

}  // namespace ml_drift
