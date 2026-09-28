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

#include "ml_drift/common/kernels/mean_stddev_normalization_parser.h"

#include <any>
#include <memory>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/kernels/mean_stddev_normalization.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {
absl::Status CheckIfValidNodeOfType(const Node* node,
                                    OperationType required_type) {
  if (node == nullptr) {
    return absl::NotFoundError("Invalid node.");
  }
  if (OperationTypeFromString(node->operation.type) != required_type) {
    return absl::NotFoundError("Type mismatch.");
  }
  return absl::OkStatus();
}

absl::Status GetElementwiseScalarValue(const Node* node, float* result) {
  auto attr = std::any_cast<ElementwiseAttributes>(node->operation.attributes);
  const float* value = GetIfFloatScalar(&attr.param);
  if (!value) {
    return absl::NotFoundError("Not a scalar value inside attributes.");
  }
  *result = *value;
  return absl::OkStatus();
}

absl::Status GetElementwiseLinearValue(
    const Node* node, Tensor<Linear, DataType::kFloat32>* result) {
  auto attr = std::any_cast<ElementwiseAttributes>(node->operation.attributes);
  const auto* linear_tensor =
      std::get_if<Tensor<Linear, DataType::kFloat32>>(&attr.param);
  if (!linear_tensor) {
    return absl::NotFoundError("Not a linear value inside attributes.");
  }
  *result = *linear_tensor;
  return absl::OkStatus();
}

absl::Status GetNextSingleNode(const GraphFloat32& graph, const Node& node,
                               OperationType next_type, Node** next_node) {
  auto consumers = graph.FindConsumers(graph.FindOutputs(node.id)[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("Not a single consumer.");
  }
  ABSL_RETURN_IF_ERROR(CheckIfValidNodeOfType(consumers[0], next_type));
  *next_node = consumers[0];
  return absl::OkStatus();
}

//       input
//       /    \
//      |    mean
//       \    /
//     subtraction
//       /    \
//      |      |
//      |    square
//      |      |
//      |     mean
//      |      |
//      |     add
//      |      |
//      |    rsqrt
//      |      |
//       \    /
//    multiplication
//          |
//        output
absl::Status TryMeanStdDevNormalizationV0(
    const GpuInfo& gpu_info, const GraphFloat32& graph, NodeId first_node_id,
    const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
  Node* first_mean_node = graph.GetNode(first_node_id);
  ABSL_RETURN_IF_ERROR(
      CheckIfValidNodeOfType(first_mean_node, OperationType::kMean));
  auto first_mean_attr =
      std::any_cast<ReduceAttributes>(first_mean_node->operation.attributes);
  if (first_mean_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  Node* sub_node;
  ABSL_RETURN_IF_ERROR(GetNextSingleNode(graph, *first_mean_node,
                                         OperationType::kSub, &sub_node));
  auto sub_inputs = graph.FindInputs(sub_node->id);
  if (sub_inputs.size() != 2) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  } else {
    // checking structure
    //       input
    //       /    \
    //      |    mean
    //       \    /
    //     subtraction
    Node* sub_first_parent = graph.FindProducer(sub_inputs[0]->id);
    Node* sub_second_parent = graph.FindProducer(sub_inputs[1]->id);
    if (sub_second_parent != first_mean_node) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
    auto mean_inputs = graph.FindInputs(first_mean_node->id);
    Node* mean_parent = graph.FindProducer(mean_inputs[0]->id);
    if (mean_parent != sub_first_parent) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }
  auto sub_output = graph.FindOutputs(sub_node->id)[0]->id;
  auto consumers = graph.FindConsumers(sub_output);
  if (consumers.size() != 2) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  Node* square_node = consumers[0];
  Node* sub_child_mul_node = consumers[1];
  if (!CheckIfValidNodeOfType(square_node, OperationType::kSquare).ok()) {
    square_node = consumers[1];
    sub_child_mul_node = consumers[0];
  }
  ABSL_RETURN_IF_ERROR(
      CheckIfValidNodeOfType(square_node, OperationType::kSquare));
  ABSL_RETURN_IF_ERROR(
      CheckIfValidNodeOfType(sub_child_mul_node, OperationType::kMul));
  Node* second_mean_node;
  ABSL_RETURN_IF_ERROR(GetNextSingleNode(
      graph, *square_node, OperationType::kMean, &second_mean_node));
  auto second_mean_attr =
      std::any_cast<ReduceAttributes>(second_mean_node->operation.attributes);
  if (second_mean_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  Node* add_node;
  ABSL_RETURN_IF_ERROR(GetNextSingleNode(graph, *second_mean_node,
                                         OperationType::kAdd, &add_node));
  float add_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(add_node, &add_value));
  Node* rsqrt_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *add_node, OperationType::kRsqrt, &rsqrt_node));
  Node* mul_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *rsqrt_node, OperationType::kMul, &mul_node));
  if (sub_child_mul_node != mul_node) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  auto input = graph.FindInputs(first_mean_node->id)[0];
  auto output_id = graph.FindOutputs(mul_node->id)[0]->id;
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input->id));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto gpu_op =
      std::make_unique<MeanStdDevNormalization>(CreateMeanStdDevNormalization(
          op_def, gpu_info, input->tensor.shape, add_value,
          /*two_step=*/false));
  model_builder->AddGpuOperation(
      std::vector<ValueId>({input->id}), std::vector<ValueId>({output_id}),
      std::move(gpu_op), "mean_stddev_normalization");

  new_consumed_nodes->insert(first_mean_node->id);
  new_consumed_nodes->insert(sub_node->id);
  new_consumed_nodes->insert(square_node->id);
  new_consumed_nodes->insert(second_mean_node->id);
  new_consumed_nodes->insert(add_node->id);
  new_consumed_nodes->insert(rsqrt_node->id);
  new_consumed_nodes->insert(mul_node->id);

  return absl::OkStatus();
}

//        input_tensor
//       /     |      \
//   square0  mean0    |
//     |      /   \   /
//   mean1 square1 sub0
//      \   /       |
//       sub1       |
//        |         |
//       add        |
//        |         |
//      rsqrt       |
//        |         |
//     mul_ones     |
//        \        /
//      multiplication
//            |
//          output
absl::Status TryMeanStdDevNormalizationV1(
    const GpuInfo& gpu_info, const GraphFloat32& graph, NodeId first_node_id,
    const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
  Node* first_node = graph.GetNode(first_node_id);
  auto first_op_type = OperationTypeFromString(first_node->operation.type);
  if (first_op_type != OperationType::kMean &&
      first_op_type != OperationType::kSquare) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  Node* mean0_node = nullptr;
  Node* sub0_node = nullptr;
  Node* square0_node = nullptr;
  {
    // checking this structure
    //        input_tensor
    //       /     |      \
    //   square0  mean0    |
    //                     |
    //                    sub0
    auto first_node_inputs = graph.FindInputs(first_node->id);
    if (first_node_inputs.size() != 1) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
    auto consumers = graph.FindConsumers(first_node_inputs[0]->id);
    for (auto node : consumers) {
      auto op_type = OperationTypeFromString(node->operation.type);
      if (op_type == OperationType::kMean) {
        mean0_node = node;
      } else if (op_type == OperationType::kSquare) {
        square0_node = node;
      } else if (op_type == OperationType::kSub) {
        sub0_node = node;
      }
    }
    if (!square0_node || !sub0_node || !mean0_node) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  if (consumed_nodes.find(square0_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  if (consumed_nodes.find(mean0_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  auto mean0_attr =
      std::any_cast<ReduceAttributes>(mean0_node->operation.attributes);
  if (mean0_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  Node* square1_node = nullptr;
  {
    // checking this structure
    //            mean0
    //            /   \
    //         square1 sub0
    auto mean0_output = graph.FindOutputs(mean0_node->id)[0]->id;
    auto consumers = graph.FindConsumers(mean0_output);
    if (consumers.size() != 2) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }

    Node* sub0_copy_node = nullptr;
    auto op0_type = OperationTypeFromString(consumers[0]->operation.type);
    auto op1_type = OperationTypeFromString(consumers[1]->operation.type);
    if (op0_type == OperationType::kSquare) {
      square1_node = consumers[0];
    } else if (op0_type == OperationType::kSub) {
      sub0_copy_node = consumers[0];
    }

    if (op1_type == OperationType::kSquare) {
      square1_node = consumers[1];
    } else if (op1_type == OperationType::kSub) {
      sub0_copy_node = consumers[1];
    }

    if (!square1_node || !sub0_copy_node || sub0_copy_node != sub0_node) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  Node* mean1_node;
  ABSL_RETURN_IF_ERROR(GetNextSingleNode(graph, *square0_node,
                                         OperationType::kMean, &mean1_node));

  Node* sub1_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *mean1_node, OperationType::kSub, &sub1_node));

  {
    // checking this structure
    //   mean1 square1
    //      \   /
    //       sub1
    Node* sub1_copy_node;
    ABSL_RETURN_IF_ERROR(GetNextSingleNode(
        graph, *square1_node, OperationType::kSub, &sub1_copy_node));
    if (sub1_copy_node != sub1_node) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  Node* multiplication_node;
  ABSL_RETURN_IF_ERROR(GetNextSingleNode(graph, *sub0_node, OperationType::kMul,
                                         &multiplication_node));

  Node* add_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *sub1_node, OperationType::kAdd, &add_node));
  float add_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(add_node, &add_value));

  Node* rsqrt_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *add_node, OperationType::kRsqrt, &rsqrt_node));

  Node* mul_ones_node;
  ABSL_RETURN_IF_ERROR(GetNextSingleNode(graph, *rsqrt_node,
                                         OperationType::kMul, &mul_ones_node));
  Tensor<Linear, DataType::kFloat32> mul_linear_value;
  ABSL_RETURN_IF_ERROR(
      GetElementwiseLinearValue(mul_ones_node, &mul_linear_value));
  for (int i = 0; i < mul_linear_value.data.size(); ++i) {
    if (mul_linear_value.data[i] != 1.0f) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  {
    // checking this structure
    //   mul_ones sub0
    //      \      /
    //    multiplication
    Node* multiplication_copy_node;
    ABSL_RETURN_IF_ERROR(GetNextSingleNode(
        graph, *mul_ones_node, OperationType::kMul, &multiplication_copy_node));
    if (multiplication_copy_node != multiplication_node) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  auto input = graph.FindInputs(mean0_node->id)[0];
  auto output_id = graph.FindOutputs(multiplication_node->id)[0]->id;
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input->id));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto gpu_op =
      std::make_unique<MeanStdDevNormalization>(CreateMeanStdDevNormalization(
          op_def, gpu_info, input->tensor.shape, add_value,
          /*two_step=*/false));
  model_builder->AddGpuOperation(
      std::vector<ValueId>({input->id}), std::vector<ValueId>({output_id}),
      std::move(gpu_op), "mean_stddev_normalization");

  new_consumed_nodes->insert(mean0_node->id);
  new_consumed_nodes->insert(mean1_node->id);
  new_consumed_nodes->insert(square0_node->id);
  new_consumed_nodes->insert(square1_node->id);
  new_consumed_nodes->insert(sub0_node->id);
  new_consumed_nodes->insert(sub1_node->id);
  new_consumed_nodes->insert(add_node->id);
  new_consumed_nodes->insert(rsqrt_node->id);
  new_consumed_nodes->insert(mul_ones_node->id);
  new_consumed_nodes->insert(multiplication_node->id);

  return absl::OkStatus();
}
}  // namespace

absl::Status TryMeanStdDevNormalization(const GpuInfo& gpu_info,
                                        const GraphFloat32& graph,
                                        NodeId first_node_id,
                                        const std::set<NodeId>& consumed_nodes,
                                        std::set<NodeId>* new_consumed_nodes,
                                        GpuModelBuilder* model_builder) {
  auto status = TryMeanStdDevNormalizationV0(gpu_info, graph, first_node_id,
                                             consumed_nodes, new_consumed_nodes,
                                             model_builder);
  if (status.ok()) {
    return status;
  }
  return TryMeanStdDevNormalizationV1(gpu_info, graph, first_node_id,
                                      consumed_nodes, new_consumed_nodes,
                                      model_builder);
}

// LayerNormalization fusion works with this subgraph
//    input_tensor
//      /  /   \
//     /  |    mean0
//    /    \   /   \
//   |    sq_diff   |
//   |       |      |
//   |     mean1    |
//   |       |      |
//   |     add0     |
//   |       |      |
//   |     rsqrt    |
//   |       |      |
//    \    mul0    /
//     \   /   \  /
//      mul1    mul2
//       |       |
//        \     sub
//         \   /
//          add1
//           |
//     output_tensor
absl::Status TryLayerNormalization(const GpuInfo& gpu_info,
                                   const GraphFloat32& graph,
                                   NodeId first_node_id,
                                   const std::set<NodeId>& consumed_nodes,
                                   std::set<NodeId>* new_consumed_nodes,
                                   GpuModelBuilder* model_builder) {
  Node* mean0_node = graph.GetNode(first_node_id);
  ABSL_RETURN_IF_ERROR(
      CheckIfValidNodeOfType(mean0_node, OperationType::kMean));
  auto mean0_attr =
      std::any_cast<ReduceAttributes>(mean0_node->operation.attributes);
  if (mean0_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  auto mean0_output = graph.FindOutputs(mean0_node->id)[0]->id;
  auto consumers = graph.FindConsumers(mean0_output);
  if (consumers.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  Node* sq_diff_node = consumers[0];
  Node* mul2_node = consumers[1];
  if (!CheckIfValidNodeOfType(sq_diff_node, OperationType::kSquaredDiff).ok()) {
    std::swap(sq_diff_node, mul2_node);
  }
  ABSL_RETURN_IF_ERROR(
      CheckIfValidNodeOfType(sq_diff_node, OperationType::kSquaredDiff));
  ABSL_RETURN_IF_ERROR(CheckIfValidNodeOfType(mul2_node, OperationType::kMul));

  auto sq_diff_inputs = graph.FindInputs(sq_diff_node->id);
  if (sq_diff_inputs.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  } else {
    // checking structure
    //       input
    //       /    \
    //      |    mean0
    //       \    /
    //       sq_diff
    Node* sq_diff_first_parent = graph.FindProducer(sq_diff_inputs[0]->id);
    Node* sq_diff_second_parent = graph.FindProducer(sq_diff_inputs[1]->id);
    if (sq_diff_second_parent != mean0_node) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
    auto mean0_inputs = graph.FindInputs(mean0_node->id);
    Node* mean0_parent = graph.FindProducer(mean0_inputs[0]->id);
    if (mean0_parent != sq_diff_first_parent) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
  }
  Node* mean1_node;
  ABSL_RETURN_IF_ERROR(GetNextSingleNode(graph, *sq_diff_node,
                                         OperationType::kMean, &mean1_node));
  auto mean1_attr =
      std::any_cast<ReduceAttributes>(mean1_node->operation.attributes);
  if (mean1_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  Node* add0_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *mean1_node, OperationType::kAdd, &add0_node));
  float add_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(add0_node, &add_value));
  Node* rsqrt_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *add0_node, OperationType::kRsqrt, &rsqrt_node));
  Node* mul0_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *rsqrt_node, OperationType::kMul, &mul0_node));
  Tensor<Linear, DataType::kFloat32> mul_linear_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseLinearValue(mul0_node, &mul_linear_value));
  auto mul0_output = graph.FindOutputs(mul0_node->id)[0]->id;
  consumers = graph.FindConsumers(mul0_output);
  if (consumers.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  if (!(consumers[0] == mul2_node || consumers[1] == mul2_node)) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  Node* mul1_node = consumers[0] == mul2_node ? consumers[1] : consumers[0];
  if (!CheckIfValidNodeOfType(sq_diff_node, OperationType::kSquaredDiff).ok()) {
    std::swap(sq_diff_node, mul2_node);
  }
  auto mul1_inputs = graph.FindInputs(mul1_node->id);
  if (mul1_inputs.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  } else {
    Node* mul1_first_parent = graph.FindProducer(mul1_inputs[0]->id);
    Node* mul1_second_parent = graph.FindProducer(mul1_inputs[1]->id);
    auto mean0_inputs = graph.FindInputs(mean0_node->id);
    Node* mean0_parent = graph.FindProducer(mean0_inputs[0]->id);
    if (!(mul1_first_parent == mean0_parent ||
          mul1_second_parent == mean0_parent)) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
    if (!(mul1_first_parent == mul0_node || mul1_second_parent == mul0_node)) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
  }
  Node* sub_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *mul2_node, OperationType::kSub, &sub_node));
  Tensor<Linear, DataType::kFloat32> sub_linear_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseLinearValue(sub_node, &sub_linear_value));

  Node* add1_node;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *sub_node, OperationType::kAdd, &add1_node));
  {
    Node* add1_copy_node;
    ABSL_RETURN_IF_ERROR(GetNextSingleNode(
        graph, *mul1_node, OperationType::kAdd, &add1_copy_node));
    if (add1_copy_node != add1_node) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
  }

  auto input = graph.FindInputs(mean0_node->id)[0];
  auto output_id = graph.FindOutputs(add1_node->id)[0]->id;
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input->id));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto gpu_op =
      std::make_unique<MeanStdDevNormalization>(CreateMeanStdDevNormalization(
          op_def, gpu_info, input->tensor.shape, add_value, mul_linear_value,
          sub_linear_value,
          /*two_step=*/false));
  model_builder->AddGpuOperation(std::vector<ValueId>({input->id}),
                                 std::vector<ValueId>({output_id}),
                                 std::move(gpu_op), "layer_normalization");

  new_consumed_nodes->insert(mean0_node->id);
  new_consumed_nodes->insert(sq_diff_node->id);
  new_consumed_nodes->insert(mean1_node->id);
  new_consumed_nodes->insert(add0_node->id);
  new_consumed_nodes->insert(rsqrt_node->id);
  new_consumed_nodes->insert(mul0_node->id);
  new_consumed_nodes->insert(mul1_node->id);
  new_consumed_nodes->insert(mul2_node->id);
  new_consumed_nodes->insert(sub_node->id);
  new_consumed_nodes->insert(add1_node->id);

  return absl::OkStatus();
}

namespace {
absl::Status CheckIfValidOpOfType(const ir::IrOp* op,
                                  OperationType required_type) {
  if (op == nullptr) {
    return absl::NotFoundError("Invalid node.");
  }
  if (OperationTypeFromString(op->name) != required_type) {
    return absl::NotFoundError("Type mismatch.");
  }
  return absl::OkStatus();
}

absl::Status GetElementwiseScalarValue(const ir::IrOp* op, float* result) {
  auto attr = std::any_cast<ElementwiseAttributes>(op->attr);
  const float* value = GetIfFloatScalar(&attr.param);
  if (!value) {
    return absl::NotFoundError("Not a scalar value inside attributes.");
  }
  *result = *value;
  return absl::OkStatus();
}

absl::Status GetElementwiseLinearValue(
    const ir::IrOp* op, Tensor<Linear, DataType::kFloat32>* result) {
  auto attr = std::any_cast<ElementwiseAttributes>(op->attr);
  const auto* linear_tensor =
      std::get_if<Tensor<Linear, DataType::kFloat32>>(&attr.param);
  if (!linear_tensor) {
    return absl::NotFoundError("Not a linear value inside attributes.");
  }
  *result = *linear_tensor;
  return absl::OkStatus();
}

absl::Status GetNextSingleOp(const ir::IrModel& ir_model, const ir::IrOp& op,
                             OperationType next_type,
                             const ir::IrOp** next_op) {
  if (op.outputs.empty()) return absl::NotFoundError("No outputs.");
  auto consumers = ir_model.FindConsumers(op.outputs[0]);
  if (consumers.size() != 1) {
    return absl::NotFoundError("Not a single consumer.");
  }
  const ir::IrOp* consumer = consumers[0];
  ABSL_RETURN_IF_ERROR(CheckIfValidOpOfType(consumer, next_type));
  *next_op = consumer;
  return absl::OkStatus();
}

//       input
//       /    \
//      |    mean
//       \    /
//     subtraction
//       /    \
//      |      |
//      |    square
//      |      |
//      |     mean
//      |      |
//      |     add
//      |      |
//      |    rsqrt
//      |      |
//       \    /
//    multiplication
//          |
//        output
absl::Status TryMeanStdDevNormalizationV0(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  const ir::IrOp* first_mean_op = ir_model.op(first_op_id);
  ABSL_RETURN_IF_ERROR(
      CheckIfValidOpOfType(first_mean_op, OperationType::kMean));
  auto first_mean_attr = std::any_cast<ReduceAttributes>(first_mean_op->attr);
  if (first_mean_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  const ir::IrOp* sub_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *first_mean_op, OperationType::kSub, &sub_op));
  auto sub_inputs = sub_op->inputs;
  if (sub_inputs.size() != 2) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  } else {
    // checking structure
    //       input
    //       /    \
    //      |    mean
    //       \    /
    //     subtraction
    const ir::IrOp* sub_first_parent = ir_model.FindProducer(sub_inputs[0]);
    const ir::IrOp* sub_second_parent = ir_model.FindProducer(sub_inputs[1]);
    if (sub_second_parent != first_mean_op) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
    auto mean_inputs = first_mean_op->inputs;
    if (mean_inputs.empty())
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    const ir::IrOp* mean_parent = ir_model.FindProducer(mean_inputs[0]);
    if (mean_parent != sub_first_parent) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }
  auto sub_output = sub_op->outputs[0];
  auto consumers = ir_model.FindConsumers(sub_output);
  if (consumers.size() != 2) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  const ir::IrOp* square_op = consumers[0];
  const ir::IrOp* sub_child_mul_op = consumers[1];
  if (!CheckIfValidOpOfType(square_op, OperationType::kSquare).ok()) {
    std::swap(square_op, sub_child_mul_op);
  }
  ABSL_RETURN_IF_ERROR(CheckIfValidOpOfType(square_op, OperationType::kSquare));
  ABSL_RETURN_IF_ERROR(
      CheckIfValidOpOfType(sub_child_mul_op, OperationType::kMul));
  const ir::IrOp* second_mean_op;
  ABSL_RETURN_IF_ERROR(GetNextSingleOp(ir_model, *square_op,
                                       OperationType::kMean, &second_mean_op));
  auto second_mean_attr = std::any_cast<ReduceAttributes>(second_mean_op->attr);
  if (second_mean_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  const ir::IrOp* add_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *second_mean_op, OperationType::kAdd, &add_op));
  float add_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(add_op, &add_value));
  const ir::IrOp* rsqrt_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *add_op, OperationType::kRsqrt, &rsqrt_op));
  const ir::IrOp* mul_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *rsqrt_op, OperationType::kMul, &mul_op));
  if (sub_child_mul_op != mul_op) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  auto input = first_mean_op->inputs[0];
  auto output_id = mul_op->outputs[0];
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto gpu_op =
      std::make_unique<MeanStdDevNormalization>(CreateMeanStdDevNormalization(
          op_def, gpu_info, ir_model.tensor(input)->desc.GetBHWCShape(),
          add_value,
          /*two_step=*/false));
  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(input)}),
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(output_id)}),
      std::move(gpu_op), "mean_stddev_normalization");

  new_consumed_ops->insert(first_mean_op->id);
  new_consumed_ops->insert(sub_op->id);
  new_consumed_ops->insert(square_op->id);
  new_consumed_ops->insert(second_mean_op->id);
  new_consumed_ops->insert(add_op->id);
  new_consumed_ops->insert(rsqrt_op->id);
  new_consumed_ops->insert(mul_op->id);

  return absl::OkStatus();
}

//        input_tensor
//       /     |      \
//   square0  mean0    |
//     |      /   \   /
//   mean1 square1 sub0
//      \   /       |
//       sub1       |
//        |         |
//       add        |
//        |         |
//      rsqrt       |
//        |         |
//     mul_ones     |
//        \        /
//      multiplication
//            |
//          output
absl::Status TryMeanStdDevNormalizationV1(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  const ir::IrOp* first_op = ir_model.op(first_op_id);
  auto first_op_type = OperationTypeFromString(first_op->name);
  if (first_op_type != OperationType::kMean &&
      first_op_type != OperationType::kSquare) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  const ir::IrOp* mean0_op = nullptr;
  const ir::IrOp* sub0_op = nullptr;
  const ir::IrOp* square0_op = nullptr;
  {
    // checking this structure
    //        input_tensor
    //       /     |      \
    //   square0  mean0    |
    //                     |
    //                    sub0
    auto first_op_inputs = first_op->inputs;
    if (first_op_inputs.size() != 1) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
    auto consumers = ir_model.FindConsumers(first_op_inputs[0]);
    for (auto* op : consumers) {
      auto op_type = OperationTypeFromString(op->name);
      if (op_type == OperationType::kMean) {
        mean0_op = op;
      } else if (op_type == OperationType::kSquare) {
        square0_op = op;
      } else if (op_type == OperationType::kSub) {
        sub0_op = op;
      }
    }
    if (!square0_op || !sub0_op || !mean0_op) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  if (consumed_ops.find(square0_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }
  if (consumed_ops.find(mean0_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  auto mean0_attr = std::any_cast<ReduceAttributes>(mean0_op->attr);
  if (mean0_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("MeanStdDevNormalization not suitable.");
  }

  const ir::IrOp* square1_op = nullptr;
  {
    // checking this structure
    //            mean0
    //            /   \
    //         square1 sub0
    auto mean0_output = mean0_op->outputs.empty() ? 0 : mean0_op->outputs[0];
    auto consumers = ir_model.FindConsumers(mean0_output);
    if (consumers.size() != 2) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }

    const ir::IrOp* sub0_copy_op = nullptr;
    auto op0_type = OperationTypeFromString(consumers[0]->name);
    auto op1_type = OperationTypeFromString(consumers[1]->name);
    if (op0_type == OperationType::kSquare) {
      square1_op = consumers[0];
    } else if (op0_type == OperationType::kSub) {
      sub0_copy_op = consumers[0];
    }

    if (op1_type == OperationType::kSquare) {
      square1_op = consumers[1];
    } else if (op1_type == OperationType::kSub) {
      sub0_copy_op = consumers[1];
    }

    if (!square1_op || !sub0_copy_op || sub0_copy_op != sub0_op) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  const ir::IrOp* mean1_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *square0_op, OperationType::kMean, &mean1_op));

  const ir::IrOp* sub1_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *mean1_op, OperationType::kSub, &sub1_op));

  {
    // checking this structure
    //   mean1 square1
    //      \   /
    //       sub1
    const ir::IrOp* sub1_copy_op;
    ABSL_RETURN_IF_ERROR(GetNextSingleOp(ir_model, *square1_op,
                                         OperationType::kSub, &sub1_copy_op));
    if (sub1_copy_op != sub1_op) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  const ir::IrOp* multiplication_op;
  ABSL_RETURN_IF_ERROR(GetNextSingleOp(ir_model, *sub0_op, OperationType::kMul,
                                       &multiplication_op));

  const ir::IrOp* add_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *sub1_op, OperationType::kAdd, &add_op));
  float add_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(add_op, &add_value));

  const ir::IrOp* rsqrt_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *add_op, OperationType::kRsqrt, &rsqrt_op));

  const ir::IrOp* mul_ones_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *rsqrt_op, OperationType::kMul, &mul_ones_op));
  Tensor<Linear, DataType::kFloat32> mul_linear_value;
  ABSL_RETURN_IF_ERROR(
      GetElementwiseLinearValue(mul_ones_op, &mul_linear_value));
  for (int i = 0; i < mul_linear_value.data.size(); ++i) {
    if (mul_linear_value.data[i] != 1.0f) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  {
    // checking this structure
    //   mul_ones sub0
    //      \      /
    //    multiplication
    const ir::IrOp* multiplication_copy_op;
    ABSL_RETURN_IF_ERROR(GetNextSingleOp(
        ir_model, *mul_ones_op, OperationType::kMul, &multiplication_copy_op));
    if (multiplication_copy_op != multiplication_op) {
      return absl::NotFoundError("MeanStdDevNormalization not suitable.");
    }
  }

  auto input = mean0_op->inputs[0];
  auto output_id = multiplication_op->outputs[0];
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto gpu_op =
      std::make_unique<MeanStdDevNormalization>(CreateMeanStdDevNormalization(
          op_def, gpu_info, ir_model.tensor(input)->desc.GetBHWCShape(),
          add_value,
          /*two_step=*/false));
  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(input)}),
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(output_id)}),
      std::move(gpu_op), "mean_stddev_normalization");

  new_consumed_ops->insert(mean0_op->id);
  new_consumed_ops->insert(mean1_op->id);
  new_consumed_ops->insert(square0_op->id);
  new_consumed_ops->insert(square1_op->id);
  new_consumed_ops->insert(sub0_op->id);
  new_consumed_ops->insert(sub1_op->id);
  new_consumed_ops->insert(add_op->id);
  new_consumed_ops->insert(rsqrt_op->id);
  new_consumed_ops->insert(mul_ones_op->id);
  new_consumed_ops->insert(multiplication_op->id);

  return absl::OkStatus();
}
}  // namespace

absl::Status TryMeanStdDevNormalization(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  auto status = TryMeanStdDevNormalizationV0(gpu_info, ir_model, first_op_id,
                                             consumed_ops, new_consumed_ops,
                                             model_builder);
  if (status.ok()) {
    return status;
  }
  return TryMeanStdDevNormalizationV1(gpu_info, ir_model, first_op_id,
                                      consumed_ops, new_consumed_ops,
                                      model_builder);
}

// LayerNormalization fusion works with this subgraph
//    input_tensor
//      /  /   \
//     /  |    mean0
//    /    \   /   \
//   |    sq_diff   |
//   |       |      |
//   |     mean1    |
//   |       |      |
//   |     add0     |
//   |       |      |
//   |     rsqrt    |
//   |       |      |
//    \    mul0    /
//     \   /   \  /
//      mul1    mul2
//       |       |
//        \     sub
//         \   /
//          add1
//           |
//     output_tensor
absl::Status TryLayerNormalization(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  const ir::IrOp* mean0_op = ir_model.op(first_op_id);
  ABSL_RETURN_IF_ERROR(CheckIfValidOpOfType(mean0_op, OperationType::kMean));
  auto mean0_attr = std::any_cast<ReduceAttributes>(mean0_op->attr);
  if (mean0_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  auto mean0_output = mean0_op->outputs[0];
  auto consumers = ir_model.FindConsumers(mean0_output);
  if (consumers.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  const ir::IrOp* sq_diff_op = consumers[0];
  const ir::IrOp* mul2_op = consumers[1];
  if (!CheckIfValidOpOfType(sq_diff_op, OperationType::kSquaredDiff).ok()) {
    std::swap(sq_diff_op, mul2_op);
  }
  ABSL_RETURN_IF_ERROR(
      CheckIfValidOpOfType(sq_diff_op, OperationType::kSquaredDiff));
  ABSL_RETURN_IF_ERROR(CheckIfValidOpOfType(mul2_op, OperationType::kMul));

  auto sq_diff_inputs = sq_diff_op->inputs;
  if (sq_diff_inputs.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  } else {
    // checking structure
    //       input
    //       /    \
    //      |    mean0
    //       \    /
    //       sq_diff
    const ir::IrOp* sq_diff_first_parent =
        ir_model.FindProducer(sq_diff_inputs[0]);
    const ir::IrOp* sq_diff_second_parent =
        ir_model.FindProducer(sq_diff_inputs[1]);
    if (sq_diff_second_parent != mean0_op) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
    auto mean0_inputs = mean0_op->inputs;
    if (mean0_inputs.empty())
      return absl::NotFoundError("LayerNormalization not suitable.");
    const ir::IrOp* mean0_parent = ir_model.FindProducer(mean0_inputs[0]);
    if (mean0_parent != sq_diff_first_parent) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
  }
  const ir::IrOp* mean1_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *sq_diff_op, OperationType::kMean, &mean1_op));
  auto mean1_attr = std::any_cast<ReduceAttributes>(mean1_op->attr);
  if (mean1_attr.dims != std::set<Axis>{Axis::kChannels}) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  const ir::IrOp* add0_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *mean1_op, OperationType::kAdd, &add0_op));
  float add_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(add0_op, &add_value));
  const ir::IrOp* rsqrt_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *add0_op, OperationType::kRsqrt, &rsqrt_op));
  const ir::IrOp* mul0_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *rsqrt_op, OperationType::kMul, &mul0_op));
  Tensor<Linear, DataType::kFloat32> mul_linear_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseLinearValue(mul0_op, &mul_linear_value));
  auto mul0_output = mul0_op->outputs[0];
  auto mul0_consumers = ir_model.FindConsumers(mul0_output);
  if (mul0_consumers.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  if (!(mul0_consumers[0] == mul2_op || mul0_consumers[1] == mul2_op)) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  }
  const ir::IrOp* mul1_op =
      mul0_consumers[0] == mul2_op ? mul0_consumers[1] : mul0_consumers[0];
  if (!CheckIfValidOpOfType(sq_diff_op, OperationType::kSquaredDiff).ok()) {
    std::swap(sq_diff_op, mul2_op);
  }
  auto mul1_inputs = mul1_op->inputs;
  if (mul1_inputs.size() != 2) {
    return absl::NotFoundError("LayerNormalization not suitable.");
  } else {
    const ir::IrOp* mul1_first_parent = ir_model.FindProducer(mul1_inputs[0]);
    const ir::IrOp* mul1_second_parent = ir_model.FindProducer(mul1_inputs[1]);
    auto mean0_inputs = mean0_op->inputs;
    const ir::IrOp* mean0_parent = ir_model.FindProducer(mean0_inputs[0]);
    if (!(mul1_first_parent == mean0_parent ||
          mul1_second_parent == mean0_parent)) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
    if (!(mul1_first_parent == mul0_op || mul1_second_parent == mul0_op)) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
  }
  const ir::IrOp* sub_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *mul2_op, OperationType::kSub, &sub_op));
  Tensor<Linear, DataType::kFloat32> sub_linear_value;
  ABSL_RETURN_IF_ERROR(GetElementwiseLinearValue(sub_op, &sub_linear_value));

  const ir::IrOp* add1_op;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *sub_op, OperationType::kAdd, &add1_op));
  {
    const ir::IrOp* add1_copy_op;
    ABSL_RETURN_IF_ERROR(GetNextSingleOp(ir_model, *mul1_op,
                                         OperationType::kAdd, &add1_copy_op));
    if (add1_copy_op != add1_op) {
      return absl::NotFoundError("LayerNormalization not suitable.");
    }
  }

  auto input = mean0_op->inputs[0];
  auto output_id = add1_op->outputs[0];
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto gpu_op =
      std::make_unique<MeanStdDevNormalization>(CreateMeanStdDevNormalization(
          op_def, gpu_info, ir_model.tensor(input)->desc.GetBHWCShape(),
          add_value, mul_linear_value, sub_linear_value,
          /*two_step=*/false));
  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(input)}),
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(output_id)}),
      std::move(gpu_op), "layer_normalization");

  new_consumed_ops->insert(mean0_op->id);
  new_consumed_ops->insert(sq_diff_op->id);
  new_consumed_ops->insert(mean1_op->id);
  new_consumed_ops->insert(add0_op->id);
  new_consumed_ops->insert(rsqrt_op->id);
  new_consumed_ops->insert(mul0_op->id);
  new_consumed_ops->insert(mul1_op->id);
  new_consumed_ops->insert(mul2_op->id);
  new_consumed_ops->insert(sub_op->id);
  new_consumed_ops->insert(add1_op->id);

  return absl::OkStatus();
}

}  // namespace ml_drift
