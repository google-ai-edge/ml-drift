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

#include "ml_drift/common/kernels/mish.h"

#include <any>  // IWYU pragma: keep
#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

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

}  // namespace

absl::Status TryMish(const GpuInfo& gpu_info, const GraphFloat32& graph,
                     NodeId first_node_id,
                     const std::set<NodeId>& consumed_nodes,
                     std::set<NodeId>* new_consumed_nodes,
                     GpuModelBuilder* model_builder) {
  Node* exp_node = graph.GetNode(first_node_id);
  ABSL_RETURN_IF_ERROR(CheckIfValidNodeOfType(exp_node, OperationType::EXP));
  ValueId input = 0;
  if (std::vector<Value*> exp_inputs = graph.FindInputs(exp_node->id);
      exp_inputs.size() == 1) {
    input = exp_inputs[0]->id;
  } else {
    return absl::NotFoundError("Mish not suitable.");
  }

  Node* increment_node = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *exp_node, OperationType::ADD, &increment_node));
  if (std::vector<Value*> increment_inputs =
          graph.FindInputs(increment_node->id);
      increment_inputs.size() != 1) {
    // This is supposed to be addition with a scalar
    return absl::NotFoundError("Mish not suitable.");
  } else {
    // Check that this is an increment by 1.
    float value = 0.f;
    ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(increment_node, &value));
    if (value != 1.f) {
      return absl::NotFoundError("Mish not suitable.");
    }
  }
  Node* log_node = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *increment_node, OperationType::LOG, &log_node));
  Node* tanh_node = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *log_node, OperationType::TANH, &tanh_node));
  Node* mul_node = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleNode(graph, *tanh_node, OperationType::MUL, &mul_node));
  // Make sure that the other input to mul is the original input.
  if (std::vector<Value*> mul_inputs = graph.FindInputs(mul_node->id);
      mul_inputs.size() != 2) {
    return absl::NotFoundError("Mish not suitable.");
  } else {
    if (!(mul_inputs[0]->id == input || mul_inputs[1]->id == input)) {
      return absl::NotFoundError("Mish not suitable.");
    }
  }

  ValueId input_id = graph.FindInputs(exp_node->id)[0]->id;
  ValueId output_id = graph.FindOutputs(mul_node->id)[0]->id;
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input_id));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));
  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  model_builder->AddGpuOperation(
      std::vector<ValueId>({input_id}), std::vector<ValueId>({output_id}),
      std::make_unique<GPUOperation>(
          CreateElementwiseOneInput(gpu_info, op_def, OperationType::MISH)),
      "mish");

  new_consumed_nodes->insert(exp_node->id);
  new_consumed_nodes->insert(increment_node->id);
  new_consumed_nodes->insert(log_node->id);
  new_consumed_nodes->insert(tanh_node->id);
  new_consumed_nodes->insert(mul_node->id);

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

absl::Status GetNextSingleOp(const ir::IrModel& ir_model, const ir::IrOp& op,
                             OperationType next_type,
                             const ir::IrOp** next_op) {
  if (op.outputs.empty()) return absl::NotFoundError("No outputs.");
  auto consumers = ir_model.FindConsumers(op.outputs[0]);
  if (consumers.size() != 1) {
    return absl::NotFoundError("Not a single consumer.");
  }
  const ir::IrOp* consumer_op = consumers[0];
  ABSL_RETURN_IF_ERROR(CheckIfValidOpOfType(consumer_op, next_type));
  *next_op = consumer_op;
  return absl::OkStatus();
}

}  // namespace

absl::Status TryMish(const GpuInfo& gpu_info, const ir::IrModel& ir_model,
                     ir::IrOpId first_op_id,
                     const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
                     absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
                     GpuModelBuilder* model_builder) {
  const ir::IrOp* exp_op = ir_model.op(first_op_id);
  ABSL_RETURN_IF_ERROR(CheckIfValidOpOfType(exp_op, OperationType::EXP));
  GpuModelBuilder::ValueId input = 0;
  if (const auto& exp_inputs = exp_op->inputs; exp_inputs.size() == 1) {
    input = static_cast<GpuModelBuilder::ValueId>(exp_inputs[0]);
  } else {
    return absl::NotFoundError("Mish not suitable.");
  }

  const ir::IrOp* increment_op = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *exp_op, OperationType::ADD, &increment_op));
  if (const auto& increment_inputs = increment_op->inputs;
      increment_inputs.size() != 1) {
    // This is supposed to be addition with a scalar
    return absl::NotFoundError("Mish not suitable.");
  } else {
    // Check that this is an increment by 1.
    float value = 0.f;
    ABSL_RETURN_IF_ERROR(GetElementwiseScalarValue(increment_op, &value));
    if (value != 1.f) {
      return absl::NotFoundError("Mish not suitable.");
    }
  }
  const ir::IrOp* log_op = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *increment_op, OperationType::LOG, &log_op));
  const ir::IrOp* tanh_op = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *log_op, OperationType::TANH, &tanh_op));
  const ir::IrOp* mul_op = nullptr;
  ABSL_RETURN_IF_ERROR(
      GetNextSingleOp(ir_model, *tanh_op, OperationType::MUL, &mul_op));
  // Make sure that the other input to mul is the original input.
  if (const auto& mul_inputs = mul_op->inputs; mul_inputs.size() != 2) {
    return absl::NotFoundError("Mish not suitable.");
  } else {
    if (!(mul_inputs[0] == input || mul_inputs[1] == input)) {
      return absl::NotFoundError("Mish not suitable.");
    }
  }

  GpuModelBuilder::ValueId input_id =
      static_cast<GpuModelBuilder::ValueId>(exp_op->inputs[0]);
  GpuModelBuilder::ValueId output_id =
      static_cast<GpuModelBuilder::ValueId>(mul_op->outputs[0]);
  ABSL_ASSIGN_OR_RETURN(auto src_handle, model_builder->GetTensor(input_id));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(output_id));
  OperationDef op_def;
  op_def.src_tensors.push_back(src_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>({input_id}),
      std::vector<GpuModelBuilder::ValueId>({output_id}),
      std::make_unique<GPUOperation>(
          CreateElementwiseOneInput(gpu_info, op_def, OperationType::MISH)),
      "mish");

  new_consumed_ops->insert(exp_op->id);
  new_consumed_ops->insert(increment_op->id);
  new_consumed_ops->insert(log_op->id);
  new_consumed_ops->insert(tanh_op->id);
  new_consumed_ops->insert(mul_op->id);

  return absl::OkStatus();
}

}  // namespace ml_drift
