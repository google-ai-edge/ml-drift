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

#include "ml_drift/common/kernels/reduce_parser.h"

#include <any>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
absl::Status TryAddThenReduce(const GpuInfo& gpu_info,
                              const GraphFloat32& graph, NodeId first_node_id,
                              const std::set<NodeId>& consumed_nodes,
                              std::set<NodeId>* new_consumed_nodes,
                              GpuModelBuilder* model_builder) {
  auto* add_node = graph.GetNode(first_node_id);
  if (add_node == nullptr ||
      OperationTypeFromString(add_node->operation.type) !=
          OperationType::kAdd) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto add_inputs = graph.FindInputs(add_node->id);
  if (add_inputs.size() != 2 ||
      add_inputs[0]->tensor.shape != add_inputs[1]->tensor.shape) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto add_output_id = graph.FindOutputs(add_node->id)[0]->id;
  auto consumers = graph.FindConsumers(add_output_id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto* reduce_node = consumers[0];
  if (reduce_node == nullptr) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto reduce_op_type = OperationTypeFromString(reduce_node->operation.type);
  std::set<Axis> axis_to_reduce;
  if (reduce_op_type == OperationType::kMean ||
      reduce_op_type == OperationType::kReduceMaximum ||
      reduce_op_type == OperationType::kReduceMinimum ||
      reduce_op_type == OperationType::kReduceProduct ||
      reduce_op_type == OperationType::kReduceSum) {
    auto attr =
        std::any_cast<ReduceAttributes>(reduce_node->operation.attributes);
    axis_to_reduce = attr.dims;
  } else {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto reduce_outputs = graph.FindOutputs(reduce_node->id);

  ABSL_ASSIGN_OR_RETURN(auto src0_handle,
                        model_builder->GetTensor(add_inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto src1_handle,
                        model_builder->GetTensor(add_inputs[1]->id));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle,
                        model_builder->GetTensor(reduce_outputs[0]->id));
  OperationDef op_def;
  op_def.src_tensors.push_back(src0_handle.tensor_desc);
  op_def.src_tensors.push_back(src1_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  const std::string op_name =
      absl::StrCat("reduce (2 input add) ", add_node->id, " ", reduce_node->id);
  model_builder->AddGpuOperation(
      std::vector<ValueId>({add_inputs[0]->id, add_inputs[1]->id}),
      std::vector<ValueId>({reduce_outputs[0]->id}),
      std::make_unique<Reduce>(
          Create2InputReduce(axis_to_reduce, add_inputs[0]->tensor.shape,
                             reduce_op_type, op_def, gpu_info)),
      op_name);

  new_consumed_nodes->insert(add_node->id);
  new_consumed_nodes->insert(reduce_node->id);
  return absl::OkStatus();
}

absl::Status TryAddThenReduce(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  auto* add_op = ir_model.op(first_op_id);
  if (add_op == nullptr ||
      OperationTypeFromString(add_op->name) != OperationType::kAdd) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  const auto& add_inputs = add_op->inputs;
  if (add_inputs.size() != 2 ||
      ir_model.tensor(add_inputs[0])->desc.GetBHWCShape() !=
          ir_model.tensor(add_inputs[1])->desc.GetBHWCShape()) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto add_output_id = add_op->outputs[0];
  auto consumers = ir_model.FindConsumers(add_output_id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto* reduce_op = consumers[0];
  if (reduce_op == nullptr) {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto reduce_op_type = OperationTypeFromString(reduce_op->name);
  std::set<Axis> axis_to_reduce;
  if (reduce_op_type == OperationType::kMean ||
      reduce_op_type == OperationType::kReduceMaximum ||
      reduce_op_type == OperationType::kReduceMinimum ||
      reduce_op_type == OperationType::kReduceProduct ||
      reduce_op_type == OperationType::kReduceSum) {
    auto attr = std::any_cast<ReduceAttributes>(reduce_op->attr);
    axis_to_reduce = attr.dims;
  } else {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  const auto& reduce_outputs = reduce_op->outputs;

  ABSL_ASSIGN_OR_RETURN(auto src0_handle,
                        model_builder->GetTensor(add_inputs[0]));
  ABSL_ASSIGN_OR_RETURN(auto src1_handle,
                        model_builder->GetTensor(add_inputs[1]));
  ABSL_ASSIGN_OR_RETURN(auto dst_handle,
                        model_builder->GetTensor(reduce_outputs[0]));
  OperationDef op_def;
  op_def.src_tensors.push_back(src0_handle.tensor_desc);
  op_def.src_tensors.push_back(src1_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  const std::string op_name =
      absl::StrCat("reduce (2 input add) ", add_op->id, " ", reduce_op->id);
  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(add_inputs[0]),
           static_cast<GpuModelBuilder::ValueId>(add_inputs[1])}),
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(reduce_outputs[0])}),
      std::make_unique<Reduce>(Create2InputReduce(
          axis_to_reduce, ir_model.tensor(add_inputs[0])->desc.GetBHWCShape(),
          reduce_op_type, op_def, gpu_info)),
      op_name);

  new_consumed_ops->insert(add_op->id);
  new_consumed_ops->insert(reduce_op->id);
  return absl::OkStatus();
}

}  // namespace ml_drift
