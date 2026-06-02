// Copyright 2025 The ML Drift Authors.
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

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
absl::Status TryAddThenReduce(const GpuInfo& gpu_info,
                              const GraphFloat32& graph, NodeId first_node_id,
                              const std::set<NodeId>& consumed_nodes,
                              std::set<NodeId>* new_consumed_nodes,
                              GpuModelBuilder* model_builder) {
  auto* add_node = graph.GetNode(first_node_id);
  if (add_node == nullptr ||
      OperationTypeFromString(add_node->operation.type) != OperationType::ADD) {
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
  if (reduce_op_type == OperationType::MEAN ||
      reduce_op_type == OperationType::REDUCE_MAXIMUM ||
      reduce_op_type == OperationType::REDUCE_MINIMUM ||
      reduce_op_type == OperationType::REDUCE_PRODUCT ||
      reduce_op_type == OperationType::REDUCE_SUM) {
    auto attr =
        std::any_cast<ReduceAttributes>(reduce_node->operation.attributes);
    axis_to_reduce = attr.dims;
  } else {
    return absl::NotFoundError("AddThenReduce not suitable.");
  }
  auto reduce_outputs = graph.FindOutputs(reduce_node->id);

  ASSIGN_OR_RETURN(auto src0_handle,
                   model_builder->GetTensor(add_inputs[0]->id));
  ASSIGN_OR_RETURN(auto src1_handle,
                   model_builder->GetTensor(add_inputs[1]->id));
  ASSIGN_OR_RETURN(auto dst_handle,
                   model_builder->GetTensor(reduce_outputs[0]->id));
  OperationDef op_def;
  op_def.src_tensors.push_back(src0_handle.tensor_desc);
  op_def.src_tensors.push_back(src1_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  const std::string fused_nodes =
      std::to_string(add_node->id) + " " + std::to_string(reduce_node->id);
  const std::string op_name = "reduce (2 input add) " + fused_nodes;
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

}  // namespace ml_drift
