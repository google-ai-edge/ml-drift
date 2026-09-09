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

#include "ml_drift/common/ir_model.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift::ir {

std::string IrOp::str() const {
  return absl::StrCat("o#", id, " (", name, ")");
}

// TODO: who/impjdi - Move shape extraction to TensorDesc::str().
std::string IrTensor::str() const {
  const auto shape = desc.GetBHWDCShape();
  return absl::StrCat("t#", id, " (", shape.b, "x", shape.h, "x", shape.w, "x",
                      shape.d, "x", shape.c, ")");
}

IrTensor* IrModel::add_tensor(const ::ml_drift::TensorDescriptor& desc) {
  IrTensorId id = tensors_.size();
  tensors_.push_back(std::make_unique<IrTensor>(id));
  tensors_[id]->desc = desc;
  return tensors_[id].get();
}

void IrModel::SetProducer(IrTensorId tensor_id, IrOpId op_id) {
  ABSL_QCHECK_LT(tensor_id, tensors_.size());
  tensors_[tensor_id]->producer = op_id;
  ABSL_QCHECK_LT(op_id, ops_.size());
  ops_[op_id]->outputs.push_back(tensor_id);
}

void IrModel::AddConsumer(IrTensorId tensor_id, IrOpId op_id) {
  ABSL_QCHECK_LT(tensor_id, tensors_.size());
  tensors_[tensor_id]->consumers.insert(op_id);
  ABSL_QCHECK_LT(op_id, ops_.size());
  ops_[op_id]->inputs.push_back(tensor_id);
}

IrOp* IrModel::FindProducer(IrTensorId tensor_id) const {
  if (tensor_id >= tensors_.size()) {
    return nullptr;
  }
  if (!tensors_[tensor_id]->producer.has_value()) {
    return nullptr;
  }
  IrOpId op_id = tensors_[tensor_id]->producer.value();
  if (op_id >= ops_.size()) {
    return nullptr;
  }
  return ops_[op_id].get();
}

std::vector<IrOp*> IrModel::FindConsumers(IrTensorId tensor_id) const {
  std::vector<IrOp*> consumers;
  consumers.reserve(tensors_[tensor_id]->consumers.size());
  if (tensor_id < tensors_.size() && tensors_[tensor_id] != nullptr) {
    for (IrOpId op_id : tensors_[tensor_id]->consumers) {
      if (op_id < ops_.size() && ops_[op_id] != nullptr) {
        consumers.push_back(ops_[op_id].get());
      }
    }
  }
  return consumers;
}

bool IrModel::IsGraphInput(IrTensorId tensor_id) const {
  return absl::c_linear_search(inputs_, tensor_id);
}

bool IrModel::IsGraphOutput(IrTensorId tensor_id) const {
  return absl::c_linear_search(outputs_, tensor_id);
}

absl::Status IrModel::ReplaceInput(IrOpId op_id, IrTensorId old_tensor_id,
                                   IrTensorId new_tensor_id) {
  if (op_id >= ops_.size() || ops_[op_id] == nullptr) {
    return absl::InvalidArgumentError("Invalid op ID");
  }
  if (old_tensor_id >= tensors_.size() || tensors_[old_tensor_id] == nullptr) {
    return absl::InvalidArgumentError("Invalid old tensor ID");
  }
  if (new_tensor_id >= tensors_.size() || tensors_[new_tensor_id] == nullptr) {
    return absl::InvalidArgumentError("Invalid new tensor ID");
  }

  IrOp* op = ops_[op_id].get();
  bool replaced = false;
  for (auto& input_id : op->inputs) {
    if (input_id == old_tensor_id) {
      input_id = new_tensor_id;
      replaced = true;
    }
  }

  if (replaced) {
    tensors_[old_tensor_id]->consumers.erase(op_id);
    tensors_[new_tensor_id]->consumers.insert(op_id);
    return absl::OkStatus();
  }

  return absl::NotFoundError("Old tensor not found in op's inputs");
}

void IrModel::ResetQuantParams(IrTensorId tensor_id) {
  if (tensor_id < tensors_.size() && tensors_[tensor_id] != nullptr) {
    tensors_[tensor_id]->quant_params.reset();
  }
}

absl::Status IrModel::RemoveSimpleOp(IrOpId op_id) {
  if (op_id >= ops_.size() || ops_[op_id] == nullptr) {
    return absl::InvalidArgumentError("Invalid op ID");
  }
  IrOp* op = ops_[op_id].get();
  if (op->inputs.size() != 1 || op->outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "Op must have exactly 1 input and 1 output");
  }

  IrTensor* input_tensor = tensors_[op->inputs[0]].get();
  IrTensor* output_tensor = tensors_[op->outputs[0]].get();

  const bool keep_output = IsGraphOutput(output_tensor->id);
  const bool keep_input = IsGraphInput(input_tensor->id);

  if (keep_input && keep_output) {
    return absl::InvalidArgumentError(
        "Cannot remove op when both input and output are graph boundaries");
  }

  if (keep_output) {
    IrOp* producer_op = FindProducer(input_tensor->id);
    if (!producer_op) {
      return absl::InvalidArgumentError("Input tensor has no producer");
    }

    for (auto& output_id : producer_op->outputs) {
      if (output_id == input_tensor->id) {
        output_id = output_tensor->id;
      }
    }
    output_tensor->producer = producer_op->id;

    for (IrOpId consumer_id : input_tensor->consumers) {
      if (consumer_id == op->id) continue;
      if (consumer_id >= ops_.size() || ops_[consumer_id] == nullptr) continue;
      IrOp* consumer_op = ops_[consumer_id].get();
      for (auto& consumer_input_id : consumer_op->inputs) {
        if (consumer_input_id == input_tensor->id) {
          consumer_input_id = output_tensor->id;
        }
      }
      output_tensor->consumers.insert(consumer_id);
    }

    input_tensor->consumers.erase(op->id);
    tensors_[input_tensor->id].reset();
  } else {
    for (IrOpId consumer_id : output_tensor->consumers) {
      if (consumer_id >= ops_.size() || ops_[consumer_id] == nullptr) continue;
      IrOp* consumer_op = ops_[consumer_id].get();
      for (auto& consumer_input_id : consumer_op->inputs) {
        if (consumer_input_id == output_tensor->id) {
          consumer_input_id = input_tensor->id;
        }
      }
      input_tensor->consumers.insert(consumer_id);
    }

    input_tensor->consumers.erase(op->id);
    tensors_[output_tensor->id].reset();
  }

  ops_[op->id].reset();
  return absl::OkStatus();
}

absl::Status IrModel::RemoveOp(IrOpId op_id) {
  if (op_id >= ops_.size() || ops_[op_id] == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat("Invalid op ID: ", op_id));
  }

  IrOp* op = ops_[op_id].get();
  absl::flat_hash_set<IrTensorId> affected_tensors;
  for (IrTensorId in_id : op->inputs) {
    affected_tensors.insert(in_id);
    if (in_id < tensors_.size() && tensors_[in_id] != nullptr) {
      tensors_[in_id]->consumers.erase(op_id);
    }
  }
  for (IrTensorId out_id : op->outputs) {
    affected_tensors.insert(out_id);
    if (out_id < tensors_.size() && tensors_[out_id] != nullptr) {
      if (tensors_[out_id]->producer == op_id) {
        tensors_[out_id]->producer.reset();
      }
    }
  }
  ops_[op_id].reset();

  for (IrTensorId tensor_id : affected_tensors) {
    if (tensor_id >= tensors_.size() || tensors_[tensor_id] == nullptr) {
      continue;
    }
    IrTensor* t = tensors_[tensor_id].get();
    if (!t->producer.has_value() && t->consumers.empty()) {
      inputs_.erase(std::remove(inputs_.begin(), inputs_.end(), tensor_id),
                    inputs_.end());
      outputs_.erase(std::remove(outputs_.begin(), outputs_.end(), tensor_id),
                     outputs_.end());
      tensors_[tensor_id].reset();
    } else if (t->consumers.empty() && t->producer.has_value() &&
               !IsGraphOutput(tensor_id)) {
      add_output(tensor_id);
    } else if (!t->consumers.empty() && !t->producer.has_value() &&
               !IsGraphInput(tensor_id)) {
      add_input(tensor_id);
    }
  }

  return absl::OkStatus();
}

}  // namespace ml_drift::ir
