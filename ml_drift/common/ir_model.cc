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

#include "ml_drift/common/ir_model.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
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

bool IrModel::IsGraphInput(IrTensorId tensor_id) const {
  return absl::c_linear_search(inputs_, tensor_id);
}

bool IrModel::IsGraphOutput(IrTensorId tensor_id) const {
  return absl::c_linear_search(outputs_, tensor_id);
}

}  // namespace ml_drift::ir
