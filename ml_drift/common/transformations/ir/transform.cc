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

#include "ml_drift/common/transformations/ir/transform.h"

#include <any>
#include <cstddef>
#include <variant>

#include "absl/status/status.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"

namespace ml_drift::ir {

namespace {

bool TryRemoveNoop(IrModel* ir_model, const IrOp* op) {
  if (op->name == ToString(::ml_drift::OperationType::CONCAT) &&
      op->inputs.size() == 1) {
    return ir_model->RemoveSimpleOp(op->id).ok();
  }

  if ((op->name == ToString(::ml_drift::OperationType::RESHAPE) ||
       op->name == ToString(::ml_drift::OperationType::RESIZE)) &&
      op->inputs.size() == 1 && op->outputs.size() == 1) {
    const auto* input_tensor = ir_model->tensor(op->inputs[0]);
    const auto* output_tensor = ir_model->tensor(op->outputs[0]);
    if (input_tensor && output_tensor &&
        input_tensor->desc.GetBHWDCShape() ==
            output_tensor->desc.GetBHWDCShape()) {
      return ir_model->RemoveSimpleOp(op->id).ok();
    }
  }

  if (op->name == ToString(::ml_drift::OperationType::ADD) &&
      op->inputs.size() == 1 && op->outputs.size() == 1 &&
      op->attr.type() == typeid(::ml_drift::ElementwiseAttributes)) {
    const auto& attr =
        std::any_cast<const ::ml_drift::ElementwiseAttributes&>(op->attr);
    if (std::holds_alternative<std::monostate>(attr.param)) {
      return ir_model->RemoveSimpleOp(op->id).ok();
    }
  }

  if (op->name == ToString(::ml_drift::OperationType::SLICE) &&
      op->inputs.size() == 1 && op->outputs.size() == 1 &&
      op->attr.type() == typeid(::ml_drift::SliceAttributes)) {
    const auto& attr =
        std::any_cast<const ::ml_drift::SliceAttributes&>(op->attr);
    const auto* input_tensor = ir_model->tensor(op->inputs[0]);
    const auto* output_tensor = ir_model->tensor(op->outputs[0]);
    if (input_tensor && output_tensor &&
        input_tensor->desc.GetBHWDCShape() ==
            output_tensor->desc.GetBHWDCShape() &&
        attr.starts == ::ml_drift::BHWC(0, 0, 0, 0) &&
        attr.strides == ::ml_drift::BHWC(1, 1, 1, 1) &&
        attr.ends == output_tensor->desc.GetBHWCShape()) {
      return ir_model->RemoveSimpleOp(op->id).ok();
    }
  }

  return false;
}

}  // namespace

absl::Status TransformIrModel(::ml_drift::ir::IrModel* ir_model) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < ir_model->ops().size(); ++i) {
      const IrOp* op = ir_model->op(i);
      if (!op) continue;

      if (TryRemoveNoop(ir_model, op)) {
        changed = true;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace ml_drift::ir
