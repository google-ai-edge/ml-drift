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
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/transformations/fuse_add_to_conv.h"
#include "ml_drift/common/transformations/fuse_mul_to_conv.h"

namespace ml_drift::ir {

namespace {

// Returns true if op was removed, and ~absl::OkStatus() if model is invalid
absl::StatusOr<bool> TryRemoveNoop(IrModel* ir_model, const IrOp* op) {
  if (op->name == ToString(::ml_drift::OperationType::CONCAT) &&
      op->inputs.size() == 1) {
    ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
    return true;
  }

  if ((op->name == ToString(::ml_drift::OperationType::RESHAPE) ||
       op->name == ToString(::ml_drift::OperationType::RESIZE)) &&
      op->inputs.size() == 1 && op->outputs.size() == 1) {
    const auto* input_tensor = ir_model->tensor(op->inputs[0]);
    const auto* output_tensor = ir_model->tensor(op->outputs[0]);
    if (input_tensor && output_tensor &&
        input_tensor->desc.GetBHWDCShape() ==
            output_tensor->desc.GetBHWDCShape()) {
      ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
      return true;
    }
  }

  if (op->name == ToString(::ml_drift::OperationType::ADD) &&
      op->inputs.size() == 1 && op->outputs.size() == 1 &&
      op->attr.type() == typeid(::ml_drift::ElementwiseAttributes)) {
    const auto& attr =
        std::any_cast<const ::ml_drift::ElementwiseAttributes&>(op->attr);
    if (std::holds_alternative<std::monostate>(attr.param)) {
      ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
      return true;
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
      ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
      return true;
    }
  }

  return false;
}

absl::StatusOr<bool> TryFusePad(IrModel* ir_model, const IrOp* op) {
  if (op->name != ToString(OperationType::PAD) ||
      op->attr.type() != typeid(PadAttributes)) {
    return false;
  }
  // Copy PadAttributes because removing the op will invalidate references to it
  PadAttributes pad_attr = std::any_cast<PadAttributes>(op->attr);

  if (pad_attr.type != PaddingContentType::ZEROS) {
    return false;
  }

  if (op->outputs.size() != 1) return false;
  const IrTensor* output_tensor = ir_model->tensor(op->outputs[0]);
  if (!output_tensor || output_tensor->consumers.size() != 1) return false;

  const IrOp* consumer_op = ir_model->op(*output_tensor->consumers.begin());
  if (!consumer_op) return false;

  // Pad -> Add
  if (consumer_op->name == ToString(OperationType::ADD) &&
      consumer_op->attr.type() == typeid(ElementwiseAttributes)) {
    if (pad_attr.prepended != BHWC(0, 0, 0, 0) || pad_attr.appended.h != 0 ||
        pad_attr.appended.w != 0 || pad_attr.appended.b != 0) {
      return false;
    }
    const IrTensor* input_tensor = ir_model->tensor(op->inputs[0]);
    if (!input_tensor || input_tensor->desc.GetBHWCShape().c % 4 != 0) {
      return false;
    }
    const auto& add_attr =
        std::any_cast<const ElementwiseAttributes&>(consumer_op->attr);
    if (!std::holds_alternative<std::monostate>(add_attr.param)) {
      return false;
    }
    ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
    return true;
  }

  if (consumer_op->inputs.size() != 1) return false;

  if (pad_attr.appended.c != 0 || pad_attr.prepended.c != 0 ||
      pad_attr.appended.b != 0 || pad_attr.prepended.b != 0) {
    return false;  // For other fusions, only HW padding is supported.
  }

  if (consumer_op->name == ToString(OperationType::CONVOLUTION_2D)) {
    if (auto* attr = ir_model->GetMutableAttr<Convolution2DAttributes>(
            consumer_op->id)) {
      ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
      attr->padding.appended.h += pad_attr.appended.h;
      attr->padding.appended.w += pad_attr.appended.w;
      attr->padding.prepended.h += pad_attr.prepended.h;
      attr->padding.prepended.w += pad_attr.prepended.w;
      return true;
    }
  } else if (consumer_op->name ==
             ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
    if (auto* attr = ir_model->GetMutableAttr<DepthwiseConvolution2DAttributes>(
            consumer_op->id)) {
      ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
      attr->padding.appended.h += pad_attr.appended.h;
      attr->padding.appended.w += pad_attr.appended.w;
      attr->padding.prepended.h += pad_attr.prepended.h;
      attr->padding.prepended.w += pad_attr.prepended.w;
      return true;
    }
  } else if (consumer_op->name == ToString(OperationType::POOLING_2D)) {
    if (auto* attr =
            ir_model->GetMutableAttr<Pooling2DAttributes>(consumer_op->id)) {
      ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(op->id));
      attr->padding.appended.h += pad_attr.appended.h;
      attr->padding.appended.w += pad_attr.appended.w;
      attr->padding.prepended.h += pad_attr.prepended.h;
      attr->padding.prepended.w += pad_attr.prepended.w;
      return true;
    }
  }

  return false;
}

template <typename AttrType>
absl::StatusOr<bool> TryAbsorbProducer(IrModel* ir_model, const IrOp* gemm_op,
                                       AttrType* attr) {
  if (gemm_op->inputs.size() != 1) return false;
  const IrTensor* input_tensor = ir_model->tensor(gemm_op->inputs[0]);
  if (!input_tensor || !input_tensor->producer.has_value() ||
      input_tensor->consumers.size() != 1) {
    return false;
  }
  const IrOp* producer = ir_model->op(input_tensor->producer.value());
  if (!producer || producer->inputs.size() != 1) return false;

  // Try to merge with ADD
  if (producer->name == ToString(OperationType::ADD) &&
      producer->attr.type() == typeid(ElementwiseAttributes)) {
    auto add_attr = std::any_cast<ElementwiseAttributes>(producer->attr);
    if (!std::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            add_attr.param) &&
        !HoldsFloatScalar(add_attr.param)) {
      return false;
    }
    if constexpr (std::is_same_v<AttrType, Convolution2DAttributes>) {
      if (attr->groups == 1 && attr->padding.appended.w == 0 &&
          attr->padding.appended.h == 0 && attr->padding.prepended.w == 0 &&
          attr->padding.prepended.h == 0) {
        ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(producer->id));
        ::ml_drift::FuseAddWithConvolution2D(add_attr, attr);
        return true;
      }
    }
    // Try to merge with MUL
  } else if (producer->name == ToString(OperationType::MUL) &&
             producer->attr.type() == typeid(ElementwiseAttributes)) {
    auto mul_attr = std::any_cast<ElementwiseAttributes>(producer->attr);
    if (!std::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            mul_attr.param) &&
        !HoldsFloatScalar(mul_attr.param)) {
      return false;
    }
    ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(producer->id));

    if constexpr (std::is_same_v<AttrType, Convolution2DAttributes>) {
      ::ml_drift::FuseMultiplyWithConvolution2D(mul_attr, attr);
    } else if constexpr (std::is_same_v<AttrType,
                                        ConvolutionTransposedAttributes>) {
      ::ml_drift::FuseMultiplyWithConvolutionTransposed(mul_attr, attr);
    } else if constexpr (std::is_same_v<AttrType,
                                        DepthwiseConvolution2DAttributes>) {
      ::ml_drift::FuseMultiplyWithDepthwiseConvolution2D(mul_attr, attr);
    } else if constexpr (std::is_same_v<AttrType, FullyConnectedAttributes>) {
      ::ml_drift::FuseMultiplyWithFullyConnected(mul_attr, attr);
    }
    return true;
  }

  return false;
}

template <typename AttrType>
absl::StatusOr<bool> TryAbsorbConsumer(IrModel* ir_model, const IrOp* gemm_op,
                                       AttrType* attr) {
  if (gemm_op->outputs.size() != 1) return false;

  const IrTensor* output_tensor = ir_model->tensor(gemm_op->outputs[0]);
  if (!output_tensor || output_tensor->consumers.size() != 1) return false;

  const IrOp* consumer = ir_model->op(*output_tensor->consumers.begin());
  if (!consumer || consumer->inputs.size() != 1) return false;

  // Try to merge with ADD
  if (consumer->name == ToString(OperationType::ADD) &&
      consumer->attr.type() == typeid(ElementwiseAttributes)) {
    auto add_attr = std::any_cast<ElementwiseAttributes>(consumer->attr);
    if (!std::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            add_attr.param) &&
        !HoldsFloatScalar(add_attr.param)) {
      return false;
    }
    ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(consumer->id));

    if constexpr (std::is_same_v<AttrType, Convolution2DAttributes>) {
      ::ml_drift::FuseConvolution2DWithAdd(add_attr, attr);
    } else if constexpr (std::is_same_v<AttrType,
                                        ConvolutionTransposedAttributes>) {
      ::ml_drift::FuseConvolutionTransposedWithAdd(add_attr, attr);
    } else if constexpr (std::is_same_v<AttrType,
                                        DepthwiseConvolution2DAttributes>) {
      ::ml_drift::FuseDepthwiseConvolution2DWithAdd(add_attr, attr);
    } else if constexpr (std::is_same_v<AttrType, FullyConnectedAttributes>) {
      ::ml_drift::FuseFullyConnectedWithAdd(add_attr, attr);
    }
    return true;
    // Try to merge with MUL
  } else if (consumer->name == ToString(OperationType::MUL) &&
             consumer->attr.type() == typeid(ElementwiseAttributes)) {
    auto mul_attr = std::any_cast<ElementwiseAttributes>(consumer->attr);
    if (!std::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            mul_attr.param) &&
        !HoldsFloatScalar(mul_attr.param)) {
      return false;
    }
    ABSL_RETURN_IF_ERROR(ir_model->RemoveSimpleOp(consumer->id));

    if constexpr (std::is_same_v<AttrType, Convolution2DAttributes>) {
      ::ml_drift::FuseConvolution2DWithMultiply(mul_attr, attr);
    } else if constexpr (std::is_same_v<AttrType,
                                        ConvolutionTransposedAttributes>) {
      ::ml_drift::FuseConvolutionTransposedWithMultiply(mul_attr, attr);
    } else if constexpr (std::is_same_v<AttrType,
                                        DepthwiseConvolution2DAttributes>) {
      ::ml_drift::FuseDepthwiseConvolution2DWithMultiply(mul_attr, attr);
    } else if constexpr (std::is_same_v<AttrType, FullyConnectedAttributes>) {
      ::ml_drift::FuseFullyConnectedWithMultiply(mul_attr, attr);
    }
    return true;
  }

  return false;
}

template <typename AttrType>
absl::StatusOr<bool> TryAbsorbElementwise(IrModel* ir_model,
                                          const IrOp* gemm_op, AttrType* attr) {
  absl::StatusOr<bool> producer_result =
      TryAbsorbProducer(ir_model, gemm_op, attr);
  if (!producer_result.ok() || *producer_result) return producer_result;

  absl::StatusOr<bool> consumer_result =
      TryAbsorbConsumer(ir_model, gemm_op, attr);
  return consumer_result;
}

absl::StatusOr<bool> TryFuseIntoGemm(IrModel* ir_model, const IrOp* op) {
  if (op->name == ToString(OperationType::CONVOLUTION_2D)) {
    if (auto* attr =
            ir_model->GetMutableAttr<Convolution2DAttributes>(op->id)) {
      return TryAbsorbElementwise(ir_model, op, attr);
    }
  } else if (op->name == ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
    if (auto* attr = ir_model->GetMutableAttr<DepthwiseConvolution2DAttributes>(
            op->id)) {
      return TryAbsorbElementwise(ir_model, op, attr);
    }
  } else if (op->name == ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
    if (auto* attr =
            ir_model->GetMutableAttr<ConvolutionTransposedAttributes>(op->id)) {
      return TryAbsorbElementwise(ir_model, op, attr);
    }
  } else if (op->name == ToString(OperationType::FULLY_CONNECTED)) {
    if (auto* attr =
            ir_model->GetMutableAttr<FullyConnectedAttributes>(op->id)) {
      return TryAbsorbElementwise(ir_model, op, attr);
    }
  }
  return false;
}

absl::StatusOr<bool> TryAddQuantAdjustments(IrModel* model, const IrOp* op) {
  if (op->name == ToString(OperationType::QUANTIZE_AND_DEQUANTIZE)) {
    return false;
  }

  for (IrTensorId output_id : op->outputs) {
    const IrTensor* output_tensor = model->tensor(output_id);
    if (!output_tensor || !output_tensor->quant_params.has_value()) {
      continue;
    }
    if (output_tensor->consumers.empty()) {
      continue;
    }

    IrOp* qdq_op = model->add_op();
    qdq_op->name = ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
    QuantizeAndDequantizeAttributes attr;
    attr.min = output_tensor->quant_params.value().min;
    attr.max = output_tensor->quant_params.value().max;
    attr.scale = output_tensor->quant_params.value().scale;
    qdq_op->attr = attr;

    IrTensor* adjusted_tensor = model->add_tensor(output_tensor->desc);
    adjusted_tensor->quant_params = output_tensor->quant_params;

    std::vector<IrOpId> original_consumers;
    original_consumers.reserve(output_tensor->consumers.size());
    for (IrOpId consumer_id : output_tensor->consumers) {
      original_consumers.push_back(consumer_id);
    }

    model->AddConsumer(output_id, qdq_op->id);
    model->SetProducer(adjusted_tensor->id, qdq_op->id);

    for (IrOpId consumer_id : original_consumers) {
      ABSL_RETURN_IF_ERROR(
          model->ReplaceInput(consumer_id, output_id, adjusted_tensor->id));
    }

    model->ResetQuantParams(output_id);
    return true;
  }
  return false;
}

}  // namespace

absl::Status TransformIrModel(::ml_drift::ir::IrModel* ir_model) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < ir_model->ops().size(); ++i) {
      // Ordering here does not matter for graph structure, and has only
      // nominal effect on performance.
      const IrOp* op = ir_model->op(i);
      if (!op) continue;
      ABSL_ASSIGN_OR_RETURN(const bool removed_noop,
                            TryRemoveNoop(ir_model, op));
      if (removed_noop) {
        changed = true;
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(const bool fused_pad, TryFusePad(ir_model, op));
      if (fused_pad) {
        changed = true;
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(const bool fused_into_gemm,
                            TryFuseIntoGemm(ir_model, op));
      if (fused_into_gemm) {
        changed = true;
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(const bool added_quant_adjustments,
                            TryAddQuantAdjustments(ir_model, op));
      if (added_quant_adjustments) {
        changed = true;
        continue;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace ml_drift::ir
