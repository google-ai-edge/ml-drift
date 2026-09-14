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

#include "ml_drift/common/selectors/operation_selector.h"

#include <any>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/mean_stddev_normalization.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/selectors/default_selector.h"
#include "ml_drift/common/selectors/simple_selectors.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

absl::Status MakeScaledDotProductAttention(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const std::vector<ValueId>& inputs, const std::vector<ValueId>& outputs,
    const ScaledDotProductAttentionAttributes& sdpa_attr,
    GpuModelBuilder* model_builder) {
  ABSL_ASSIGN_OR_RETURN(auto q, model_builder->GetTensor(inputs[0]));
  ABSL_ASSIGN_OR_RETURN(auto k, model_builder->GetTensor(inputs[1]));
  ABSL_ASSIGN_OR_RETURN(auto v, model_builder->GetTensor(inputs[2]));
  auto q_transposed = model_builder->Transpose(q, BHWC(0, 2, 1, 3));
  auto k_transposed = model_builder->Transpose(k, BHWC(0, 2, 3, 1));
  auto v_transposed = model_builder->Transpose(v, BHWC(0, 2, 1, 3));
  auto q_scaled =
      sdpa_attr.scale.has_value()
          ? model_builder->Multiplication(q_transposed, sdpa_attr.scale.value())
          : q_transposed;

  const int n = q.tensor_desc.GetBHWCShape().b;
  if (n > 1) {
    q_scaled = model_builder->Reshape(
        q_scaled, BHWC{1, q_scaled.tensor_desc.GetBHWCShape().h * n,
                       q_scaled.tensor_desc.GetBHWCShape().w,
                       q_scaled.tensor_desc.GetBHWCShape().c});
    k_transposed = model_builder->Reshape(
        k_transposed, BHWC{1, k_transposed.tensor_desc.GetBHWCShape().h * n,
                           k_transposed.tensor_desc.GetBHWCShape().w,
                           k_transposed.tensor_desc.GetBHWCShape().c});
    v_transposed = model_builder->Reshape(
        v_transposed, BHWC{1, v_transposed.tensor_desc.GetBHWCShape().h * n,
                           v_transposed.tensor_desc.GetBHWCShape().w,
                           v_transposed.tensor_desc.GetBHWCShape().c});
  }

  GpuModelBuilder::TensorHandle out;
  if (inputs.size() == 3) {
    ABSL_ASSIGN_OR_RETURN(out, model_builder->BatchedMatMulSoftmaxBatchedMatMul(
                                   q_scaled, k_transposed, v_transposed));
  } else {
    ABSL_ASSIGN_OR_RETURN(auto mask, model_builder->GetTensor(inputs[3]));
    // Note that transpose is not necessary for mask since it should already be
    // shaped accordingly.
    if (n > 1) {
      mask = model_builder->Reshape(
          mask, BHWC{1, mask.tensor_desc.GetBHWCShape().h * n,
                     mask.tensor_desc.GetBHWCShape().w,
                     mask.tensor_desc.GetBHWCShape().c});
    }
    ABSL_ASSIGN_OR_RETURN(out,
                          model_builder->BatchedMatMulSoftmaxBatchedMatMul(
                              q_scaled, k_transposed, v_transposed, &mask));
  }
  out =
      model_builder->Reshape(out, BHWC{n, out.tensor_desc.GetBHWCShape().h / n,
                                       out.tensor_desc.GetBHWCShape().w,
                                       out.tensor_desc.GetBHWCShape().c});
  return model_builder->UpdateOutputTensor(
      model_builder->Transpose(out, BHWC(0, 2, 1, 3)), outputs[0]);
}

absl::Status MakeRmsNorm(const GpuInfo& gpu_info,
                         const CreateGpuModelInfo& create_info,
                         const std::vector<ValueId>& inputs,
                         const std::vector<ValueId>& outputs,
                         const RmsNormAttributes& attr,
                         GpuModelBuilder* model_builder) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "RmsNorm operation expects a single input and a single output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto src_tensor, model_builder->GetTensor(inputs[0]));
  Tensor<Linear, DataType::FLOAT32> gamma, beta;
  Tensor<Linear, DataType::FLOAT32>* gamma_ptr = nullptr;
  Tensor<Linear, DataType::FLOAT32>* beta_ptr = nullptr;
  if (attr.scale.has_value()) {
    gamma.shape = Linear(attr.scale.value().data.size());
    gamma.data = attr.scale.value().data;
    gamma_ptr = &gamma;
  }
  if (attr.bias.has_value()) {
    beta.shape = Linear(attr.bias.value().data.size());
    beta.data = attr.bias.value().data;
    beta_ptr = &beta;
  }

  return model_builder->UpdateOutputTensor(
      model_builder->RMSNormalization(src_tensor, attr.epsilon, gamma_ptr,
                                      beta_ptr),
      outputs[0]);
}

absl::Status MakePositionalEmbedding(const GpuInfo& gpu_info,
                                     const CreateGpuModelInfo& create_info,
                                     const std::vector<Value*>& inputs,
                                     const std::vector<Value*>& outputs,
                                     GpuModelBuilder* model_builder) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "PositionalEmbedding operation expects 2 inputs and a single "
        "output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto position, model_builder->GetTensor(inputs[1]->id));

  return model_builder->UpdateOutputTensor(
      model_builder->PositionalEmbedding(src, position), outputs[0]->id);
}

absl::Status MakeFullyConnectedExternalWeights(
    const std::vector<ValueId>& inputs, const std::vector<ValueId>& outputs,
    const FullyConnectedAttributes& attr, GpuModelBuilder* model_builder) {
  if (inputs.size() < 2 && inputs.size() > 5) {
    return absl::InvalidArgumentError(
        "Expected 2, 3, 4, or 5 inputs for FullyConnectedExternalWeights.");
  }

  ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]));
  ABSL_ASSIGN_OR_RETURN(auto weights, model_builder->GetTensor(inputs[1]));
  GpuModelBuilder::TensorHandle bias;
  GpuModelBuilder::TensorHandle* bias_ptr = nullptr;
  GpuModelBuilder::TensorHandle src_exp;
  GpuModelBuilder::TensorHandle* src_exp_ptr = nullptr;
  GpuModelBuilder::TensorHandle runtime_check_tensor;
  GpuModelBuilder::TensorHandle* runtime_check_tensor_ptr = nullptr;
  auto& runtime_check_attr = attr.external_weights->runtime_check;
  if (runtime_check_attr.src_end_ch_index.has_value() ||
      runtime_check_attr.dst_end_ch_index.has_value()) {
    if (inputs.size() < 3) {
      return absl::InvalidArgumentError("Missing runtime check tensor.");
    }
    if (inputs.size() > 3) {
      ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]));
      bias_ptr = &bias;
    }
    if (inputs.size() > 4) {
      ABSL_ASSIGN_OR_RETURN(src_exp, model_builder->GetTensor(inputs[3]));
      src_exp_ptr = &src_exp;
    }
    ABSL_ASSIGN_OR_RETURN(runtime_check_tensor,
                          model_builder->GetTensor(inputs[inputs.size() - 1]));
    runtime_check_tensor_ptr = &runtime_check_tensor;
  } else {
    if (inputs.size() > 4) {
      return absl::InvalidArgumentError("Unexpected runtime check tensor.");
    }
    if (inputs.size() > 2) {
      ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]));
      bias_ptr = &bias;
    }
    if (inputs.size() > 3) {
      ABSL_ASSIGN_OR_RETURN(src_exp, model_builder->GetTensor(inputs[3]));
      src_exp_ptr = &src_exp;
    }
  }

  const OHWI& weights_shape = attr.weights.shape;
  const WeightsDescription& weights_desc = attr.external_weights->desc;
  ConvRuntimeCheckDesc runtime_check = {
      .src_end_ch_index = runtime_check_attr.src_end_ch_index,
      .dst_end_ch_index = runtime_check_attr.dst_end_ch_index,
  };

  const GpuModelBuilder::Weights external_weights =
      CreateExternalWeights(weights, weights_desc, weights_shape);
  ABSL_ASSIGN_OR_RETURN(auto output,
                        model_builder->FullyConnectedExternalWeights(
                            src, external_weights, bias_ptr, src_exp_ptr,
                            runtime_check, runtime_check_tensor_ptr));
  return model_builder->UpdateOutputTensor(output, outputs[0]);
}

absl::Status MakeQuantizedFullyConnectedExternalWeights(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const OperationDef& op_def, OperationType op_type,
    const std::vector<ValueId>& inputs, const std::vector<ValueId>& outputs,
    GpuModelBuilder* model_builder, const OHWI& weights_shape_ohwi,
    const OHWI& scale_zp_shape) {
  if (op_type != OperationType::FULLY_CONNECTED_INT2 &&
      op_type != OperationType::FULLY_CONNECTED_INT4 &&
      op_type != OperationType::FULLY_CONNECTED_INT8) {
    return absl::InternalError(
        "Expected only int2, int4, or int8 Fully Connected.");
  }

  ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]));
  ABSL_ASSIGN_OR_RETURN(auto weights, model_builder->GetTensor(inputs[1]));
  ABSL_ASSIGN_OR_RETURN(auto scale, model_builder->GetTensor(inputs[2]));
  ABSL_ASSIGN_OR_RETURN(auto zero_point, model_builder->GetTensor(inputs[3]));

  GpuModelBuilder::TensorHandle weights_sum_i;
  GpuModelBuilder::TensorHandle* weights_sum_i_ptr = nullptr;
  GpuModelBuilder::TensorHandle bias;
  GpuModelBuilder::TensorHandle* bias_th_ptr = nullptr;
  if (inputs.size() == 5) {
    // weight_sum_i is expected to be int32 (zero_point expected to be float).
    ABSL_ASSIGN_OR_RETURN(auto fifth_tensor,
                          model_builder->GetTensor(inputs[4]));
    const bool has_weight_sum_i =
        fifth_tensor.tensor_desc.GetDataType() == DataType::INT32;
    if (has_weight_sum_i) {
      weights_sum_i = fifth_tensor;
      weights_sum_i_ptr = &weights_sum_i;
    } else {
      ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]));
      ABSL_ASSIGN_OR_RETURN(scale, model_builder->GetTensor(inputs[3]));
      zero_point = fifth_tensor;
      bias_th_ptr = &bias;
    }
  } else if (inputs.size() == 6) {
    ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]));
    ABSL_ASSIGN_OR_RETURN(scale, model_builder->GetTensor(inputs[3]));
    ABSL_ASSIGN_OR_RETURN(zero_point, model_builder->GetTensor(inputs[4]));
    ABSL_ASSIGN_OR_RETURN(weights_sum_i, model_builder->GetTensor(inputs[5]));
    bias_th_ptr = &bias;
    weights_sum_i_ptr = &weights_sum_i;
  }

  WeightsDescription weights_desc;
  GpuModelBuilder::TensorHandle out;
  switch (op_type) {
    case OperationType::FULLY_CONNECTED_INT8: {
      weights_desc = GetFullyConnectedInt8WeightsDesc(
          gpu_info, weights_shape_ohwi,
          /*prefer_textures=*/
          create_info.hints.Check(ModelHints::kPreferTextureWeights));
      const GpuModelBuilder::Weights external_weights = CreateExternalWeights(
          weights, weights_desc, weights_shape_ohwi, scale_zp_shape, &scale,
          &zero_point, weights_sum_i_ptr);
      out = model_builder->FullyConnectedInt8ExternalWeights(
          src, external_weights, bias_th_ptr);
      break;
    }
    case OperationType::FULLY_CONNECTED_INT4: {
      weights_desc = GetFullyConnectedInt4WeightsDesc(
          gpu_info, weights_shape_ohwi,
          /*prefer_textures=*/
          create_info.hints.Check(ModelHints::kPreferTextureWeights));
      const GpuModelBuilder::Weights external_weights = CreateExternalWeights(
          weights, weights_desc, weights_shape_ohwi, scale_zp_shape, &scale,
          &zero_point, weights_sum_i_ptr);
      out = model_builder->FullyConnectedInt4ExternalWeights(
          src, external_weights, bias_th_ptr);
      break;
    }
    case OperationType::FULLY_CONNECTED_INT2: {
      weights_desc = GetFullyConnectedInt2WeightsDesc(
          gpu_info, weights_shape_ohwi,
          /*prefer_textures=*/
          create_info.hints.Check(ModelHints::kPreferTextureWeights));
      const GpuModelBuilder::Weights external_weights = CreateExternalWeights(
          weights, weights_desc, weights_shape_ohwi, scale_zp_shape, &scale,
          &zero_point, weights_sum_i_ptr);
      out = model_builder->FullyConnectedInt2ExternalWeights(
          src, external_weights, bias_th_ptr);
      break;
    }
    default:
      return absl::InternalError("Unsupported operation type.");
  }
  return model_builder->UpdateOutputTensor(out, outputs[0]);
}

absl::Status MakeLayerNorm(const GpuInfo& gpu_info,
                           const CreateGpuModelInfo& create_info,
                           const LayerNormAttributes& attr,
                           const std::vector<ValueId>& inputs,
                           const std::vector<ValueId>& outputs,
                           GpuModelBuilder* model_builder) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "LayerNorm operation expects a single input and a single output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto src_tensor, model_builder->GetTensor(inputs[0]));
  auto input_shape = src_tensor.tensor_desc.GetBHWCShape();

  Tensor<Linear, DataType::FLOAT32> gamma, beta;
  if (attr.scale.has_value()) {
    gamma.shape = Linear(attr.scale.value().data.size());
    gamma.data = attr.scale.value().data;
  } else {
    // Tensor::shape and Tensor::data are independent members; assigning the
    // shape leaves data empty, so writing through data[i] indexes an empty
    // vector.
    gamma.shape = Linear(input_shape.c);
    gamma.data.assign(input_shape.c, 1.0f);
  }
  if (attr.bias.has_value()) {
    beta.shape = Linear(attr.bias.value().data.size());
    beta.data = attr.bias.value().data;
  } else {
    // Tensor::shape and Tensor::data are independent members; assigning the
    // shape leaves data empty, so writing through data[i] indexes an empty
    // vector.
    beta.shape = Linear(input_shape.c);
    beta.data.assign(input_shape.c, 0.0f);
  }

  return model_builder->UpdateOutputTensor(
      model_builder->LayerNormalization(src_tensor, gamma, beta, attr.epsilon),
      outputs[0]);
}

absl::Status MakeGroupNorm(const GpuInfo& gpu_info,
                           const CreateGpuModelInfo& create_info,
                           const GroupNormAttributes& attr,
                           const std::vector<ValueId>& inputs,
                           const std::vector<ValueId>& outputs,
                           GpuModelBuilder* model_builder) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "GroupNorm operation expects a single input and a single output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto src_tensor, model_builder->GetTensor(inputs[0]));
  auto input_shape = src_tensor.tensor_desc.GetBHWCShape();
  Tensor<Linear, DataType::FLOAT32> gamma, beta;
  if (attr.gamma.has_value()) {
    gamma.shape = Linear(attr.gamma.value().data.size());
    gamma.data = attr.gamma.value().data;
  } else {
    // Tensor::shape and Tensor::data are independent members; assigning the
    // shape leaves data empty, so writing through data[i] indexes an empty
    // vector.
    gamma.shape = Linear(input_shape.c);
    gamma.data.assign(input_shape.c, 1.0f);
  }
  if (attr.beta.has_value()) {
    beta.shape = Linear(attr.beta.value().data.size());
    beta.data = attr.beta.value().data;
  } else {
    // Tensor::shape and Tensor::data are independent members; assigning the
    // shape leaves data empty, so writing through data[i] indexes an empty
    // vector.
    beta.shape = Linear(input_shape.c);
    beta.data.assign(input_shape.c, 0.0f);
  }
  return model_builder->UpdateOutputTensor(
      model_builder->HWCGroupNorm(src_tensor, attr.groups, attr.epsilon, gamma,
                                  beta),
      outputs[0]);
}

inline bool IsEmbeddingLookupQuantized(const EmbeddingLookupAttributes& attr) {
  const auto weights_type = attr.weights_type;
  return weights_type == EmbeddingLookupAttributes::WeightsType::kInt2 ||
         weights_type == EmbeddingLookupAttributes::WeightsType::kInt4 ||
         weights_type == EmbeddingLookupAttributes::WeightsType::kInt8;
}

template <typename T>
absl::Status MakeQuantizedEmbeddingLookup(const GpuInfo& gpu_info,
                                          const CreateGpuModelInfo& create_info,
                                          const std::vector<T*>& inputs,
                                          const std::vector<T*>& outputs,
                                          const EmbeddingLookupAttributes& attr,
                                          GpuModelBuilder* model_builder) {
  static_assert(std::is_same_v<std::decay_t<T>, Value> ||
                std::is_same_v<std::decay_t<T>, ir::IrTensor>,
                "T must be either Value or ir::IrTensor.");
  EmbeddingLookupAttributes::WeightsType weights_type = attr.weights_type;
  if (!IsEmbeddingLookupQuantized(attr)) {
    return absl::InternalError(
        "Expected only int2, int4 or int8 Embedding Lookup.");
  } else if (inputs.size() < 3 || inputs.size() > 4 || outputs.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "EmbeddingLookup operation expects 3 or 4 inputs and 1 output. Got: ",
        inputs.size(), " inputs and ", outputs.size(), " outputs."));
  }
  ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto weights, model_builder->GetTensor(inputs[1]->id));

  WeightsDescription weights_desc;
  if (weights_type == EmbeddingLookupAttributes::WeightsType::kInt2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(
        gpu_info, attr.original_weights_shape,
        create_info.hints.Check(ModelHints::kPreferTextureWeights));
  } else if (weights_type == EmbeddingLookupAttributes::WeightsType::kInt4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(
        gpu_info, attr.original_weights_shape,
        create_info.hints.Check(ModelHints::kPreferTextureWeights));
  } else {
    weights_desc = GetFullyConnectedInt8WeightsDesc(
        gpu_info, attr.original_weights_shape,
        create_info.hints.Check(ModelHints::kPreferTextureWeights));
  }

  const DataType dst_type = DeduceDataTypeFromPrecision(create_info.precision);
  ABSL_ASSIGN_OR_RETURN(auto scale, model_builder->GetTensor(inputs[2]->id));
  GpuModelBuilder::TensorHandle* zero_point_ptr = nullptr;
  GpuModelBuilder::TensorHandle zero_point;
  if (inputs.size() > 3) {
    ABSL_ASSIGN_OR_RETURN(zero_point, model_builder->GetTensor(inputs[3]->id));
    zero_point_ptr = &zero_point;
  }

  const GpuModelBuilder::Weights external_weights =
      CreateExternalWeights(weights, weights_desc, attr.original_weights_shape,
                            attr.scale_zp_shape, &scale, zero_point_ptr);
  return model_builder->UpdateOutputTensor(
      model_builder->EmbeddingLookup(src, external_weights, dst_type),
      outputs[0]->id);
}

absl::Status MakeRoPE(const GpuInfo& gpu_info,
                      const CreateGpuModelInfo& create_info,
                      const std::vector<ValueId>& inputs,
                      const std::vector<ValueId>& outputs,
                      const RoPEAttributes& attr,
                      GpuModelBuilder* model_builder) {
  if (inputs.size() == 2) {
    if (outputs.size() != 1) {
      return absl::InvalidArgumentError("RoPE expects 1 output for 2 inputs.");
    }
    if (attr.kernel_type == RoPEKernelType::INTERLEAVED_2D) {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]));
      auto shape = src.tensor_desc.GetBHWCShape();
      if (shape.c % 8 != 0) {
        // SplitRoPEConcat falls back to RoPE for c % 8 != 0, and RoPE only
        // supports PLANAR_1D.
        return absl::InvalidArgumentError(
            "RoPE with INTERLEAVED_2D requires channels divisible by 8.");
      }
      if (shape.w != shape.h) {
        // The pos tensor is assumed to be a 1D tensor with length W. The Y
        // coord is read from that as well, so we must have W == H.
        return absl::InvalidArgumentError(
            "RoPE with INTERLEAVED_2D requires square input.");
      }
    }
  } else if (inputs.size() == 3) {
    if (outputs.size() != 2) {
      return absl::InvalidArgumentError("RoPE expects 2 outputs for 3 inputs.");
    }
    if (attr.kernel_type == RoPEKernelType::INTERLEAVED_2D) {
      // RoPE (3 inputs case) only supports PLANAR_1D so far.
      return absl::InvalidArgumentError(
          "RoPE does not support INTERLEAVED_2D for 3 inputs.");
    }
  } else {
    return absl::InvalidArgumentError("RoPE expects 2 or 3 inputs.");
  }

  GpuModel gpu_model;
  const bool with_split_concat = inputs.size() == 2;
  if (with_split_concat) {
    ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]));
    ABSL_ASSIGN_OR_RETURN(auto position, model_builder->GetTensor(inputs[1]));

    return model_builder->UpdateOutputTensor(
        model_builder->SplitRoPEConcat(src, position, attr), outputs[0]);
  } else {
    ABSL_ASSIGN_OR_RETURN(auto src_l, model_builder->GetTensor(inputs[0]));
    ABSL_ASSIGN_OR_RETURN(auto src_r, model_builder->GetTensor(inputs[1]));
    ABSL_ASSIGN_OR_RETURN(auto position, model_builder->GetTensor(inputs[2]));

    return model_builder->UpdateOutputTensors(
        model_builder->RoPE(src_l, src_r, position, attr),
        {outputs[0], outputs[1]});
  }
}

absl::Status MakeRuntimeBatchedMatMul(
    const GpuModelBuilder::TensorHandle left,
    const GpuModelBuilder::TensorHandle right,
    const GpuModelBuilder::TensorHandle runtime_check_tensor,
    const ValueId output_id, const BatchedMatMulAttributes& attr,
    GpuModelBuilder* model_builder) {
  if (!attr.runtime_check.src_end_ch_index.has_value() &&
      !attr.runtime_check.dst_end_ch_index.has_value()) {
    return absl::InvalidArgumentError(
        "Runtime Batched MatMul requires runtime check information.");
  }

  ConvRuntimeCheckDesc runtime_check = {
      .src_end_ch_index = attr.runtime_check.src_end_ch_index,
      .dst_end_ch_index = attr.runtime_check.dst_end_ch_index,
  };

  ABSL_ASSIGN_OR_RETURN(auto output, model_builder->BatchedMatMul(
                                         left, right, attr, nullptr,
                                         runtime_check, &runtime_check_tensor));
  return model_builder->UpdateOutputTensor(output, output_id);
}
}  // namespace

absl::Status GPUOperationFromNode(const GpuInfo& gpu_info,
                                  const OperationDef& op_def,
                                  const CreateGpuModelInfo& create_info,
                                  const std::vector<Value*>& inputs,
                                  const std::vector<Value*>& outputs,
                                  const Node& node,
                                  GpuModelBuilder* model_builder) {
  std::vector<ValueId> src_ids(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    src_ids[i] = inputs[i]->id;
  }
  std::vector<ValueId> dst_ids(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    dst_ids[i] = outputs[i]->id;
  }
  auto op_type = OperationTypeFromString(node.operation.type);
  switch (op_type) {
    case OperationType::ABS:
    case OperationType::CEIL:
    case OperationType::COPY:
    case OperationType::COS:
    case OperationType::ELU:
    case OperationType::EXP:
    case OperationType::FLOOR:
    case OperationType::GELU:
    case OperationType::GELU_TANH_APPROX:
    case OperationType::HARD_SWISH:
    case OperationType::LOG:
    case OperationType::LOGICAL_NOT:
    case OperationType::NEG:
    case OperationType::ROUND:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIGN:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH: {
      GPUOperation operation;
      if (inputs[0]->tensor.shape != outputs[0]->tensor.shape) {
        operation = CreateElementwiseOneInputWithBroadcast(
            gpu_info, op_def, op_type, inputs[0]->tensor.shape,
            outputs[0]->tensor.shape);
      } else {
        operation = CreateElementwiseOneInput(gpu_info, op_def, op_type);
      }
      model_builder->AddGpuOperation(
          src_ids, dst_ids,
          std::make_unique<GPUOperation>(std::move(operation)),
          node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::ADD:
    case OperationType::ATAN2:
    case OperationType::DIV:
    case OperationType::EQUAL:
    case OperationType::FLOOR_DIV:
    case OperationType::FLOOR_MOD:
    case OperationType::GREATER:
    case OperationType::GREATER_EQUAL:
    case OperationType::LESS:
    case OperationType::LESS_EQUAL:
    case OperationType::LOGICAL_AND:
    case OperationType::LOGICAL_OR:
    case OperationType::LOGICAL_XOR:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::MISH:
    case OperationType::MOD:
    case OperationType::MUL:
    case OperationType::NOT_EQUAL:
    case OperationType::POW:
    case OperationType::REMAINDER:
    case OperationType::SHIFT_LEFT:
    case OperationType::SHIFT_RIGHT:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      if (op_type == OperationType::ADD && inputs.size() >= 2) {
        const bool first_input_pad_with_zero =
            inputs[0]->tensor.shape.c % 4 == 0 &&
            inputs[0]->tensor.shape.c != outputs[0]->tensor.shape.c;
        const bool second_input_pad_with_zero =
            inputs[1]->tensor.shape.c % 4 == 0 &&
            inputs[1]->tensor.shape.c != outputs[0]->tensor.shape.c;
        if (inputs.size() >= 3 || first_input_pad_with_zero ||
            second_input_pad_with_zero) {
          auto output = outputs[0];
          std::vector<int> channels(inputs.size());
          for (int i = 0; i < inputs.size(); ++i) {
            channels[i] = inputs[i]->tensor.shape.c;
          }
          std::unique_ptr<GPUOperation> gpu_op;
          SelectAdd(op_def, channels, output->tensor.shape.c, &gpu_op);
          model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                         node.operation.type);
          return absl::OkStatus();
        }
      }

      GPUOperation operation;
      if (inputs.size() == 2) {
        ElementwiseAttributes attr;
        if (node.operation.attributes.has_value()) {
          attr = std::any_cast<const ElementwiseAttributes&>(
              node.operation.attributes);
        }
        if (inputs[0]->tensor.shape != outputs[0]->tensor.shape) {
          operation = CreateElementwiseTwoInputWithBroadcast(
              gpu_info, op_def, op_type, inputs[0]->tensor.shape,
              inputs[1]->tensor.shape, outputs[0]->tensor.shape, attr);
        } else {
          operation = CreateElementwiseTwoInput(gpu_info, op_def, op_type,
                                                inputs[1]->tensor.shape,
                                                outputs[0]->tensor.shape);
        }
      } else if (inputs.size() == 1 && node.operation.attributes.has_value()) {
        const auto& attr = std::any_cast<const ElementwiseAttributes&>(
            node.operation.attributes);
        if (std::holds_alternative<std::monostate>(attr.param)) {
          return absl::InternalError("Received unsupported tensor.");
        }
        if (inputs[0]->tensor.shape != outputs[0]->tensor.shape) {
          operation = CreateElementwiseWithBroadcast(
              gpu_info, op_def, op_type, attr, inputs[0]->tensor.shape,
              outputs[0]->tensor.shape);
        } else {
          operation = CreateElementwise(gpu_info, op_def, op_type, attr);
        }
      } else {
        return absl::UnimplementedError(absl::StrCat(
            "No support of ", node.operation.type, " with this parameters"));
      }
      model_builder->AddGpuOperation(
          src_ids, dst_ids,
          std::make_unique<GPUOperation>(std::move(operation)),
          node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::MEAN:
    case OperationType::REDUCE_ALL:
    case OperationType::REDUCE_ANY:
    case OperationType::REDUCE_MAXIMUM:
    case OperationType::REDUCE_MINIMUM:
    case OperationType::REDUCE_PRODUCT:
    case OperationType::REDUCE_SUM: {
      const auto& attr =
          std::any_cast<const ReduceAttributes&>(node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->Reduce(src, op_type, attr.dims), outputs[0]->id);
    }
    case OperationType::BATCHED_MATMUL: {
      const auto& batched_mat_mul_attr =
          std::any_cast<const BatchedMatMulAttributes&>(
              node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto left, model_builder->GetTensor(inputs[0]->id));
      ABSL_ASSIGN_OR_RETURN(auto right,
                            model_builder->GetTensor(inputs[1]->id));
      if (inputs.size() == 3) {
        ABSL_ASSIGN_OR_RETURN(auto runtime_check_tensor,
                              model_builder->GetTensor(inputs[2]->id));
        return MakeRuntimeBatchedMatMul(left, right, runtime_check_tensor,
                                        outputs[0]->id, batched_mat_mul_attr,
                                        model_builder);
      }
      ABSL_ASSIGN_OR_RETURN(
          auto output,
          model_builder->BatchedMatMul(left, right, batched_mat_mul_attr));
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::BITCAST: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->BitCast(src, op_def.dst_tensors[0].GetDataType()),
          outputs[0]->id);
    }
    case OperationType::CAST: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->Cast(src, op_def.dst_tensors[0].GetDataType()),
          outputs[0]->id);
    }
    case OperationType::CONCAT: {
      const auto& attr =
          std::any_cast<const ConcatAttributes&>(node.operation.attributes);
      std::vector<GpuModelBuilder::TensorHandle> src_handles(inputs.size());
      for (int i = 0; i < inputs.size(); ++i) {
        ABSL_ASSIGN_OR_RETURN(src_handles[i],
                              model_builder->GetTensor(inputs[i]->id));
      }
      return model_builder->UpdateOutputTensor(
          model_builder->Concat(src_handles, attr.axis), outputs[0]->id);
    }
    case OperationType::CONVOLUTION_2D: {
      const auto& attr = std::any_cast<const Convolution2DAttributes&>(
          node.operation.attributes);
      if (inputs.size() == 1) {
        ABSL_ASSIGN_OR_RETURN(auto src,
                              model_builder->GetTensor(inputs[0]->id));
        if (gpu_info.IsApiWebGpu() &&
            op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 &&
            op_def.dst_tensors[0].GetDataType() == DataType::FLOAT16) {
          // see CheckExternalTensorDescription in
          // ml_drift/common/gpu_model_util.cc
          src = model_builder->Cast(src, DataType::FLOAT16);
        }
        return model_builder->UpdateOutputTensor(
            model_builder->Convolution(src, attr), outputs[0]->id);
      } else {
        // CONVOLUTION_2D with runtime weights
        ABSL_ASSIGN_OR_RETURN(auto src,
                              model_builder->GetTensor(inputs[0]->id));
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        GpuModelBuilder::TensorHandle bias;
        GpuModelBuilder::TensorHandle* bias_ptr = nullptr;
        if (!attr.bias.data.empty()) {
          DataType bias_type = src.tensor_desc.GetDataType();
          bias = model_builder->AddConstantTensor(attr.bias, bias_type);
          bias_ptr = &bias;
        }
        ABSL_ASSIGN_OR_RETURN(auto output, model_builder->Convolution(
                                               src, weights, bias_ptr, attr));
        return model_builder->UpdateOutputTensor(output, outputs[0]->id);
      }
    }
    case OperationType::CONVOLUTION_TRANSPOSED: {
      const auto& attr = std::any_cast<const ConvolutionTransposedAttributes&>(
          node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      if (inputs.size() == 1) {
        return model_builder->UpdateOutputTensor(
            model_builder->ConvolutionTransposed(src, attr), outputs[0]->id);
      } else {
        // CONVOLUTION_TRANSPOSED with runtime weights
        auto attr_copy = std::any_cast<ConvolutionTransposedAttributes>(
            node.operation.attributes);
        const OHWI weights_shape =
            OHWI(inputs[1]->tensor.shape.b, inputs[1]->tensor.shape.h,
                 inputs[1]->tensor.shape.w, inputs[1]->tensor.shape.c);
        attr_copy.weights.shape = weights_shape;
        if (attr_copy.bias.data.empty()) {
          attr_copy.bias.shape = Linear(weights_shape.o);
          attr_copy.bias.data.resize(weights_shape.o, 0.0f);
        }
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        return model_builder->UpdateOutputTensor(
            model_builder->ConvolutionTransposed(src, weights, attr_copy),
            outputs[0]->id);
      }
    }
    case OperationType::CUMSUM: {
      const auto& attr =
          std::any_cast<const CumsumAttributes&>(node.operation.attributes);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectCumsum(op_def, attr, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::DEPTHWISE_CONVOLUTION: {
      const auto& attr = std::any_cast<const DepthwiseConvolution2DAttributes&>(
          node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      GpuModelBuilder::TensorHandle output;
      if (inputs.size() == 1) {
        output = model_builder->DepthwiseConvolution(src, attr);
      } else if (inputs.size() == 2) {
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        ABSL_ASSIGN_OR_RETURN(
            output, model_builder->DepthwiseConvolution(src, weights, attr));
      } else {
        return absl::InternalError(
            "Depthwise convolution only supports 1 or 2 inputs.");
      }
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::DEPTH_TO_SPACE: {
      const auto& attr = std::any_cast<const SpaceToDepthAttributes&>(
          node.operation.attributes);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectDepthToSpace(attr, op_def, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::DYNAMIC_UPDATE_SLICE: {
      auto gpu_op = SelectDynamicUpdateSlice(op_def, gpu_info);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::EMBEDDING_LOOKUP: {
      const auto& attr = std::any_cast<const EmbeddingLookupAttributes&>(
          node.operation.attributes);
      if (IsEmbeddingLookupQuantized(attr) &&
          (inputs.size() == 3 || inputs.size() == 4)) {
        return MakeQuantizedEmbeddingLookup(gpu_info, create_info, inputs,
                                            outputs, attr, model_builder);
      }
      std::unique_ptr<GPUOperation> gpu_op;
      SelectEmbeddingLookup(attr, op_def, gpu_info, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::FULLY_CONNECTED: {
      const auto& attr = std::any_cast<const FullyConnectedAttributes&>(
          node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      if (attr.external_weights.has_value()) {
        return MakeFullyConnectedExternalWeights(src_ids, dst_ids, attr,
                                                 model_builder);
      } else if (inputs.size() >= 2) {
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        GpuModelBuilder::TensorHandle bias;
        GpuModelBuilder::TensorHandle* bias_ptr = nullptr;
        if (inputs.size() == 3) {
          ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]->id));
          bias_ptr = &bias;
        }
        Convolution2DAttributes conv_attr;
        conv_attr.strides = HW(1, 1);
        conv_attr.dilations = HW(1, 1);
        conv_attr.padding.appended = HW(0, 0);
        conv_attr.padding.prepended = HW(0, 0);
        auto& conv_weights =
            conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
        auto weights_shape = inputs[1]->tensor.shape;
        conv_weights.shape = OHWI(weights_shape.b, weights_shape.h,
                                  weights_shape.w, weights_shape.c);
        ABSL_ASSIGN_OR_RETURN(
            auto output,
            model_builder->Convolution(src, weights, bias_ptr, conv_attr));
        return model_builder->UpdateOutputTensor(output, outputs[0]->id);
      } else {
        return model_builder->UpdateOutputTensor(
            model_builder->FullyConnected(src, attr), outputs[0]->id);
      }
    }
    case OperationType::FULLY_CONNECTED_INT2: {
      const auto& attr = std::any_cast<const FullyConnectedInt2Attributes&>(
          node.operation.attributes);
      if (inputs.size() >= 4) {
        return MakeQuantizedFullyConnectedExternalWeights(
            gpu_info, create_info, op_def, OperationType::FULLY_CONNECTED_INT2,
            src_ids, dst_ids, model_builder,
            std::visit([](const auto& w) { return w.shape; }, attr.weights),
            attr.scale.shape);
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->FullyConnected(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::FULLY_CONNECTED_INT4: {
      const auto& attr = std::any_cast<const FullyConnectedInt4Attributes&>(
          node.operation.attributes);
      if (inputs.size() >= 4) {
        return MakeQuantizedFullyConnectedExternalWeights(
            gpu_info, create_info, op_def, OperationType::FULLY_CONNECTED_INT4,
            src_ids, dst_ids, model_builder,
            std::visit([](const auto& w) { return w.shape; }, attr.weights),
            attr.scale.shape);
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->FullyConnected(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::FULLY_CONNECTED_INT8: {
      const auto& attr = std::any_cast<const FullyConnectedInt8Attributes&>(
          node.operation.attributes);
      if (inputs.size() >= 4) {
        return MakeQuantizedFullyConnectedExternalWeights(
            gpu_info, create_info, op_def, OperationType::FULLY_CONNECTED_INT8,
            src_ids, dst_ids, model_builder, attr.weights.shape,
            attr.scale.shape);
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->FullyConnected(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::GATHER: {
      const auto& attr =
          std::any_cast<const GatherAttributes&>(node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      ABSL_ASSIGN_OR_RETURN(auto indices,
                            model_builder->GetTensor(inputs[1]->id));
      auto output = model_builder->Gather(src, indices, attr.axis);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::GROUP_NORM: {
      const auto& attr =
          std::any_cast<const GroupNormAttributes&>(node.operation.attributes);
      return MakeGroupNorm(gpu_info, create_info, attr, src_ids, dst_ids,
                           model_builder);
    }
    case OperationType::LAYER_NORM: {
      const auto& attr =
          std::any_cast<const LayerNormAttributes&>(node.operation.attributes);
      return MakeLayerNorm(gpu_info, create_info, attr, src_ids, dst_ids,
                           model_builder);
    }
    case OperationType::LSTM: {
      auto gpu_op = SelectLSTM(op_def, gpu_info);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::MAX_INDEX: {
      const auto& attr =
          std::any_cast<const MaxIndexAttributes&>(node.operation.attributes);
      auto gpu_op = std::make_unique<Reduce>(
          CreateReduce({attr.dim}, inputs[0]->tensor.shape,
                       Reduce::Type::kMaximumIndex, op_def, gpu_info));
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::MAX_UNPOOLING_2D: {
      const auto& attr =
          std::any_cast<MaxUnpooling2DAttributes>(node.operation.attributes);
      auto gpu_op = SelectMaxUnpooling(attr, gpu_info, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::MEAN_STDDEV_NORMALIZATION: {
      MeanStdDevNormalization operation = CreateMeanStdDevNormalization(
          op_def, gpu_info, inputs[0]->tensor.shape);
      auto gpu_op =
          std::make_unique<MeanStdDevNormalization>(std::move(operation));
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::ONE_HOT: {
      const auto& attr =
          std::any_cast<const OneHotAttributes&>(node.operation.attributes);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectOneHot(op_def, attr, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::PAD: {
      const auto& attr =
          std::any_cast<const PadAttributes&>(node.operation.attributes);
      auto gpu_op =
          SelectPadding(gpu_info, attr, op_def, inputs[0]->tensor.shape.c);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::POOLING_2D: {
      const auto& attr =
          std::any_cast<const Pooling2DAttributes&>(node.operation.attributes);
      auto gpu_op = SelectPooling(attr, gpu_info, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::POSITIONAL_EMBEDDING:
      return MakePositionalEmbedding(gpu_info, create_info, inputs, outputs,
                                     model_builder);
    case OperationType::PRELU: {
      const auto& attr =
          std::any_cast<const PReLUAttributes&>(node.operation.attributes);
      auto gpu_op = SelectPReLU(attr, gpu_info, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::QUANTIZE_AND_DEQUANTIZE: {
      const auto& attr = std::any_cast<const QuantizeAndDequantizeAttributes&>(
          node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->QuantizeAndDequantize(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::RELU: {
      const auto& attr =
          std::any_cast<const ReLUAttributes&>(node.operation.attributes);
      auto gpu_op = SelectReLU(attr, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::RESAMPLER: {
      auto gpu_op = SelectResampler(op_def, gpu_info);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::RESHAPE: {
      const auto& attr =
          std::any_cast<const ReshapeAttributes&>(node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->Reshape(src, attr.new_shape);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::RESIZE: {
      const auto& attr =
          std::any_cast<const Resize2DAttributes&>(node.operation.attributes);
      std::unique_ptr<GPUOperation> gpu_op;
      ABSL_RETURN_IF_ERROR(SelectResize(attr, op_def, &gpu_op));
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::REVERSE: {
      const auto& attr =
          std::any_cast<const ReverseAttributes&>(node.operation.attributes);
      auto gpu_op = SelectReverse(attr, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::RMS_NORM: {
      const auto& attr =
          std::any_cast<const RmsNormAttributes&>(node.operation.attributes);
      return MakeRmsNorm(gpu_info, create_info, src_ids, dst_ids, attr,
                         model_builder);
    }
    case OperationType::ROPE: {
      RoPEAttributes attr;
      if (node.operation.attributes.has_value()) {
        attr = std::any_cast<const RoPEAttributes&>(node.operation.attributes);
      }
      return MakeRoPE(gpu_info, create_info, src_ids, dst_ids, attr,
                      model_builder);
    }
    case OperationType::SELECT_V2: {
      const auto& attr =
          std::any_cast<const SelectV2Attributes&>(node.operation.attributes);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectSelectV2(op_def, attr, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::SCALED_DOT_PRODUCT_ATTENTION: {
      const auto& attr =
          std::any_cast<const ScaledDotProductAttentionAttributes&>(
              node.operation.attributes);
      return MakeScaledDotProductAttention(gpu_info, create_info, src_ids,
                                           dst_ids, attr, model_builder);
    }
    case OperationType::SLICE: {
      const auto& attr =
          std::any_cast<const SliceAttributes&>(node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->StridedSlice(src, attr), outputs[0]->id);
    }
    case OperationType::SOFTMAX: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(model_builder->Softmax(src),
                                               outputs[0]->id);
    }
    case OperationType::SPACE_TO_DEPTH: {
      const auto& attr = std::any_cast<const SpaceToDepthAttributes&>(
          node.operation.attributes);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectSpaceToDepth(attr, op_def, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
    }
    case OperationType::SPLIT: {
      const auto& attr =
          std::any_cast<const SplitAttributes&>(node.operation.attributes);
      std::vector<int> sizes(outputs.size());
      std::vector<ValueId> output_ids(outputs.size());
      for (int i = 0; i < outputs.size(); ++i) {
        sizes[i] = outputs[i]->tensor.shape.get(attr.axis);
        output_ids[i] = outputs[i]->id;
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensors(
          model_builder->Split(src, attr.axis, sizes), output_ids);
    }
    case OperationType::TILE: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->Tile(src, outputs[0]->tensor.shape), outputs[0]->id);
    }
    case OperationType::TOP_K: {
      const auto& attr =
          std::any_cast<const TopKAttributes&>(node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensors(
          model_builder->TopK(src, attr.k), {outputs[0]->id, outputs[1]->id});
    }
    case OperationType::TRANSPOSE: {
      const auto& attr =
          std::any_cast<const TransposeAttributes&>(node.operation.attributes);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->Transpose(src, attr.perm), outputs[0]->id);
    }
    default:
      ABSL_ASSIGN_OR_RETURN(auto gpu_op, SelectDefault(gpu_info, op_def, node));
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.operation.type);
      return absl::OkStatus();
  }
}  // NOLINT(readability/fn_size)

absl::Status GPUOperationFromNode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    const CreateGpuModelInfo& create_info,
    const std::vector<const ir::IrTensor*>& inputs,
    const std::vector<const ir::IrTensor*>& outputs, const ir::IrOp& node,
    GpuModelBuilder* model_builder) {
  std::vector<ValueId> src_ids(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    src_ids[i] = inputs[i]->id;
  }
  std::vector<ValueId> dst_ids(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    dst_ids[i] = outputs[i]->id;
  }
  auto op_type = OperationTypeFromString(node.name);
  switch (op_type) {
    case OperationType::ABS:
    case OperationType::CEIL:
    case OperationType::COPY:
    case OperationType::COS:
    case OperationType::ELU:
    case OperationType::EXP:
    case OperationType::FLOOR:
    case OperationType::GELU:
    case OperationType::GELU_TANH_APPROX:
    case OperationType::HARD_SWISH:
    case OperationType::LOG:
    case OperationType::LOGICAL_NOT:
    case OperationType::NEG:
    case OperationType::ROUND:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIGN:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH: {
      GPUOperation operation;
      if (inputs[0]->desc.GetBHWDCShape() != outputs[0]->desc.GetBHWDCShape()) {
        operation = CreateElementwiseOneInputWithBroadcast(
            gpu_info, op_def, op_type, inputs[0]->desc.GetBHWDCShape(),
            outputs[0]->desc.GetBHWDCShape());
      } else {
        operation = CreateElementwiseOneInput(gpu_info, op_def, op_type);
      }
      model_builder->AddGpuOperation(
          src_ids, dst_ids,
          std::make_unique<GPUOperation>(std::move(operation)), node.name);
      return absl::OkStatus();
    }
    case OperationType::ADD:
    case OperationType::ATAN2:
    case OperationType::DIV:
    case OperationType::EQUAL:
    case OperationType::FLOOR_DIV:
    case OperationType::FLOOR_MOD:
    case OperationType::GREATER:
    case OperationType::GREATER_EQUAL:
    case OperationType::LESS:
    case OperationType::LESS_EQUAL:
    case OperationType::LOGICAL_AND:
    case OperationType::LOGICAL_OR:
    case OperationType::LOGICAL_XOR:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::MISH:
    case OperationType::MOD:
    case OperationType::MUL:
    case OperationType::NOT_EQUAL:
    case OperationType::POW:
    case OperationType::REMAINDER:
    case OperationType::SHIFT_LEFT:
    case OperationType::SHIFT_RIGHT:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB: {
      GPUOperation operation;
      if (inputs.size() == 2) {
        ElementwiseAttributes attr;
        if (node.attr.has_value()) {
          attr = std::any_cast<const ElementwiseAttributes&>(node.attr);
        }
        if (inputs[0]->desc.GetBHWDCShape() !=
                outputs[0]->desc.GetBHWDCShape() ||
            inputs[1]->desc.GetBHWDCShape() !=
                outputs[0]->desc.GetBHWDCShape()) {
          operation = CreateElementwiseTwoInputWithBroadcast(
              gpu_info, op_def, op_type, inputs[0]->desc.GetBHWDCShape(),
              inputs[1]->desc.GetBHWDCShape(), outputs[0]->desc.GetBHWDCShape(),
              attr);
        } else {
          operation = CreateElementwiseTwoInput(
              gpu_info, op_def, op_type, inputs[1]->desc.GetBHWCShape(),
              outputs[0]->desc.GetBHWCShape());
        }
      } else if (inputs.size() == 1 && node.attr.has_value()) {
        const auto& attr =
            std::any_cast<const ElementwiseAttributes&>(node.attr);
        if (std::holds_alternative<std::monostate>(attr.param)) {
          return absl::InternalError("Received unsupported tensor.");
        }
        if (inputs[0]->desc.GetBHWDCShape() !=
            outputs[0]->desc.GetBHWDCShape()) {
          operation = CreateElementwiseWithBroadcast(
              gpu_info, op_def, op_type, attr, inputs[0]->desc.GetBHWDCShape(),
              outputs[0]->desc.GetBHWDCShape());
        } else {
          operation = CreateElementwise(gpu_info, op_def, op_type, attr);
        }
      } else {
        return absl::UnimplementedError(
            absl::StrCat("No support of ", node.name, " with this parameters"));
      }
      model_builder->AddGpuOperation(
          src_ids, dst_ids,
          std::make_unique<GPUOperation>(std::move(operation)), node.name);
      return absl::OkStatus();
    }
    case OperationType::MEAN:
    case OperationType::REDUCE_ALL:
    case OperationType::REDUCE_ANY:
    case OperationType::REDUCE_MAXIMUM:
    case OperationType::REDUCE_MINIMUM:
    case OperationType::REDUCE_PRODUCT:
    case OperationType::REDUCE_SUM: {
      const auto& attr = std::any_cast<const ReduceAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->Reduce(src, op_type, attr.dims), outputs[0]->id);
    }
    case OperationType::BATCHED_MATMUL: {
      const auto& batched_mat_mul_attr =
          std::any_cast<const BatchedMatMulAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto left, model_builder->GetTensor(inputs[0]->id));
      ABSL_ASSIGN_OR_RETURN(auto right,
                            model_builder->GetTensor(inputs[1]->id));
      if (inputs.size() == 3) {
        ABSL_ASSIGN_OR_RETURN(auto runtime_check_tensor,
                              model_builder->GetTensor(inputs[2]->id));
        return MakeRuntimeBatchedMatMul(left, right, runtime_check_tensor,
                                        outputs[0]->id, batched_mat_mul_attr,
                                        model_builder);
      }
      ABSL_ASSIGN_OR_RETURN(
          auto output,
          model_builder->BatchedMatMul(left, right, batched_mat_mul_attr));
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::BITCAST: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->BitCast(src, op_def.dst_tensors[0].GetDataType()),
          outputs[0]->id);
    }
    case OperationType::CAST: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->Cast(src, op_def.dst_tensors[0].GetDataType()),
          outputs[0]->id);
    }
    case OperationType::CONCAT: {
      const auto& attr = std::any_cast<const ConcatAttributes&>(node.attr);
      std::vector<GpuModelBuilder::TensorHandle> src_handles(inputs.size());
      for (int i = 0; i < inputs.size(); ++i) {
        ABSL_ASSIGN_OR_RETURN(src_handles[i],
                              model_builder->GetTensor(inputs[i]->id));
      }
      return model_builder->UpdateOutputTensor(
          model_builder->Concat(src_handles, attr.axis), outputs[0]->id);
    }
    case OperationType::CONVOLUTION_2D: {
      const auto& attr =
          std::any_cast<const Convolution2DAttributes&>(node.attr);
      if (inputs.size() == 1) {
        ABSL_ASSIGN_OR_RETURN(auto src,
                              model_builder->GetTensor(inputs[0]->id));
        if (gpu_info.IsApiWebGpu() &&
            op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 &&
            op_def.dst_tensors[0].GetDataType() == DataType::FLOAT16) {
          // see CheckExternalTensorDescription in
          // ml_drift/common/gpu_model_util.cc
          src = model_builder->Cast(src, DataType::FLOAT16);
        }
        return model_builder->UpdateOutputTensor(
            model_builder->Convolution(src, attr), outputs[0]->id);
      } else {
        // CONVOLUTION_2D with runtime weights
        ABSL_ASSIGN_OR_RETURN(auto src,
                              model_builder->GetTensor(inputs[0]->id));
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        GpuModelBuilder::TensorHandle bias;
        GpuModelBuilder::TensorHandle* bias_ptr = nullptr;
        if (inputs.size() == 3) {
          ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]->id));
          bias_ptr = &bias;
        } else if (!attr.bias.data.empty()) {
          DataType bias_type = src.tensor_desc.GetDataType();
          bias = model_builder->AddConstantTensor(attr.bias, bias_type);
          bias_ptr = &bias;
        }
        ABSL_ASSIGN_OR_RETURN(auto output, model_builder->Convolution(
                                               src, weights, bias_ptr, attr));
        return model_builder->UpdateOutputTensor(output, outputs[0]->id);
      }
    }
    case OperationType::CONVOLUTION_TRANSPOSED: {
      const auto& attr =
          std::any_cast<const ConvolutionTransposedAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      if (inputs.size() == 1) {
        return model_builder->UpdateOutputTensor(
            model_builder->ConvolutionTransposed(src, attr), outputs[0]->id);
      } else {
        // CONVOLUTION_TRANSPOSED with runtime weights
        auto attr_copy =
            std::any_cast<ConvolutionTransposedAttributes>(node.attr);
        const BHWC weights_bhwc_shape = inputs[1]->desc.GetBHWCShape();
        const OHWI weights_shape =
            OHWI(weights_bhwc_shape.b, weights_bhwc_shape.h,
                 weights_bhwc_shape.w, weights_bhwc_shape.c);
        attr_copy.weights.shape = weights_shape;
        if (attr_copy.bias.data.empty()) {
          attr_copy.bias.shape = Linear(weights_shape.o);
          attr_copy.bias.data.resize(weights_shape.o, 0.0f);
        }
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        return model_builder->UpdateOutputTensor(
            model_builder->ConvolutionTransposed(src, weights, attr_copy),
            outputs[0]->id);
      }
    }
    case OperationType::CUMSUM: {
      const auto& attr = std::any_cast<const CumsumAttributes&>(node.attr);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectCumsum(op_def, attr, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::DEPTHWISE_CONVOLUTION: {
      const auto& attr =
          std::any_cast<const DepthwiseConvolution2DAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      GpuModelBuilder::TensorHandle output;
      if (inputs.size() == 1) {
        output = model_builder->DepthwiseConvolution(src, attr);
      } else if (inputs.size() == 2) {
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        ABSL_ASSIGN_OR_RETURN(
            output, model_builder->DepthwiseConvolution(src, weights, attr));
      } else {
        return absl::InternalError(
            "Depthwise convolution only supports 1 or 2 inputs.");
      }
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::DEPTH_TO_SPACE: {
      const auto& attr =
          std::any_cast<const SpaceToDepthAttributes&>(node.attr);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectDepthToSpace(attr, op_def, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::DYNAMIC_UPDATE_SLICE: {
      auto gpu_op = SelectDynamicUpdateSlice(op_def, gpu_info);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::EMBEDDING_LOOKUP: {
      const auto& attr =
          std::any_cast<const EmbeddingLookupAttributes&>(node.attr);
      if (IsEmbeddingLookupQuantized(attr) &&
          (inputs.size() == 3 || inputs.size() == 4)) {
        return MakeQuantizedEmbeddingLookup(gpu_info, create_info, inputs,
                                            outputs, attr, model_builder);
      }
      std::unique_ptr<GPUOperation> gpu_op;
      SelectEmbeddingLookup(attr, op_def, gpu_info, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::FULLY_CONNECTED: {
      const auto& attr =
          std::any_cast<const FullyConnectedAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      if (attr.external_weights.has_value()) {
        return MakeFullyConnectedExternalWeights(src_ids, dst_ids, attr,
                                                 model_builder);
      } else if (inputs.size() >= 2) {
        ABSL_ASSIGN_OR_RETURN(auto weights,
                              model_builder->GetTensor(inputs[1]->id));
        GpuModelBuilder::TensorHandle bias;
        GpuModelBuilder::TensorHandle* bias_ptr = nullptr;
        if (inputs.size() == 3) {
          ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(inputs[2]->id));
          bias_ptr = &bias;
        }
        Convolution2DAttributes conv_attr;
        conv_attr.strides = HW(1, 1);
        conv_attr.dilations = HW(1, 1);
        conv_attr.padding.appended = HW(0, 0);
        conv_attr.padding.prepended = HW(0, 0);
        auto& conv_weights =
            conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
        auto weights_shape = inputs[1]->desc.GetBHWCShape();
        conv_weights.shape = OHWI(weights_shape.b, weights_shape.h,
                                  weights_shape.w, weights_shape.c);
        ABSL_ASSIGN_OR_RETURN(
            auto output,
            model_builder->Convolution(src, weights, bias_ptr, conv_attr));
        return model_builder->UpdateOutputTensor(output, outputs[0]->id);
      } else {
        return model_builder->UpdateOutputTensor(
            model_builder->FullyConnected(src, attr), outputs[0]->id);
      }
    }
    case OperationType::FULLY_CONNECTED_INT2: {
      const auto& attr =
          std::any_cast<const FullyConnectedInt2Attributes&>(node.attr);
      if (inputs.size() >= 4) {
        return MakeQuantizedFullyConnectedExternalWeights(
            gpu_info, create_info, op_def, OperationType::FULLY_CONNECTED_INT2,
            src_ids, dst_ids, model_builder,
            std::visit([](const auto& w) { return w.shape; }, attr.weights),
            attr.scale.shape);
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->FullyConnected(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::FULLY_CONNECTED_INT4: {
      const auto& attr =
          std::any_cast<const FullyConnectedInt4Attributes&>(node.attr);
      if (inputs.size() >= 4) {
        return MakeQuantizedFullyConnectedExternalWeights(
            gpu_info, create_info, op_def, OperationType::FULLY_CONNECTED_INT4,
            src_ids, dst_ids, model_builder,
            std::visit([](const auto& w) { return w.shape; }, attr.weights),
            attr.scale.shape);
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->FullyConnected(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::FULLY_CONNECTED_INT8: {
      const auto& attr =
          std::any_cast<const FullyConnectedInt8Attributes&>(node.attr);
      if (inputs.size() >= 4) {
        return MakeQuantizedFullyConnectedExternalWeights(
            gpu_info, create_info, op_def, OperationType::FULLY_CONNECTED_INT8,
            src_ids, dst_ids, model_builder, attr.weights.shape,
            attr.scale.shape);
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->FullyConnected(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::GROUP_NORM: {
      const auto& attr = std::any_cast<const GroupNormAttributes&>(node.attr);
      return MakeGroupNorm(gpu_info, create_info, attr, src_ids, dst_ids,
                           model_builder);
    }
    case OperationType::GATHER: {
      const auto& attr = std::any_cast<const GatherAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      ABSL_ASSIGN_OR_RETURN(auto indices,
                            model_builder->GetTensor(inputs[1]->id));
      auto output = model_builder->Gather(src, indices, attr.axis);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::LAYER_NORM: {
      const auto& attr = std::any_cast<const LayerNormAttributes&>(node.attr);
      return MakeLayerNorm(gpu_info, create_info, attr, src_ids, dst_ids,
                           model_builder);
    }
    case OperationType::MAX_INDEX: {
      const auto& attr = std::any_cast<const MaxIndexAttributes&>(node.attr);
      auto gpu_op = std::make_unique<Reduce>(
          CreateReduce({attr.dim}, inputs[0]->desc.GetBHWCShape(),
                       Reduce::Type::kMaximumIndex, op_def, gpu_info));
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::MAX_UNPOOLING_2D: {
      const auto& attr =
          std::any_cast<const MaxUnpooling2DAttributes&>(node.attr);
      auto gpu_op = SelectMaxUnpooling(attr, gpu_info, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::MEAN_STDDEV_NORMALIZATION: {
      MeanStdDevNormalization operation = CreateMeanStdDevNormalization(
          op_def, gpu_info, inputs[0]->desc.GetBHWCShape());
      auto gpu_op =
          std::make_unique<MeanStdDevNormalization>(std::move(operation));
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::ONE_HOT: {
      const auto& attr = std::any_cast<const OneHotAttributes&>(node.attr);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectOneHot(op_def, attr, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::PAD: {
      const auto& attr = std::any_cast<const PadAttributes&>(node.attr);
      auto gpu_op = SelectPadding(gpu_info, attr, op_def,
                                  inputs[0]->desc.GetBHWDCShape().c);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::POOLING_2D: {
      const auto& attr = std::any_cast<const Pooling2DAttributes&>(node.attr);
      auto gpu_op = SelectPooling(attr, gpu_info, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::POSITIONAL_EMBEDDING: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return absl::InvalidArgumentError(
            "PositionalEmbedding operation expects 2 inputs and a single "
            "output.");
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      ABSL_ASSIGN_OR_RETURN(auto position,
                            model_builder->GetTensor(inputs[1]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->PositionalEmbedding(src, position), outputs[0]->id);
    }
    case OperationType::PRELU: {
      const auto& attr = std::any_cast<const PReLUAttributes&>(node.attr);
      auto gpu_op = SelectPReLU(attr, gpu_info, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::QUANTIZE_AND_DEQUANTIZE: {
      const auto& attr =
          std::any_cast<const QuantizeAndDequantizeAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      auto output = model_builder->QuantizeAndDequantize(src, attr);
      return model_builder->UpdateOutputTensor(output, outputs[0]->id);
    }
    case OperationType::RELU: {
      const auto& attr = std::any_cast<const ReLUAttributes&>(node.attr);
      auto gpu_op = SelectReLU(attr, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::RESAMPLER: {
      auto gpu_op = SelectResampler(op_def, gpu_info);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::RESHAPE: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      if (const auto* attr3d = std::any_cast<Reshape3DAttributes>(&node.attr)) {
        auto output = model_builder->Reshape(src, attr3d->new_shape);
        return model_builder->UpdateOutputTensor(output, outputs[0]->id);
      } else {
        const auto& attr = std::any_cast<const ReshapeAttributes&>(node.attr);
        auto output = model_builder->Reshape(src, attr.new_shape);
        return model_builder->UpdateOutputTensor(output, outputs[0]->id);
      }
    }
    case OperationType::RESIZE: {
      const auto& attr = std::any_cast<const Resize2DAttributes&>(node.attr);
      std::unique_ptr<GPUOperation> gpu_op;
      ABSL_RETURN_IF_ERROR(SelectResize(attr, op_def, &gpu_op));
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::REVERSE: {
      const auto& attr = std::any_cast<const ReverseAttributes&>(node.attr);
      auto gpu_op = SelectReverse(attr, op_def);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::RMS_NORM: {
      const auto& attr = std::any_cast<const RmsNormAttributes&>(node.attr);
      return MakeRmsNorm(gpu_info, create_info, src_ids, dst_ids, attr,
                         model_builder);
    }
    case OperationType::ROPE: {
      RoPEAttributes attr;
      if (node.attr.has_value()) {
        attr = std::any_cast<const RoPEAttributes&>(node.attr);
      }
      return MakeRoPE(gpu_info, create_info, src_ids, dst_ids, attr,
                      model_builder);
    }
    case OperationType::SCALED_DOT_PRODUCT_ATTENTION: {
      const auto& attr =
          std::any_cast<const ScaledDotProductAttentionAttributes&>(node.attr);
      return MakeScaledDotProductAttention(gpu_info, create_info, src_ids,
                                           dst_ids, attr, model_builder);
    }
    case OperationType::SELECT_V2: {
      const auto& attr = std::any_cast<const SelectV2Attributes&>(node.attr);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectSelectV2(op_def, attr, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::SLICE: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      if (const auto* attr3d = std::any_cast<Slice3DAttributes>(&node.attr)) {
        return model_builder->UpdateOutputTensor(
            model_builder->StridedSlice(src, *attr3d), outputs[0]->id);
      } else {
        const auto& attr = std::any_cast<const SliceAttributes&>(node.attr);
        return model_builder->UpdateOutputTensor(
            model_builder->StridedSlice(src, attr), outputs[0]->id);
      }
    }
    case OperationType::SOFTMAX: {
      ABSL_ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle src,
                            model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(model_builder->Softmax(src),
                                               outputs[0]->id);
    }
    case OperationType::SPACE_TO_DEPTH: {
      const auto& attr =
          std::any_cast<const SpaceToDepthAttributes&>(node.attr);
      std::unique_ptr<GPUOperation> gpu_op;
      SelectSpaceToDepth(attr, op_def, &gpu_op);
      model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                     node.name);
      return absl::OkStatus();
    }
    case OperationType::SPLIT: {
      const auto& attr = std::any_cast<const SplitAttributes&>(node.attr);
      std::vector<int> sizes(outputs.size());
      std::vector<ValueId> output_ids(outputs.size());
      for (int i = 0; i < outputs.size(); ++i) {
        sizes[i] = outputs[i]->desc.GetBHWCShape().get(attr.axis);
        output_ids[i] = outputs[i]->id;
      }
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensors(
          model_builder->Split(src, attr.axis, sizes), output_ids);
    }
    case OperationType::TILE: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensor(
          model_builder->Tile(src, outputs[0]->desc.GetBHWDCShape()),
          outputs[0]->id);
    }
    case OperationType::TOP_K: {
      const auto& attr = std::any_cast<const TopKAttributes&>(node.attr);
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      return model_builder->UpdateOutputTensors(
          model_builder->TopK(src, attr.k),
          {static_cast<ValueId>(outputs[0]->id),
           static_cast<ValueId>(outputs[1]->id)});
    }
    case OperationType::TRANSPOSE: {
      ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
      if (const auto* attr3d =
              std::any_cast<Transpose3DAttributes>(&node.attr)) {
        return model_builder->UpdateOutputTensor(
            model_builder->Transpose(src, attr3d->perm), outputs[0]->id);
      } else {
        const auto& attr = std::any_cast<const TransposeAttributes&>(node.attr);
        return model_builder->UpdateOutputTensor(
            model_builder->Transpose(src, attr.perm), outputs[0]->id);
      }
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported op type: ", node.name));
  }
}  // NOLINT(readability/fn_size)

}  // namespace ml_drift
