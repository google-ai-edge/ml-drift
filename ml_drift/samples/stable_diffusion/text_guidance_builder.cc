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

#include "ml_drift/samples/stable_diffusion/text_guidance_builder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/stable_diffusion/model_data_loader.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {

// TODO(jqtang): Support the public SD 2.x weights file naming.
absl::Status TextGuidanceBuilder::Build(
    const Config& config, const GpuInfo& gpu_info,
    const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
    GpuModelBuilder::TensorHandle* src_ptr,
    GpuModelBuilder::TensorHandle* mask_ptr,
    GpuModelBuilder::TensorHandle* dst_ptr,
    GpuModelBuilder::TensorHandle* text_proj_ptr) {
  config_ = config;
  ASSIGN_OR_RETURN(model_data_loader_,
                   ModelDataLoader::CreateFromWeightsDir(config.file_folder));
  builder_ = GpuModelBuilder(gpu_info, create_info.hints, create_info.precision,
                             create_info.storage_type);
  const DataType float_type =
      DeduceDataTypeFromPrecision(create_info.precision);
  auto src_tensor = builder_.AddTensor(BHWC(1, 1, 2, 77), DataType::INT32);
  std::vector<GpuModelBuilder::TensorHandle> input_tensors;
  input_tensors.push_back(std::move(src_tensor));

  if (mask_ptr) {
    GpuModelBuilder::TensorHandle mask_tensor =
        builder_.AddTensor(BHWC(1, 2, 77, 77), float_type);
    auto split_masks = builder_.Split(mask_tensor, Axis::HEIGHT, 1);
    for (int i = 0; i < 2; ++i) {
      split_masks[i] = builder_.Tile(split_masks[i], Axis::HEIGHT, 16);
    }
    mask_ = builder_.Concat(split_masks[0], split_masks[1], Axis::HEIGHT);
    input_tensors.push_back(std::move(mask_tensor));
  } else {
    TensorFloat32 causal_mask_tensor;
    const BHWC shape(1, 1, 77, 77);
    causal_mask_tensor.shape = shape;
    causal_mask_tensor.data.resize(
        causal_mask_tensor.shape.DimensionsProduct());
    for (int w = 0; w < shape.w; ++w) {
      for (int c = 0; c < shape.c; ++c) {
        float value = w < c ? -65500.0f : 65500.0f;
        causal_mask_tensor.data[w * shape.c + c] = value;
      }
    }
    mask_ = builder_.AddConstantTensor(causal_mask_tensor, float_type);
  }
  std::vector<GpuModelBuilder::ValueId> output_ids;
  if (text_proj_ptr) {
    ASSIGN_OR_RETURN(auto t, MakeTextGuidanceWithTextProjection(
                                 input_tensors[0], float_type));
    if (dst_ptr) {
      *dst_ptr = t[0];
    }
    if (text_proj_ptr) {
      *text_proj_ptr = t[1];
    }
    output_ids = {t[0].id, t[1].id};
  } else {
    ASSIGN_OR_RETURN(auto t, MakeTextGuidance(input_tensors[0], float_type));
    if (dst_ptr) {
      *dst_ptr = t;
    }
    output_ids = {t.id};
  }

  if (src_ptr) {
    *src_ptr = input_tensors[0];
  }
  if (mask_ptr) {
    *mask_ptr = input_tensors[1];
  }
  std::vector<GpuModelBuilder::ValueId> input_ids(input_tensors.size());
  for (int i = 0; i < input_tensors.size(); ++i) {
    input_ids[i] = input_tensors[i].id;
  }
  return builder_.GetGpuModel(input_ids, output_ids, gpu_model);
}

GpuModelBuilder::TensorHandle TextGuidanceBuilder::MakeLinear(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels, bool bias) {
  auto weights_shape =
      OHWI(out_channels, 1, 1, src.tensor_desc.GetBHWCShape().c);
  if (bias) {
    auto [weights_data, bias_data] =
        model_data_loader_
            ->GetData(name + ".weight.bin", name + ".bias.bin",
                      weights_shape.DimensionsProduct(), weights_shape.o)
            .value();
    return builder_.Convolution(
        src, MakeConvAttributes(weights_data, bias_data, weights_shape,
                                HW(1, 1), name));
  }
  auto weights_data =
      model_data_loader_
          ->GetData(name + ".weight.bin", weights_shape.DimensionsProduct())
          .value();
  return builder_.Convolution(
      src, MakeConvAttributes(weights_data, std::vector<half>(), weights_shape,
                              HW(1, 1), name));
}

GpuModelBuilder::TensorHandle TextGuidanceBuilder::MakeLayerNorm(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  auto gamma = CreateLinearTensor(
      model_data_loader_
          ->GetData(name + ".weight.bin", src.tensor_desc.GetBHWCShape().c)
          .value());
  auto beta = CreateLinearTensor(
      model_data_loader_
          ->GetData(name + ".bias.bin", src.tensor_desc.GetBHWCShape().c)
          .value());
  return builder_.LayerNormalization(src, gamma, beta, /*epsilon*/ 1e-5f);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TextGuidanceBuilder::MakeTextGuidance(const GpuModelBuilder::TensorHandle& x_in,
                                      DataType float_type) {
  const std::string name = "cond_stage_model.transformer.text_model";
  ASSIGN_OR_RETURN(auto x,
                   MakeTextEmbeddings(x_in, float_type, name + ".embeddings"));
  ASSIGN_OR_RETURN(x, MakeTextEncoder(x, name + ".encoder"));
  if (config_.skip_final_layer_norm) {
    return x;
  }
  return MakeLayerNorm(x, name + ".final_layer_norm");
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
TextGuidanceBuilder::MakeTextGuidanceWithTextProjection(
    const GpuModelBuilder::TensorHandle& x_in, DataType float_type) {
  std::vector<GpuModelBuilder::TensorHandle> outputs(2);
  ASSIGN_OR_RETURN(outputs[0], MakeTextGuidance(x_in, float_type));
  const std::string name = "cond_stage_model.transformer.text_model";
  auto final_norm = outputs[0];
  if (config_.skip_final_layer_norm) {
    // not calculated in MakeTextGuidance when skip_final_layer_norm is true.
    final_norm = MakeLayerNorm(outputs[0], name + ".final_layer_norm");
  }
  auto argmax =
      builder_.Reduce(x_in, Reduce::Type::kMaximumIndex, {Axis::CHANNELS});
  auto inds = builder_.Reshape(argmax, BHWC(1, 1, 2, 1));
  auto text_proj = AddCustomGather(final_norm, inds);
  auto text_proj_split = builder_.Split(text_proj, Axis::HEIGHT, 1);
  for (int i = 0; i < 2; ++i) {
    text_proj_split[i] = MakeLinear(
        text_proj_split[i], name + ".text_projection", config_.embedding_size);
  }
  outputs[1] =
      builder_.Concat(text_proj_split[0], text_proj_split[1], Axis::BATCH);
  return outputs;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TextGuidanceBuilder::MakeTextEncoderLayer(
    const GpuModelBuilder::TensorHandle& xIn, const std::string& name) {
  auto x = xIn;
  x = MakeLayerNorm(x, name + ".layer_norm1");
  ASSIGN_OR_RETURN(x, MakeTextAttention(x, name + ".self_attn"));
  x = builder_.Add(x, xIn);
  auto skip = x;
  x = MakeLayerNorm(x, name + ".layer_norm2");
  x = MakeLinear(x, name + ".mlp.fc1", config_.embedding_size * 4);

  // To match JAX's quick_gelu function: x * jax.nn.sigmoid(1.702 * x)
  auto quick_gelu = builder_.Multiplication(x, 1.702f);
  quick_gelu = builder_.Elementwise(quick_gelu, OperationType::SIGMOID);
  x = builder_.Multiplication(x, quick_gelu);

  x = MakeLinear(x, name + ".mlp.fc2", config_.embedding_size);
  return builder_.Add(x, skip);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TextGuidanceBuilder::MakeTextEncoder(const GpuModelBuilder::TensorHandle& xIn,
                                     const std::string& name) {
  auto x = xIn;
  for (int i = 0; i < config_.num_layers; i++) {
    ASSIGN_OR_RETURN(
        x, MakeTextEncoderLayer(x, name + ".layers." + std::to_string(i)));
  }
  return x;
}

GpuModelBuilder::TensorHandle TextGuidanceBuilder::AddCustomGather(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& indices) {
  std::vector<GpuModelBuilder::TensorHandle> src_split = {src};
  std::vector<GpuModelBuilder::TensorHandle> indices_split = {indices};
  if (src.tensor_desc.GetBHWCShape().h != 1) {
    src_split = builder_.Split(src, Axis::HEIGHT, /*tile_size=*/1);
  }
  if (indices.tensor_desc.GetBHWCShape().w != 1) {
    indices_split = builder_.Split(indices, Axis::WIDTH, /*tile_size=*/1);
  }
  const size_t num_splits = std::max(src_split.size(), indices_split.size());
  std::vector<GpuModelBuilder::TensorHandle> dst_split(num_splits);
  for (size_t i = 0; i < num_splits; ++i) {
    auto src_i = src_split[std::min(i, src_split.size() - 1)];
    auto indices_i = indices_split[std::min(i, indices_split.size() - 1)];
    dst_split[i] = builder_.Gather(src_i, indices_i, Axis::WIDTH);
  }
  return builder_.Concat(dst_split, Axis::HEIGHT);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TextGuidanceBuilder::MakeTextEmbeddings(
    const GpuModelBuilder::TensorHandle& x_in, DataType float_type,
    const std::string& name) {
  TensorFloat32 token_embeddings_tensor;
  {
    const BHWC shape(1, 1, 49408, config_.embedding_size);
    auto data = model_data_loader_
                    ->GetData(name + ".token_embedding.weight.bin",
                              shape.b * shape.h * shape.w * shape.c)
                    .value();
    token_embeddings_tensor.shape = shape;
    token_embeddings_tensor.data.resize(
        token_embeddings_tensor.shape.DimensionsProduct());
    for (int i = 0; i < token_embeddings_tensor.data.size(); ++i) {
      token_embeddings_tensor.data[i] = data[i];
    }
  }
  auto token_embeddings =
      builder_.AddConstantTensor(token_embeddings_tensor, float_type);
  TensorFloat32 position_embeddings_tensor;
  {
    const BHWC shape(1, 1, 77, config_.embedding_size);
    auto data = model_data_loader_
                    ->GetData(name + ".position_embedding.weight.bin",
                              shape.b * shape.h * shape.w * shape.c)
                    .value();
    position_embeddings_tensor.shape = shape;
    position_embeddings_tensor.data.resize(
        position_embeddings_tensor.shape.DimensionsProduct());
    for (int i = 0; i < position_embeddings_tensor.data.size(); ++i) {
      position_embeddings_tensor.data[i] = data[i];
    }
  }
  auto position_embeddings =
      builder_.AddConstantTensor(position_embeddings_tensor, float_type);
  auto embeddings = AddCustomGather(token_embeddings, x_in);
  return builder_.Add(embeddings, position_embeddings);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TextGuidanceBuilder::MakeTextAttention(const GpuModelBuilder::TensorHandle& xIn,
                                       const std::string& name) {
  int dHead = 64;
  int c = config_.embedding_size;
  auto q = MakeLinear(xIn, name + ".q_proj", c);
  auto k = MakeLinear(xIn, name + ".k_proj", c);
  auto v = MakeLinear(xIn, name + ".v_proj", c);

  int n = xIn.tensor_desc.GetBHWCShape().h;
  int t = xIn.tensor_desc.GetBHWCShape().w;
  q = builder_.Reshape(q, BHWC(n, t, config_.num_heads, dHead));
  k = builder_.Reshape(k, BHWC(n, t, config_.num_heads, dHead));
  v = builder_.Reshape(v, BHWC(n, t, config_.num_heads, dHead));

  q = builder_.Transpose(q, BHWC(0, 2, 1, 3));
  k = builder_.Transpose(k, BHWC(0, 2, 1, 3));
  k = builder_.Transpose(k, BHWC(0, 1, 3, 2));
  v = builder_.Transpose(v, BHWC(0, 2, 1, 3));

  q = builder_.Reshape(
      q, BHWC{1, q.tensor_desc.GetBHWCShape().h * n,
              q.tensor_desc.GetBHWCShape().w, q.tensor_desc.GetBHWCShape().c});
  k = builder_.Reshape(
      k, BHWC{1, k.tensor_desc.GetBHWCShape().h * n,
              k.tensor_desc.GetBHWCShape().w, k.tensor_desc.GetBHWCShape().c});
  v = builder_.Reshape(
      v, BHWC{1, v.tensor_desc.GetBHWCShape().h * n,
              v.tensor_desc.GetBHWCShape().w, v.tensor_desc.GetBHWCShape().c});

  ASSIGN_OR_RETURN(auto att, builder_.BatchedMatMul(q, k));
  att = builder_.Multiplication(att, 1.0f / sqrt(static_cast<float>(dHead)));
  att = builder_.Elementwise(att, mask_, OperationType::MINIMUM);
  att = builder_.Softmax(att);
  ASSIGN_OR_RETURN(att, builder_.BatchedMatMul(att, v));

  att = builder_.Reshape(att, BHWC{n, att.tensor_desc.GetBHWCShape().h / n,
                                   att.tensor_desc.GetBHWCShape().w,
                                   att.tensor_desc.GetBHWCShape().c});

  att = builder_.Transpose(att, BHWC(0, 2, 1, 3));
  att = builder_.Reshape(att, BHWC(1, n, t, c));
  return MakeLinear(att, name + ".out_proj", c);
}

}  // namespace ml_drift
