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

#include "ml_drift/samples/stable_diffusion/unet_builder.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {

absl::Status UnetBuilder::Build(
    const Config& config, const GpuInfo& gpu_info,
    const CreateGpuModelInfo& create_info, int width, int height,
    GpuModel* gpu_model, GpuModelBuilder::TensorHandle* latent_ptr,
    GpuModelBuilder::TensorHandle* temb_ptr,
    GpuModelBuilder::TensorHandle* guidance_ptr,
    // TODO(dlho): Delete text_proj_ptr after all binaries updated.
    GpuModelBuilder::TensorHandle* text_proj_ptr,
    GpuModelBuilder::TensorHandle* masked_image_latent_ptr,
    GpuModelBuilder::TensorHandle* eta0_ptr,
    GpuModelBuilder::TensorHandle* eta1_ptr,
    GpuModelBuilder::TensorHandle* debug_tensor_ptr) {
  config_ = config;
  if (!config_.lora_file_dir.empty() &&
      !config_.lora_weights_layer_mapping.empty()) {
    return absl::InvalidArgumentError(
        "Only one of lora_file_dir and lora_weights_layer_mapping should "
        "be set.");
  }
  builder_ = GpuModelBuilder(gpu_info, create_info.hints, create_info.precision,
                             create_info.storage_type);
  gpu_info_ = gpu_info;
  const DataType float_type =
      DeduceDataTypeFromPrecision(create_info.precision);

  auto src_tensor = builder_.AddTensor(
      BHWC(1, height, width, config.in_channels), float_type);
  auto input_tensor = src_tensor;
  auto masked_image_latent_tensor =
      builder_.AddTensor(BHWC(1, height, width, 9), float_type);
  if (masked_image_latent_ptr) {
    input_tensor =
        builder_.Concat(src_tensor, masked_image_latent_tensor, Axis::CHANNELS);
  }
  auto temb_tensor = builder_.AddTensor(
      BHWC(1, 1, 1, config.model_channels * config.channel_mult[0]),
      float_type);
  auto guidance_tensor =
      builder_.AddTensor(BHWC(1, 2, 77, config.context_dim), float_type);
  auto text_proj_tensor =
      builder_.AddTensor(BHWC(2, 1, 1, config.context_dim), float_type);
  auto emb = MakeTimeEmbed(temb_tensor, "model.diffusion_model.time_embed");
  if (text_proj_ptr) {
    auto pooled_text_proj_tensor = MakePooledTextProjection(
        text_proj_tensor, "model.diffusion_model.pooled_text_embedding.linear",
        emb.tensor_desc.GetBHWCShape().c);
    emb = builder_.Tile(emb, Axis::BATCH);
    emb = builder_.Add(emb, pooled_text_proj_tensor);
  }
  emb = builder_.SiLU(emb);
  ABSL_ASSIGN_OR_RETURN(
      auto t, MakeUNet(config, input_tensor, emb, guidance_tensor,
                       "model.diffusion_model", /*plugins=*/{},
                       /*plugins_strength=*/nullptr, /*control=*/nullptr,
                       /*only_mid_control=*/false, debug_tensor_ptr));
  auto etas = builder_.Split(t, Axis::BATCH, 1);

  if (latent_ptr) {
    *latent_ptr = src_tensor;
  }
  if (temb_ptr) {
    *temb_ptr = temb_tensor;
  }
  if (text_proj_ptr) {
    *text_proj_ptr = text_proj_tensor;
  }
  if (masked_image_latent_ptr) {
    *masked_image_latent_ptr = masked_image_latent_tensor;
  }
  if (guidance_ptr) {
    *guidance_ptr = guidance_tensor;
  }
  if (eta0_ptr) {
    *eta0_ptr = etas[0];
  }
  if (eta1_ptr) {
    *eta1_ptr = etas[1];
  }
  std::vector<ValueId> input_ids = {src_tensor.id, temb_tensor.id,
                                    guidance_tensor.id};
  std::vector<ValueId> output_ids = {etas[0].id, etas[1].id, src_tensor.id,
                                     guidance_tensor.id};
  if (text_proj_ptr) {
    input_ids.push_back(text_proj_tensor.id);
    output_ids.push_back(text_proj_tensor.id);
  }
  if (masked_image_latent_ptr) {
    input_ids.push_back(masked_image_latent_tensor.id);
    output_ids.push_back(masked_image_latent_tensor.id);
  }
  return builder_.GetGpuModel(input_ids, output_ids, gpu_model);
}

absl::Status UnetBuilder::BuildControlNet(
    const Config& config, const GpuInfo& gpu_info,
    const CreateGpuModelInfo& create_info, int width, int height,
    GpuModel* gpu_model, GpuModelBuilder::TensorHandle* latent_ptr,
    GpuModelBuilder::TensorHandle* condition_ptr,
    GpuModelBuilder::TensorHandle* temb_ptr,
    GpuModelBuilder::TensorHandle* guidance_ptr,
    GpuModelBuilder::TensorHandle* eta0_ptr,
    GpuModelBuilder::TensorHandle* eta1_ptr,
    GpuModelBuilder::TensorHandle* debug_tensor_ptr) {
  config_ = config;
  builder_ = GpuModelBuilder(gpu_info, create_info.hints, create_info.precision,
                             create_info.storage_type);
  const DataType float_type =
      DeduceDataTypeFromPrecision(create_info.precision);
  gpu_info_ = gpu_info;

  auto src_tensor = builder_.AddTensor(
      BHWC(1, height, width, config.in_channels), float_type);
  auto condition_tensor =
      builder_.AddTensor(BHWC(1, height * 8, width * 8, 3), float_type);
  auto temb_tensor = builder_.AddTensor(
      BHWC(1, 1, 1, config.model_channels * config.channel_mult[0]),
      float_type);
  auto guidance_tensor =
      builder_.AddTensor(BHWC(1, 2, 77, config.context_dim), float_type);

  auto emb_control = MakeTimeEmbed(temb_tensor, "control_model.time_embed");
  ABSL_ASSIGN_OR_RETURN(
      auto control_tensors,
      MakeControlNet(config, src_tensor, condition_tensor, emb_control,
                     guidance_tensor, "control_model", debug_tensor_ptr));
  auto emb = MakeTimeEmbed(temb_tensor, "model.diffusion_model.time_embed");
  ABSL_ASSIGN_OR_RETURN(
      auto t, MakeUNet(config, src_tensor, emb, guidance_tensor,
                       "model.diffusion_model",

                       /*plugins=*/{}, /*plugins_strength=*/nullptr,
                       /*control=*/&control_tensors, /*only_mid_control=*/false,
                       debug_tensor_ptr));
  auto etas = builder_.Split(t, Axis::BATCH, 1);

  if (latent_ptr) {
    *latent_ptr = src_tensor;
  }
  if (condition_ptr) {
    *condition_ptr = condition_tensor;
  }
  if (temb_ptr) {
    *temb_ptr = temb_tensor;
  }
  if (guidance_ptr) {
    *guidance_ptr = guidance_tensor;
  }
  if (eta0_ptr) {
    *eta0_ptr = etas[0];
  }
  if (eta1_ptr) {
    *eta1_ptr = etas[1];
  }
  return builder_.GetGpuModel(
      {src_tensor.id, condition_tensor.id, temb_tensor.id, guidance_tensor.id},
      {etas[0].id, etas[1].id, src_tensor.id, condition_tensor.id,
       guidance_tensor.id},
      gpu_model);
}

absl::Status UnetBuilder::BuildUNetWithPlugins(
    const Config& config, const GpuInfo& gpu_info,
    const CreateGpuModelInfo& create_info, int width, int height,
    GpuModel* gpu_model, GpuModelBuilder::TensorHandle* latent_ptr,
    std::vector<GpuModelBuilder::TensorHandle>* plugin_tensors,
    GpuModelBuilder::TensorHandle* temb_ptr,
    GpuModelBuilder::TensorHandle* guidance_ptr,
    GpuModelBuilder::TensorHandle* eta0_ptr,
    GpuModelBuilder::TensorHandle* eta1_ptr,
    GpuModelBuilder::TensorHandle* plugins_strength_ptr,
    GpuModelBuilder::TensorHandle* debug_tensor_ptr) {
  config_ = config;
  builder_ = GpuModelBuilder(gpu_info, create_info.hints, create_info.precision,
                             create_info.storage_type);
  gpu_info_ = gpu_info;
  const DataType float_type =
      DeduceDataTypeFromPrecision(create_info.precision);

  auto src_tensor = builder_.AddTensor(
      BHWC(1, height, width, config.in_channels), float_type);
  auto temb_tensor = builder_.AddTensor(
      BHWC(1, 1, 1, config.model_channels * config.channel_mult[0]),
      float_type);
  auto guidance_tensor =
      builder_.AddTensor(BHWC(1, 2, 77, config.context_dim), float_type);
  plugin_tensors->emplace_back(
      builder_.AddTensor(BHWC(1, 64, 64, 320), float_type));
  plugin_tensors->emplace_back(
      builder_.AddTensor(BHWC(1, 32, 32, 320), float_type));
  plugin_tensors->emplace_back(
      builder_.AddTensor(BHWC(1, 16, 16, 640), float_type));
  plugin_tensors->emplace_back(
      builder_.AddTensor(BHWC(1, 8, 8, 1280), float_type));
  plugin_tensors->emplace_back(
      builder_.AddTensor(BHWC(1, 8, 8, 1280), float_type));
  auto plugins_strength_tensor =
      builder_.AddTensor(BHWC(1, 1, 1, 1), float_type);
  auto emb = MakeTimeEmbed(temb_tensor, "model.diffusion_model.time_embed");
  ABSL_ASSIGN_OR_RETURN(
      auto t, MakeUNet(config, src_tensor, emb, guidance_tensor,
                       "model.diffusion_model", /*plugins=*/*plugin_tensors,
                       /*plugins_strength=*/&plugins_strength_tensor,
                       /*control=*/nullptr, /*only_mid_control=*/false,
                       debug_tensor_ptr));
  auto etas = builder_.Split(t, Axis::BATCH, 1);

  if (latent_ptr) {
    *latent_ptr = src_tensor;
  }
  if (temb_ptr) {
    *temb_ptr = temb_tensor;
  }
  if (guidance_ptr) {
    *guidance_ptr = guidance_tensor;
  }
  if (eta0_ptr) {
    *eta0_ptr = etas[0];
  }
  if (eta1_ptr) {
    *eta1_ptr = etas[1];
  }
  if (plugins_strength_ptr) {
    *plugins_strength_ptr = plugins_strength_tensor;
  }
  return builder_.GetGpuModel(
      {
          src_tensor.id,
          temb_tensor.id,
          guidance_tensor.id,
          (*plugin_tensors)[0].id,
          (*plugin_tensors)[1].id,
          (*plugin_tensors)[2].id,
          (*plugin_tensors)[3].id,
          (*plugin_tensors)[4].id,
          plugins_strength_tensor.id,
      },
      {
          etas[0].id,
          etas[1].id,
          src_tensor.id,
          guidance_tensor.id,
          (*plugin_tensors)[0].id,
          (*plugin_tensors)[1].id,
          (*plugin_tensors)[2].id,
          (*plugin_tensors)[3].id,
          (*plugin_tensors)[4].id,
      },
      gpu_model);
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeConv(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels, int kernel_hw, int stride_hw, bool bias) {
  auto weights_shape = OHWI(out_channels, kernel_hw, kernel_hw,
                            src.tensor_desc.GetBHWCShape().c);
  const int weights_size = weights_shape.DimensionsProduct();
  auto weight_buffer = std::make_unique<half[]>(weights_size);
  const int bias_size = bias ? weights_shape.o : 0;
  auto bias_buffer = std::make_unique<half[]>(bias_size);
  auto weights = LoadF16Ex(config_.unet_file_dir + name + ".weight.bin",
                           weights_size, weight_buffer.get());
  auto bias2 = LoadF16Ex(config_.unet_file_dir + name + ".bias.bin", bias_size,
                         bias_buffer.get());
  return builder_.Convolution(
      src, MakeConvAttributes(weights, bias2, weights_shape,
                              HW(stride_hw, stride_hw), name));
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeDwConv(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int kernel_hw, int stride_hw, bool bias) {
  auto weights_shape =
      OHWI(1, kernel_hw, kernel_hw, src.tensor_desc.GetBHWCShape().c);
  const int weights_size = weights_shape.DimensionsProduct();
  auto weight_buffer = std::make_unique<half[]>(weights_size);
  const int bias_size = bias ? weights_shape.i : 0;
  auto bias_buffer = std::make_unique<half[]>(bias_size);
  auto weights = LoadF16Ex(config_.unet_file_dir + name + ".weight.bin",
                           weights_size, weight_buffer.get());
  auto bias2 = LoadF16Ex(config_.unet_file_dir + name + ".bias.bin", bias_size,
                         bias_buffer.get());
  return builder_.DepthwiseConvolution(
      src, MakeDwConvAttributes(weights, bias2, weights_shape,
                                HW(stride_hw, stride_hw), name));
}

Convolution2DAttributes UnetBuilder::ReadConvAttributes(
    const OHWI& weights_shape, const std::string& name, bool bias, bool lora) {
  const int weights_size = weights_shape.DimensionsProduct();
  auto weight_buffer = std::make_unique<half[]>(weights_size);
  const int bias_size = bias ? weights_shape.o : 0;
  auto bias_buffer = std::make_unique<half[]>(bias_size);
  absl::Span<const half> weights;
  absl::Span<const half> bias2;
  if (lora) {
    if (config_.lora_weights_layer_mapping.empty()) {
      weights = LoadF16Ex(config_.lora_file_dir + name + ".weight.bin",
                          weights_size, weight_buffer.get());
      bias2 = LoadF16Ex(config_.lora_file_dir + name + ".bias.bin", bias_size,
                        bias_buffer.get());
    } else {
      weights = absl::MakeConstSpan(
          reinterpret_cast<const half*>(
              config_.lora_weights_layer_mapping[name + ".weight.bin"]),
          weights_size);
      bias2 = absl::MakeConstSpan(
          reinterpret_cast<const half*>(
              config_.lora_weights_layer_mapping[name + ".bias.bin"]),
          bias_size);
    }
  } else {
    weights = LoadF16Ex(config_.unet_file_dir + name + ".weight.bin",
                        weights_size, weight_buffer.get());
    if (bias) {
      bias2 = LoadF16Ex(config_.unet_file_dir + name + ".bias.bin", bias_size,
                        bias_buffer.get());
    }
  }
  return ml_drift::MakeConvAttributes(weights, bias2, weights_shape, HW(1, 1),
                                      name);
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeLinear(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels, bool bias, bool lora) {
  auto weights_shape =
      OHWI(out_channels, 1, 1, src.tensor_desc.GetBHWCShape().c);
  Convolution2DAttributes attributes =
      ReadConvAttributes(weights_shape, name, bias, lora);
  return builder_.Convolution(src, attributes);
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeLoraInjectedLinear(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels) {
  auto lora_down = MakeLinear(src, name + ".down", config_.lora_rank,
                              /*bias=*/false, /*lora=*/true);
  auto lora_up = MakeLinear(lora_down, name + ".up", out_channels,
                            /*bias=*/false, /*lora=*/true);
  return builder_.Multiplication(lora_up, config_.lora_scale);
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeGroupNorm(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  const int size = src.tensor_desc.GetBHWCShape().c;
  auto weight_buffer = std::make_unique<half[]>(size);
  auto bias_buffer = std::make_unique<half[]>(size);
  auto gamma = LoadF16Ex(config_.unet_file_dir + name + ".weight.bin", size,
                         weight_buffer.get());
  auto gamma2 = CreateLinearTensor(gamma);
  auto beta = LoadF16Ex(config_.unet_file_dir + name + ".bias.bin", size,
                        bias_buffer.get());
  auto beta2 = CreateLinearTensor(beta);
  return builder_.HWCGroupNorm(src, /*groups=*/32, /*epsilon=*/1e-5f, gamma2,
                               beta2);
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeLayerNorm(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    float epsilon) {
  const int size = src.tensor_desc.GetBHWCShape().c;
  auto weight_buffer = std::make_unique<half[]>(size);
  auto bias_buffer = std::make_unique<half[]>(size);
  auto gamma = LoadF16Ex(config_.unet_file_dir + name + ".weight.bin", size,
                         weight_buffer.get());
  auto gamma2 = CreateLinearTensor(gamma);
  auto beta = LoadF16Ex(config_.unet_file_dir + name + ".bias.bin", size,
                        bias_buffer.get());
  auto beta2 = CreateLinearTensor(beta);
  return builder_.LayerNormalization(src, gamma2, beta2, epsilon);
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeTimeEmbed(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  auto x = src;
  x = MakeLinear(x, name + ".0", src.tensor_desc.GetBHWCShape().c * 4);
  x = builder_.SiLU(x);
  x = MakeLinear(x, name + ".2", src.tensor_desc.GetBHWCShape().c * 4);
  return x;
}

GpuModelBuilder::TensorHandle UnetBuilder::MakePooledTextProjection(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_dim_size) {
  auto x = src;
  x = MakeLinear(x, name + ".1", out_dim_size);
  x = builder_.SiLU(x);
  x = MakeLinear(x, name + ".2", out_dim_size);
  return x;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> UnetBuilder::MakeCrossAttention(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    const GpuModelBuilder::TensorHandle* context_ptr, int num_attn_heads) {
  const int c = src.tensor_desc.GetBHWCShape().c;
  const int attn_head_dim = c / num_attn_heads;
  auto context = context_ptr ? *context_ptr : src;
  auto q = MakeLinear(src, name + ".to_q", c, false);
  auto k = MakeLinear(context, name + ".to_k", c, false);
  GpuModelBuilder::TensorHandle v;
  if (context_ptr == nullptr && config_.self_attn_share_kv_proj) {
    v = k;
  } else {
    v = MakeLinear(context, name + ".to_v", c, false);
  }
  bool apply_lora = !config_.lora_file_dir.empty() ||
                    !config_.lora_weights_layer_mapping.empty();
  if (apply_lora) {
    auto lora_q = MakeLoraInjectedLinear(src, name + ".to_q_lora", c);
    q = builder_.Add(q, lora_q);
    auto lora_k = MakeLoraInjectedLinear(context, name + ".to_k_lora", c);
    k = builder_.Add(k, lora_k);
    auto lora_v = MakeLoraInjectedLinear(context, name + ".to_v_lora", c);
    v = builder_.Add(v, lora_v);
  }
  k = builder_.Multiplication(
      k, 1.0f / std::sqrt(static_cast<float>(attn_head_dim)));
  const int n = src.tensor_desc.GetBHWCShape().b;
  const int hw =
      src.tensor_desc.GetBHWCShape().h * src.tensor_desc.GetBHWCShape().w;
  int t = context.tensor_desc.GetBHWCShape().w;
  if (!context_ptr) {
    t *= context.tensor_desc.GetBHWCShape().h;
  }
  q = builder_.Reshape(q, BHWC{n, hw, num_attn_heads, attn_head_dim});
  k = builder_.Reshape(k, BHWC{n, t, num_attn_heads, attn_head_dim});
  v = builder_.Reshape(v, BHWC{n, t, num_attn_heads, attn_head_dim});

  q = builder_.Transpose(q, BHWC(0, 2, 1, 3));
  k = builder_.Transpose(k, BHWC(0, 2, 3, 1));
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

  ABSL_ASSIGN_OR_RETURN(GpuModelBuilder::TensorHandle att,
                        builder_.BatchedMatMulSoftmaxBatchedMatMul(q, k, v));
  att = builder_.Reshape(att, BHWC{n, att.tensor_desc.GetBHWCShape().h / n,
                                   att.tensor_desc.GetBHWCShape().w,
                                   att.tensor_desc.GetBHWCShape().c});
  att = builder_.Transpose(att, BHWC(0, 2, 1, 3));
  att = builder_.Reshape(att, src.tensor_desc.GetBHWCShape());
  auto out = MakeLinear(att, name + ".to_out.0", c);
  if (apply_lora) {
    auto lora_out = MakeLoraInjectedLinear(att, name + ".to_out_lora", c);
    out = builder_.Add(out, lora_out);
  }
  return out;
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeFeedForward(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  const int dim = src.tensor_desc.GetBHWCShape().c;
  const int dim_proj = dim * config_.ff_multiplier;
  const OHWI weights_shape = OHWI(dim_proj, 1, 1, dim);
  const auto attributes = ReadConvAttributes(weights_shape, name + ".0.proj");
  auto attrs = SplitAttributes(attributes);
  auto proj0 = builder_.Convolution(src, attrs.first);
  auto proj1 = builder_.Convolution(src, attrs.second);
  GpuModelBuilder::TensorHandle x;
  if (config_.activation_function == Config::ActivationFunction::kGatedSiLU) {
    auto gated_val = builder_.Elementwise(proj1, OperationType::SIGMOID);
    gated_val = builder_.Multiplication(gated_val, proj1);
    x = builder_.Multiplication(gated_val, proj0);
  } else {
    auto gated_val = builder_.Elementwise(proj1, OperationType::GELU);
    x = builder_.Multiplication(gated_val, proj0);
  }
  return MakeLinear(x, name + ".2", dim);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
UnetBuilder::MakeBasicTransformerBlock(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& context, const std::string& name,
    int num_attn_heads, bool use_self_attn) {
  auto x = src;
  if (use_self_attn) {
    auto attn1 = MakeLayerNorm(x, name + ".norm1", /*epsilon=*/1e-5f);
    ABSL_ASSIGN_OR_RETURN(attn1, MakeCrossAttention(attn1, name + ".attn1",
                                                    nullptr, num_attn_heads));
    x = builder_.Add(attn1, x);
  }
  auto attn2 = MakeLayerNorm(x, name + ".norm2", /*epsilon=*/1e-5f);
  ABSL_ASSIGN_OR_RETURN(attn2, MakeCrossAttention(attn2, name + ".attn2",
                                                  &context, num_attn_heads));
  x = builder_.Add(attn2, x);
  auto ff = MakeLayerNorm(x, name + ".norm3", /*epsilon=*/1e-5f);
  ff = MakeFeedForward(ff, name + ".ff.net");
  return builder_.Add(ff, x);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
UnetBuilder::MakeSpatialTransformerBlock(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& context, const std::string& name,
    int num_attn_heads, int transformer_blocks, bool use_self_attn) {
  auto src_shape = src.tensor_desc.GetBHWCShape();
  auto x = src;
  x = MakeGroupNorm(x, name + ".norm");
  x = MakeConv(x, name + ".proj_in", src_shape.c, 1);
  for (int i = 0; i < transformer_blocks; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        x, MakeBasicTransformerBlock(
               x, context,
               absl::StrCat(name, ".transformer_blocks.", std::to_string(i)),
               num_attn_heads, use_self_attn));
  }
  x = MakeConv(x, name + ".proj_out", src_shape.c, 1);
  return builder_.Add(x, src);
}

GpuModelBuilder::TensorHandle UnetBuilder::MakeUNetResBlock(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& emb_in, const std::string& name,
    int in_channels, int out_channels, bool use_convnext) {
  auto skip = src;
  if (in_channels != out_channels) {
    skip = MakeConv(src, name + ".skip_connection", out_channels, 1);
  }
  auto x = src;
  x = MakeGroupNorm(x, name + ".in_layers.0");
  x = builder_.SiLU(x);
  if (use_convnext) {
    auto emb = MakeLinear(emb_in, name + ".emb_layers.1", in_channels);
    x = MakeDwConv(x, name + ".dw_conv", 3);
    x = MakeLayerNorm(x, name + ".out_layers.0", /*epsilon=*/1e-6f);
    x = builder_.Add(x, emb);
    x = MakeGroupNorm(x, name + ".norm3");
    x = MakeLinear(x, name + ".pw_in_layers.2", out_channels * 4);
    x = builder_.SiLU(x);
    x = MakeLinear(x, name + ".pw_out_layers.3", out_channels);
  } else {
    auto emb = MakeLinear(emb_in, name + ".emb_layers.1", out_channels);
    x = MakeConv(x, name + ".in_layers.2", out_channels, 3);
    x = builder_.Add(x, emb);
    x = MakeGroupNorm(x, name + ".out_layers.0");
    x = builder_.SiLU(x);
    x = MakeConv(x, name + ".out_layers.3", out_channels, 3);
  }
  return builder_.Add(x, skip);
}

absl::StatusOr<GpuModelBuilder::TensorHandle> UnetBuilder::MakeUNet(
    const Config& config, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& emb,
    const GpuModelBuilder::TensorHandle& context, const std::string& name,
    const std::vector<GpuModelBuilder::TensorHandle>& plugins,
    GpuModelBuilder::TensorHandle* plugins_strength,
    std::vector<GpuModelBuilder::TensorHandle>* control, bool only_mid_control,
    GpuModelBuilder::TensorHandle* debug_tensor_ptr) {
  std::vector<GpuModelBuilder::TensorHandle> saved_inputs;
  auto x = src;

  bool resblock_updown = false;

  x = builder_.Tile(x, Axis::BATCH);

  std::vector<GpuModelBuilder::TensorHandle> tiled_plugins;
  bool use_plugins = !plugins.empty();
  if (use_plugins) {
    for (const auto& elem : plugins) {
      tiled_plugins.push_back(builder_.Tile(elem, Axis::BATCH));
    }
  }
  int ch = config.model_channels * config.channel_mult[0];
  int input_block_counter = 0;
  const std::string name_block =
      name + ".input_blocks." + std::to_string(input_block_counter++);
  x = MakeConv(x, name_block + ".0", ch, 3);

  if (!tiled_plugins.empty()) {
    auto scaled_plugin =
        builder_.Multiplication(tiled_plugins[0], *plugins_strength);
    x = builder_.Add(x, scaled_plugin);
  }
  saved_inputs.push_back(x);

  for (int level = 0; level < config.channel_mult.size(); ++level) {
    int mult = config.channel_mult[level];
    bool use_convnext =
        !config_.use_convnext.empty() && config_.use_convnext[level] == true;
    for (int block = 0; block < config.num_res_blocks[level]; ++block) {
      const std::string name_block =
          name + ".input_blocks." + std::to_string(input_block_counter++);
      x = MakeUNetResBlock(x, emb, name_block + ".0", ch,
                           mult * config.model_channels, use_convnext);
      ch = mult * config.model_channels;
      if (config.use_spatial_transformer) {
        bool use_self_attn =
            config.use_self_attn.empty() || config.use_self_attn[level] == true;
        if (!config.num_transformer_blocks[level].empty() &&
            config.num_transformer_blocks[level].size() > block) {
          int num_transformer_blocks =
              config.num_transformer_blocks[level][block];
          ABSL_ASSIGN_OR_RETURN(x, MakeSpatialTransformerBlock(
                                       x, context, name_block + ".1",
                                       config.num_attn_heads[level],
                                       num_transformer_blocks, use_self_attn));
        }
      } else {
        // AttentionBlock
      }
      saved_inputs.push_back(x);
    }
    if (level != config.channel_mult.size() - 1) {
      const std::string name_block =
          name + ".input_blocks." + std::to_string(input_block_counter++);
      if (resblock_updown) {
        // ResBlock
      } else {
        x = MakeConv(x, name_block + ".0.op", ch, 3, 2, true);
        if (use_plugins) {
          auto scaled_plugin = builder_.Multiplication(tiled_plugins[level + 1],
                                                       *plugins_strength);
          x = builder_.Add(x, scaled_plugin);
        }
        saved_inputs.push_back(x);
      }
    } else if (use_plugins) {
      auto scaled_plugin =
          builder_.Multiplication(tiled_plugins[level + 1], *plugins_strength);
      // Add the last plugin tensor to the output of the last encoding layer.
      x = builder_.Add(x, scaled_plugin);
    }
  }

  // middle blocks
  if (!config_.skip_middle_blocks) {
    bool use_convnext =
        !config_.use_convnext.empty() && config_.use_convnext.back() == true;
    x = MakeUNetResBlock(x, emb, name + ".middle_block.0", ch, ch,
                         use_convnext);
    ABSL_ASSIGN_OR_RETURN(
        x, MakeSpatialTransformerBlock(
               x, context, name + ".middle_block.1",
               config.num_attn_heads[config.num_attn_heads.size() - 1]));
    x = MakeUNetResBlock(x, emb, name + ".middle_block.2", ch, ch,
                         use_convnext);
  }

  if (control) {
    x = builder_.Add(x, control->back());
    control->pop_back();
  }

  // output blocks
  int output_block_counter = 0;
  for (int level = config.channel_mult.size() - 1; level >= 0; --level) {
    int mult = config.channel_mult[level];
    bool use_convnext =
        !config_.use_convnext.empty() && config_.use_convnext[level] == true;
    for (int block = 0; block < config.num_res_blocks[level] + 1; ++block) {
      int sub_block_id = 0;
      const std::string name_block =
          name + ".output_blocks." + std::to_string(output_block_counter++);
      GpuModelBuilder::TensorHandle tensor_to_concat = saved_inputs.back();
      saved_inputs.pop_back();
      if (!only_mid_control && control) {
        tensor_to_concat = builder_.Add(tensor_to_concat, control->back());
        control->pop_back();
      }
      if (!config_.skip_middle_blocks ||
          level != config.channel_mult.size() - 1) {
        x = builder_.Concat(x, tensor_to_concat, Axis::CHANNELS);
      }
      x = MakeUNetResBlock(x, emb,
                           name_block + "." + std::to_string(sub_block_id++),
                           x.tensor_desc.GetBHWCShape().c,
                           mult * config.model_channels, use_convnext);
      ch = mult * config.model_channels;
      if (config.use_spatial_transformer) {
        int level_idx = config.num_transformer_blocks.size() - 1 - level;
        bool use_self_attn =
            config.use_self_attn.empty() || config.use_self_attn[level] == true;
        if (!config.num_transformer_blocks[level_idx].empty() &&
            config.num_transformer_blocks[level_idx].size() > block) {
          int num_transformer_blocks =
              config.num_transformer_blocks[level_idx][block];
          ABSL_ASSIGN_OR_RETURN(x, MakeSpatialTransformerBlock(
                                       x, context, name_block + ".1",
                                       config.num_attn_heads[level],
                                       num_transformer_blocks, use_self_attn));
          ++sub_block_id;
        }
      } else {
        // AttentionBlock
      }
      if (level != 0 && block == config.num_res_blocks[level]) {
        if (resblock_updown) {
          // ResBlock
        } else {
          x = builder_.ResizeNearest(x, 2, false, true);
          x = MakeConv(
              x, name_block + "." + std::to_string(sub_block_id++) + ".conv",
              ch, 3);
        }
      }
    }
  }

  x = MakeGroupNorm(x, "model.diffusion_model.out.0");
  x = builder_.SiLU(x);
  x = MakeConv(x, "model.diffusion_model.out.2", config.out_channels, 3);
  // Points the debug tensor to the unet output tensor.
  // To check an intermediate tensor, move the debug_tensor_ptr to point to the
  // target tensor.
  if (debug_tensor_ptr) {
    *debug_tensor_ptr = x;
  }
  return x;
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
UnetBuilder::MakeControlNet(const Config& config,
                            const GpuModelBuilder::TensorHandle& src,
                            const GpuModelBuilder::TensorHandle& condition,
                            const GpuModelBuilder::TensorHandle& emb,
                            const GpuModelBuilder::TensorHandle& context,
                            const std::string& name,
                            GpuModelBuilder::TensorHandle* debug_tensor_ptr) {
  GpuModelBuilder::TensorHandle hint = condition;

  // input_hint_block
  {
    hint = MakeConv(hint, name + ".input_hint_block.0", 16, 3);
    hint = builder_.SiLU(hint);
    hint = MakeConv(hint, name + ".input_hint_block.2", 16, 3);
    hint = builder_.SiLU(hint);
    hint = MakeConv(hint, name + ".input_hint_block.4", 32, 3, 2);
    hint = builder_.SiLU(hint);
    hint = MakeConv(hint, name + ".input_hint_block.6", 32, 3);
    hint = builder_.SiLU(hint);
    hint = MakeConv(hint, name + ".input_hint_block.8", 96, 3, 2);
    hint = builder_.SiLU(hint);
    hint = MakeConv(hint, name + ".input_hint_block.10", 96, 3);
    hint = builder_.SiLU(hint);
    hint = MakeConv(hint, name + ".input_hint_block.12", 256, 3, 2);
    hint = builder_.SiLU(hint);
    hint = MakeConv(hint, name + ".input_hint_block.14", 320, 3);
  }
  hint = builder_.Tile(hint, Axis::BATCH);

  auto x = src;
  x = builder_.Tile(x, Axis::BATCH);
  std::vector<GpuModelBuilder::TensorHandle> outs;

  // input blocks
  int input_block_counter = 0;
  int zero_convs_counter = 0;
  {
    const std::string name_block =
        name + ".input_blocks." + std::to_string(input_block_counter++);
    x = MakeConv(x, name_block + ".0", 320, 3);
    x = builder_.Add(x, hint);
    // "zero conv"
    const std::string name_conv =
        name + ".zero_convs." + std::to_string(zero_convs_counter++);
    outs.push_back(MakeConv(x, name_conv + ".0", 320, 1));
  }

  int ch = config.model_channels * config.channel_mult[0];
  bool resblock_updown = false;

  for (int level = 0; level < config.channel_mult.size(); ++level) {
    int mult = config.channel_mult[level];
    for (int block = 0; block < config.num_res_blocks[level]; ++block) {
      const std::string name_block =
          name + ".input_blocks." + std::to_string(input_block_counter++);
      x = MakeUNetResBlock(x, emb, name_block + ".0", ch,
                           mult * config.model_channels);
      ch = mult * config.model_channels;
      if (config.use_spatial_transformer) {
        if (!config.num_transformer_blocks[level].empty() &&
            config.num_transformer_blocks[level].size() > block) {
          int num_transformer_blocks =
              config.num_transformer_blocks[level][block];
          ABSL_ASSIGN_OR_RETURN(
              x, MakeSpatialTransformerBlock(x, context, name_block + ".1",
                                             config.num_attn_heads[level],
                                             num_transformer_blocks));
        }
      } else {
        // AttentionBlock
      }
      // "zero conv"
      const std::string name_conv =
          name + ".zero_convs." + std::to_string(zero_convs_counter++);
      outs.push_back(MakeConv(x, name_conv + ".0", ch, 1));
    }
    if (level != config.channel_mult.size() - 1) {
      const std::string name_block =
          name + ".input_blocks." + std::to_string(input_block_counter++);
      if (resblock_updown) {
        // ResBlock
      } else {
        x = MakeConv(x, name_block + ".0.op", ch, 3, 2, true);
        // "zero conv"
        const std::string name_conv =
            name + ".zero_convs." + std::to_string(zero_convs_counter++);
        outs.push_back(MakeConv(x, name_conv + ".0", ch, 1));
      }
    }
  }
  if (!config_.skip_middle_blocks) {
    // middle blocks
    x = MakeUNetResBlock(x, emb, name + ".middle_block.0", ch, ch);
    ABSL_ASSIGN_OR_RETURN(
        x, MakeSpatialTransformerBlock(
               x, context, name + ".middle_block.1",
               config.num_attn_heads[config.num_attn_heads.size() - 1]));
    x = MakeUNetResBlock(x, emb, name + ".middle_block.2", ch, ch);
    // "zero conv"
    outs.push_back(MakeConv(x, name + ".middle_block_out.0", ch, 1));
  }
  return outs;
}

}  // namespace ml_drift
