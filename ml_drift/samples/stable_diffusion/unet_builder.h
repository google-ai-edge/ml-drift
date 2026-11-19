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

#ifndef ML_DRIFT_SAMPLES_STABLE_DIFFUSION_UNET_BUILDER_H_
#define ML_DRIFT_SAMPLES_STABLE_DIFFUSION_UNET_BUILDER_H_

#include <map>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

class UnetBuilder {
 public:
  struct Config {
    int in_channels;
    int out_channels;
    int model_channels;
    std::vector<int> num_res_blocks;
    std::vector<bool> use_convnext;
    std::vector<bool> use_self_attn;
    std::vector<int> channel_mult;
    std::vector<int> num_attn_heads;
    // Specifies the number of transformers after each resblock per input and
    // output block of the unet. The size of num_transformer_blocks must be
    // twice as the size of num_res_blocks.
    std::vector<std::vector<int>> num_transformer_blocks = {};
    bool skip_middle_blocks = false;
    bool use_spatial_transformer;
    int transformer_depth;
    int context_dim;
    bool self_attn_share_kv_proj = false;
    int ff_multiplier = 8;
    enum ActivationFunction {
      kGatedGelu = 0,
      kGatedSiLU = 1,
    };
    ActivationFunction activation_function = kGatedGelu;
    std::string unet_file_dir = "";
    int lora_rank;
    float lora_scale = 1.0;
    // Note: only one of lora_file_dir and lora_weights_layer_mapping should be
    // set.
    std::string lora_file_dir = "";
    std::map<std::string, const char*> lora_weights_layer_mapping;
  };

  // https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
  absl::Status Build(
      const Config& config, const GpuInfo& gpu_info,
      const CreateGpuModelInfo& create_info, int width, int height,
      GpuModelBuilder* absl_nonnull builder_ptr,
      GpuModelBuilder::TensorHandle* latent_ptr = nullptr,
      GpuModelBuilder::TensorHandle* temb_ptr = nullptr,
      GpuModelBuilder::TensorHandle* guidance_ptr = nullptr,
      GpuModelBuilder::TensorHandle* text_proj_ptr = nullptr,
      GpuModelBuilder::TensorHandle* masked_image_latent_ptr = nullptr,
      GpuModelBuilder::TensorHandle* eta0_ptr = nullptr,
      GpuModelBuilder::TensorHandle* eta1_ptr = nullptr,
      GpuModelBuilder::TensorHandle* debug_tensor_ptr = nullptr);

  // https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py
  absl::Status BuildControlNet(
      const Config& config, const GpuInfo& gpu_info,
      const CreateGpuModelInfo& create_info, int width, int height,
      GpuModelBuilder* absl_nonnull builder_ptr,
      GpuModelBuilder::TensorHandle* latent_ptr = nullptr,
      GpuModelBuilder::TensorHandle* condition_ptr = nullptr,
      GpuModelBuilder::TensorHandle* temb_ptr = nullptr,
      GpuModelBuilder::TensorHandle* guidance_ptr = nullptr,
      GpuModelBuilder::TensorHandle* eta0_ptr = nullptr,
      GpuModelBuilder::TensorHandle* eta1_ptr = nullptr,
      GpuModelBuilder::TensorHandle* debug_tensor_ptr = nullptr);

  absl::Status BuildUNetWithPlugins(
      const Config& config, const GpuInfo& gpu_info,
      const CreateGpuModelInfo& create_info, int width, int height,
      GpuModelBuilder* absl_nonnull builder_ptr,
      GpuModelBuilder::TensorHandle* latent_ptr = nullptr,
      std::vector<GpuModelBuilder::TensorHandle>* plugin_tensors = nullptr,
      GpuModelBuilder::TensorHandle* temb_ptr = nullptr,
      GpuModelBuilder::TensorHandle* guidance_ptr = nullptr,
      GpuModelBuilder::TensorHandle* eta0_ptr = nullptr,
      GpuModelBuilder::TensorHandle* eta1_ptr = nullptr,
      GpuModelBuilder::TensorHandle* plugins_strength_ptr = nullptr,
      GpuModelBuilder::TensorHandle* debug_tensor_ptr = nullptr);

 private:
  Convolution2DAttributes ReadConvAttributes(const OHWI& weights_shape,
                                             const std::string& name,
                                             bool bias = true,
                                             bool lora = false);

  GpuModelBuilder::TensorHandle MakeConv(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels, int kernel_hw, int stride_hw = 1, bool bias = true);

  GpuModelBuilder::TensorHandle MakeDwConv(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int kernel_hw, int stride_hw = 1, bool bias = true);

  GpuModelBuilder::TensorHandle MakeLinear(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels, bool bias = true, bool lora = false);
  GpuModelBuilder::TensorHandle MakeLoraInjectedLinear(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels);
  GpuModelBuilder::TensorHandle MakeGroupNorm(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);
  GpuModelBuilder::TensorHandle MakeLayerNorm(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      float epsilon);

  GpuModelBuilder::TensorHandle MakeTimeEmbed(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);

  GpuModelBuilder::TensorHandle MakePooledTextProjection(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_dim_size);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeCrossAttention(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      const GpuModelBuilder::TensorHandle* context_ptr, int num_attn_heads);

  GpuModelBuilder::TensorHandle MakeFeedForward(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeBasicTransformerBlock(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& context, const std::string& name,
      int num_attn_heads, bool use_self_attn);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeSpatialTransformerBlock(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& context, const std::string& name,
      int num_attn_heads, int transformer_blocks = 1,
      bool use_self_attn = true);

  GpuModelBuilder::TensorHandle MakeUNetResBlock(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& emb_in, const std::string& name,
      int in_channels, int out_channels, bool use_convnext = false);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeUNet(
      const Config& config, const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& emb,
      const GpuModelBuilder::TensorHandle& context, const std::string& name,
      const std::vector<GpuModelBuilder::TensorHandle>& plugins = {},
      GpuModelBuilder::TensorHandle* plugins_strength = nullptr,
      std::vector<GpuModelBuilder::TensorHandle>* control = nullptr,
      bool only_mid_control = false,
      GpuModelBuilder::TensorHandle* debug_tensor_ptr = nullptr);

  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>> MakeControlNet(
      const Config& config, const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& condition,
      const GpuModelBuilder::TensorHandle& emb,
      const GpuModelBuilder::TensorHandle& context, const std::string& name,
      GpuModelBuilder::TensorHandle* debug_tensor_ptr = nullptr);

  Config config_;
  GpuModelBuilder* builder_ptr_;
  GpuInfo gpu_info_;
};
}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_STABLE_DIFFUSION_UNET_BUILDER_H_
