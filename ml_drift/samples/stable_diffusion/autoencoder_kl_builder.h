// Copyright 2023 The ML Drift Authors.
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

#ifndef ML_DRIFT_SAMPLES_STABLE_DIFFUSION_AUTOENCODER_KL_BUILDER_H_
#define ML_DRIFT_SAMPLES_STABLE_DIFFUSION_AUTOENCODER_KL_BUILDER_H_

#include <string>
#include <vector>

#include "ml_drift/common/gpu_model_builder.h"

namespace ml_drift {

class AutoencoderKLBuilder {
 public:
  // Note: resolution, in_channels, and out_ch aren't used.  When they are,
  // please update:
  // http://google3/experimental/mediapipe/stable_diffusion/cc/stable_diffusion_iterate_calculator.cc
  struct Config {
    bool double_z;
    int z_channels;
    int resolution;
    int in_channels;
    int out_ch;
    int ch;
    std::vector<int> ch_mult;
    int num_res_blocks;
  };
  // Decoder for transforming tensor from latent space to image
  // https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
  absl::Status BuildDecoder(const Config& config, const GpuInfo& gpu_info,
                            const CreateGpuModelInfo& create_info, int width,
                            int height, GpuModel* gpu_model,
                            GpuModelBuilder::TensorHandle* src_ptr = nullptr,
                            GpuModelBuilder::TensorHandle* dst_ptr = nullptr,
                            const std::string& file_folder = "");

  // Encoder for transforming image to tensor in latent space
  // https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
  absl::Status BuildEncoder(const Config& config, const GpuInfo& gpu_info,
                            const CreateGpuModelInfo& create_info, int width,
                            int height, GpuModel* gpu_model,
                            GpuModelBuilder::TensorHandle* src_ptr = nullptr,
                            GpuModelBuilder::TensorHandle* dst_ptr = nullptr,
                            const std::string& file_folder = "");

 private:
  GpuModelBuilder::TensorHandle MakeConv(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels, int kernel_hw, int stride_hw = 1, bool bias = true);
  GpuModelBuilder::TensorHandle MakeLinear(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels, bool bias = true);
  GpuModelBuilder::TensorHandle MakeGroupNorm(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);
  GpuModelBuilder::TensorHandle Upsample(
      const GpuModelBuilder::TensorHandle& src, bool with_conv = false,
      const std::string& name = "");
  GpuModelBuilder::TensorHandle PixelShuffle(
      const GpuModelBuilder::TensorHandle& src, int block_size,
      const std::string& name = "");
  GpuModelBuilder::TensorHandle Downsample(
      const GpuModelBuilder::TensorHandle& src, bool with_conv = false,
      const std::string& name = "");
  GpuModelBuilder::TensorHandle MakeResnetBlock(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeAttention(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeDecoder(
      const Config& config, const GpuModelBuilder::TensorHandle& src);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeEncoder(
      const Config& config, const GpuModelBuilder::TensorHandle& src);

  GpuModelBuilder builder_;
  GpuInfo gpu_info_;
  std::string file_folder_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_STABLE_DIFFUSION_AUTOENCODER_KL_BUILDER_H_
