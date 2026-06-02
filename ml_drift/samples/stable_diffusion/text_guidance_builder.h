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

#ifndef ML_DRIFT_SAMPLES_STABLE_DIFFUSION_TEXT_GUIDANCE_BUILDER_H_
#define ML_DRIFT_SAMPLES_STABLE_DIFFUSION_TEXT_GUIDANCE_BUILDER_H_

#include <memory>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/status.h"
#include "ml_drift/samples/stable_diffusion/model_data_loader.h"

namespace ml_drift {

class TextGuidanceBuilder {
 public:
  struct Config {
    std::string file_folder;
    int embedding_size;
    int num_layers;
    int num_heads;
    bool skip_final_layer_norm = false;
  };

  absl::Status Build(const Config& config, const GpuInfo& gpu_info,
                     const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
                     GpuModelBuilder::TensorHandle* src_ptr = nullptr,
                     GpuModelBuilder::TensorHandle* mask_ptr = nullptr,
                     GpuModelBuilder::TensorHandle* dst_ptr = nullptr,
                     GpuModelBuilder::TensorHandle* text_proj_ptr = nullptr);

 private:
  GpuModelBuilder::TensorHandle MakeLinear(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels, bool bias = true);
  GpuModelBuilder::TensorHandle MakeLayerNorm(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeTextGuidance(
      const GpuModelBuilder::TensorHandle& x_in, DataType float_type);

  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  MakeTextGuidanceWithTextProjection(const GpuModelBuilder::TensorHandle& x_in,
                                     DataType float_type);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeTextEncoderLayer(
      const GpuModelBuilder::TensorHandle& xIn, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeTextEncoder(
      const GpuModelBuilder::TensorHandle& xIn, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeTextEmbeddings(
      const GpuModelBuilder::TensorHandle& x_in, DataType float_type,
      const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeTextAttention(
      const GpuModelBuilder::TensorHandle& xIn, const std::string& name);

  GpuModelBuilder::TensorHandle AddCustomGather(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& indices);

  GpuModelBuilder builder_;
  Config config_;
  GpuModelBuilder::TensorHandle mask_;
  std::unique_ptr<ModelDataLoader> model_data_loader_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_STABLE_DIFFUSION_TEXT_GUIDANCE_BUILDER_H_
