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

#ifndef THIRD_PARTY_ML_DRIFT_SAMPLES_LLM_GEMMA3_MODEL_BUILDER_H_
#define THIRD_PARTY_ML_DRIFT_SAMPLES_LLM_GEMMA3_MODEL_BUILDER_H_

#include <memory>

#include "absl/status/status.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/samples/llm/llm_config.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"

namespace ml_drift {

class Gemma3ModelBuilder {
 public:
  Gemma3ModelBuilder(const LlmConfig& config, const GpuInfo& gpu_info,
                     CreateGpuModelInfo& create_info,
                     std::unique_ptr<LlmTensorLoader> tensor_loader);

  absl::Status Build(
      GpuModel* prefill_model, GpuModel* decode_model,
      GpuModelBuilder::TensorHandle* input_handle,
      GpuModelBuilder::TensorHandle* decode_input_handle,
      GpuModelBuilder::TensorHandle* params_i32_handle,
      GpuModelBuilder::TensorHandle* output_handle,
      GpuModelBuilder::TensorHandle* decode_output_handle,
      std::vector<GpuModelBuilder::TensorHandle>* k_caches_in_handles,
      std::vector<GpuModelBuilder::TensorHandle>* v_caches_in_handles);

  // Runs greedy post-processing on the logits to generate the next token.
  absl::Status BuildPostProcessGreedy(
      GpuModel* gpu_model, GpuModelBuilder::TensorHandle* input_logits_handle,
      GpuModelBuilder::TensorHandle* output_token_handle);

 private:
  absl::Status BuildModelInternal(
      int sequence_size, GpuModel* gpu_model,
      const GpuModelBuilder::TensorHandle& params_i32_handle,
      const std::vector<GpuModelBuilder::TensorHandle>& k_caches_in_handles,
      const std::vector<GpuModelBuilder::TensorHandle>& v_caches_in_handles,
      GpuModelBuilder::TensorHandle* input_handle,
      GpuModelBuilder::TensorHandle* output_handle);

  void CreateKVCache(
      std::vector<GpuModelBuilder::TensorHandle>* k_caches_in_handles,
      std::vector<GpuModelBuilder::TensorHandle>* v_caches_in_handles);

  LlmConfig config_;
  GpuInfo gpu_info_;
  CreateGpuModelInfo& create_info_;
  std::unique_ptr<LlmTensorLoader> tensor_loader_;
  GpuModelBuilder model_builder_;
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ML_DRIFT_SAMPLES_LLM_GEMMA3_MODEL_BUILDER_H_
