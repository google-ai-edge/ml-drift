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

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/functional/bind_front.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/samples/llm/gemma3_model_builder.h"
#include "ml_drift/samples/llm/gemma4_model_builder.h"
#include "ml_drift/samples/llm/llm_config.h"
#include "ml_drift/samples/llm/llm_file_tensor_loader.h"
#include "ml_drift/samples/llm/llm_runner.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"
#include "ml_drift/samples/llm/qwen3_model_builder.h"
#include "ml_drift/webgpu/compute_task.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/inference_context.h"
#include "ml_drift/webgpu/instance.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"

ABSL_DECLARE_FLAG(std::string, model);

namespace ml_drift {

using ::ml_drift::webgpu::ComputeTask;
using ::ml_drift::webgpu::Environment;
using ::ml_drift::webgpu::InferenceContext;
using ::ml_drift::webgpu::SpatialTensor;

namespace {

absl::StatusOr<SpatialTensor*> CreateTensor(
    Environment* env, LlmTensorLoader* tensor_loader,
    std::vector<std::unique_ptr<SpatialTensor>>* model_tensors,
    const TensorDescriptor& tensor_desc) {
  SpatialTensor& tensor =
      *model_tensors->emplace_back(std::make_unique<SpatialTensor>());
  env->device().Tick();
  ABSL_RETURN_IF_ERROR(tensor.CreateFromDescriptor(env->device(), tensor_desc));
  env->device().Tick();
  return &tensor;
}

absl::Status ReadTokenFromTensor(const wgpu::Device& device,
                                 const wgpu::Queue& queue,
                                 const SpatialTensor& tensor, int* token) {
  TensorStorageType storage_type = tensor.GetStorageType();
  if (storage_type == TensorStorageType::kBuffer ||
      storage_type == TensorStorageType::kImageBuffer) {
    return webgpu::ReadDataFromBuffer(device, queue, tensor.GetBufferHandle(),
                                      sizeof(int), token);
  } else {
    // Treat as Texture. Use bytes_per_pixel = 4 for INT32.
    return webgpu::ReadDataFromTexture(device, queue, tensor.GetTextureHandle(),
                                       /*bytes_per_pixel=*/sizeof(int),
                                       /*width=*/1, /*height=*/1, /*depth=*/1,
                                       token);
  }
}

}  // namespace

class WebGpuLlmModelRunner : public LlmModelRunner {
 public:
  struct DecodeOptions {
    int start_token_offset = 0;
    int num_tokens = 0;
    bool download_token = true;
    bool check_eos = true;
    Tokenizer* tokenizer = nullptr;
    std::vector<int>* output_tokens = nullptr;
    std::vector<double>* itl_ms = nullptr;
  };

  absl::Status Init(const std::string& weights_path, int prefill_sequence_size,
                    int context_size, bool use_fp32 = false) override;
  absl::Status InitSliceAndGreedy() override;
  absl::Status PostProcessGreedy(Tokenizer& tokenizer,
                                 const std::vector<int>& input_tokens_vec,
                                 int num_gen_tokens,
                                 std::vector<int>& output_tokens_vec) override;
  absl::Status PostProcessGreedyBench(const std::vector<int>& input_tokens_vec,
                                      int num_gen_tokens) override;

  absl::Status Prefill(const std::vector<int>& input_tokens_vec,
                       bool download_token = true, int* output_token = nullptr);
  absl::Status Decode(const DecodeOptions& options);

 private:
  absl::Status EnsureParamBuffers(int count);
  absl::Status UpdateParams(const LlmRuntimeParams& params,
                            int buffer_index = 0);

  std::unique_ptr<Environment> env_;
  CreateGpuModelInfo create_info_;
  LlmConfig config_;
  ModelType model_type_;
  std::vector<std::unique_ptr<SpatialTensor>> model_tensors_;

  // Prefill and decode
  std::unique_ptr<InferenceContext> prefill_context_;
  std::unique_ptr<InferenceContext> decode_context_;
  GpuModelBuilder::TensorHandle input_handle_;
  GpuModelBuilder::TensorHandle decode_input_handle_;
  GpuModelBuilder::TensorHandle params_i32_handle_;
  GpuModelBuilder::TensorHandle output_logits_handle_;
  GpuModelBuilder::TensorHandle decode_output_logits_handle_;
  std::unique_ptr<SpatialTensor> input_tokens_;
  std::unique_ptr<SpatialTensor> decode_input_tokens_;
  std::vector<std::unique_ptr<SpatialTensor>> params_i32_;
  std::unique_ptr<SpatialTensor> output_logits_;
  std::unique_ptr<SpatialTensor> decode_output_logits_;

  std::vector<GpuModelBuilder::TensorHandle> k_caches_in_handles_;
  std::vector<GpuModelBuilder::TensorHandle> v_caches_in_handles_;
  std::vector<std::unique_ptr<SpatialTensor>> kv_cache_tensors_;

  // Post-processing
  std::unique_ptr<InferenceContext> slice_context_;
  GpuModelBuilder::TensorHandle slice_input_handle_;
  GpuModelBuilder::TensorHandle slice_output_handle_;
  std::unique_ptr<InferenceContext> greedy_context_;
  GpuModelBuilder::TensorHandle greedy_input_logits_handle_;
  GpuModelBuilder::TensorHandle greedy_output_token_handle_;
};

absl::Status WebGpuLlmModelRunner::Init(const std::string& weights_path,
                                        int prefill_sequence_size,
                                        int context_size, bool use_fp32) {
  env_ = std::make_unique<Environment>();
  ABSL_RETURN_IF_ERROR(env_->Initialize());
  const auto& gpu_info = env_->GetInfo();

  // GPU model configuration.
  create_info_.precision =
      use_fp32 ? CalculationsPrecision::kF32 : CalculationsPrecision::kF16;
  create_info_.storage_type = webgpu::GetFastestStorageType(gpu_info);
  create_info_.hints.Add(ModelHints::kFastTuning);
  create_info_.hints.Add(ModelHints::kDisallow8bitConvs);
  create_info_.hints.Add(ModelHints::kEnableHostMappedPointer);

  if (create_info_.precision == ml_drift::CalculationsPrecision::kF16 ||
      create_info_.precision == ml_drift::CalculationsPrecision::kF32F16) {
    env_->RequestExtension("shader_f16");
  }

  // Load model config.
  auto model_info = GetModelInfo(absl::GetFlag(FLAGS_model));
  if (!model_info.ok()) {
    return model_info.status();
  }
  config_ = model_info->config;
  model_type_ = model_info->type;
  // kMaxPrefillChunkSize caps config_.sequence_size (the compiled prefill
  // size). Prefill() uses chunk_size = config_.sequence_size to process longer
  // inputs in chunks.
  constexpr int kMaxPrefillChunkSize = 1024;
  config_.sequence_size = std::min(prefill_sequence_size, kMaxPrefillChunkSize);
  config_.cache_size = context_size;

  // Cached tensor loading for prefill/decode models.
  auto tensor_loader = LlmTensorLoader::MakeCaching(
      std::make_unique<LlmFileTensorLoader>(weights_path));
  tensor_loader->SetCreateTensorFn(absl::bind_front(
      &CreateTensor, env_.get(), tensor_loader.get(), &model_tensors_));

  ABSL_LOG(INFO) << "Building prefill/decode models.";
  GpuModel prefill_model;
  GpuModel decode_model;
  switch (model_type_) {
    case ModelType::kQwen3: {
      Qwen3ModelBuilder builder(config_, gpu_info, create_info_,
                                std::move(tensor_loader));
      ABSL_RETURN_IF_ERROR(
          builder.Build(&prefill_model, &decode_model, &input_handle_,
                        &decode_input_handle_, &params_i32_handle_,
                        &output_logits_handle_, &decode_output_logits_handle_,
                        &k_caches_in_handles_, &v_caches_in_handles_));
      break;
    }
    case ModelType::kGemma3: {
      Gemma3ModelBuilder builder(config_, gpu_info, create_info_,
                                 std::move(tensor_loader));
      ABSL_RETURN_IF_ERROR(
          builder.Build(&prefill_model, &decode_model, &input_handle_,
                        &decode_input_handle_, &params_i32_handle_,
                        &output_logits_handle_, &decode_output_logits_handle_,
                        &k_caches_in_handles_, &v_caches_in_handles_));
      break;
    }
    case ModelType::kGemma4: {
      Gemma4ModelBuilder builder(config_, gpu_info, create_info_,
                                 std::move(tensor_loader));
      ABSL_RETURN_IF_ERROR(
          builder.Build(&prefill_model, &decode_model, &input_handle_,
                        &decode_input_handle_, &params_i32_handle_,
                        &output_logits_handle_, &decode_output_logits_handle_,
                        &k_caches_in_handles_, &v_caches_in_handles_));
      break;
    }
  }

  // Map tensor storage.
  input_tokens_ = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(input_tokens_->CreateFromDescriptor(
      env_->device(), input_handle_.tensor_desc));
  decode_input_tokens_ = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(decode_input_tokens_->CreateFromDescriptor(
      env_->device(), decode_input_handle_.tensor_desc));
  ABSL_RETURN_IF_ERROR(EnsureParamBuffers(2));
  output_logits_ = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(output_logits_->CreateFromDescriptor(
      env_->device(), output_logits_handle_.tensor_desc));
  decode_output_logits_ = std::make_unique<SpatialTensor>();
  ABSL_RETURN_IF_ERROR(decode_output_logits_->CreateFromDescriptor(
      env_->device(), decode_output_logits_handle_.tensor_desc));

  // Create KV cache tensors.
  ABSL_LOG(INFO) << "Creating KV cache.";
  kv_cache_tensors_.resize(config_.stack_size * 2);
  const int v_offset = config_.stack_size;
  for (int i = 0; i < config_.stack_size; ++i) {
    kv_cache_tensors_[i] = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(kv_cache_tensors_[i]->CreateFromDescriptor(
        env_->device(), k_caches_in_handles_[i].tensor_desc));
    kv_cache_tensors_[i + v_offset] = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(kv_cache_tensors_[i + v_offset]->CreateFromDescriptor(
        env_->device(), v_caches_in_handles_[i].tensor_desc));
  }

  CreateGpuModelInfo prefill_create_info = create_info_;
  prefill_create_info.external_immutable_tensors[input_handle_.id] =
      input_tokens_.get();
  prefill_create_info.external_immutable_tensors[params_i32_handle_.id] =
      params_i32_[0].get();
  prefill_create_info.external_immutable_tensors[output_logits_handle_.id] =
      output_logits_.get();
  for (int i = 0; i < config_.stack_size; ++i) {
    prefill_create_info.external_immutable_tensors[k_caches_in_handles_[i].id] =
        kv_cache_tensors_[i].get();
    prefill_create_info.external_immutable_tensors[v_caches_in_handles_[i].id] =
        kv_cache_tensors_[i + v_offset].get();
  }

  CreateGpuModelInfo decode_create_info = create_info_;
  decode_create_info.external_immutable_tensors[decode_input_handle_.id] =
      decode_input_tokens_.get();
  decode_create_info.external_mutable_tensors[params_i32_handle_.id] =
      params_i32_handle_.tensor_desc;
  decode_create_info
      .external_immutable_tensors[decode_output_logits_handle_.id] =
      decode_output_logits_.get();
  for (int i = 0; i < config_.stack_size; ++i) {
    decode_create_info.external_immutable_tensors[k_caches_in_handles_[i].id] =
        kv_cache_tensors_[i].get();
    decode_create_info.external_immutable_tensors[v_caches_in_handles_[i].id] =
        kv_cache_tensors_[i + v_offset].get();
  }

  ABSL_LOG(INFO) << "Initializing inference contexts.";
  prefill_context_ = std::make_unique<InferenceContext>();
  ABSL_RETURN_IF_ERROR(prefill_context_->InitFromGpuModel(
      *env_, prefill_create_info, &prefill_model));
  decode_context_ = std::make_unique<InferenceContext>();
  ABSL_RETURN_IF_ERROR(decode_context_->InitFromGpuModel(
      *env_, decode_create_info, &decode_model));
  ABSL_RETURN_IF_ERROR(
      decode_context_->SetTensor(params_i32_handle_.id, params_i32_[0].get()));
  ABSL_LOG(INFO) << "Inference contexts initialized.";

  return absl::OkStatus();
}

absl::Status WebGpuLlmModelRunner::EnsureParamBuffers(int count) {
  while (params_i32_.size() < count) {
    auto tensor = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(tensor->CreateFromDescriptor(
        env_->device(), params_i32_handle_.tensor_desc));
    params_i32_.push_back(std::move(tensor));
  }
  return absl::OkStatus();
}

absl::Status WebGpuLlmModelRunner::UpdateParams(const LlmRuntimeParams& params,
                                                int buffer_index) {
  if (buffer_index < 0 || buffer_index >= params_i32_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("buffer_index out of range: ", buffer_index));
  }
  int data[LlmRuntimeParams::kTotalParamsCount] = {0};
  data[LlmRuntimeParams::kTokenOffsetIndex] = params.token_index_offset;
  data[LlmRuntimeParams::kActiveTokensIndex] = params.active_tokens;
  const int ch_alignment = ConvRuntimeCheckDesc::kChannelsAlignment;
  data[LlmRuntimeParams::kActiveTokensAlignedIndex] = std::min(
      config_.cache_size, AlignByN(params.active_tokens, ch_alignment));
  data[LlmRuntimeParams::kRingOffsetIndex] =
      params.token_index_offset + GetRingOffset(config_);
  return params_i32_[buffer_index]->WriteData(env_->queue(), data);
}

absl::Status WebGpuLlmModelRunner::InitSliceAndGreedy() {
  ABSL_LOG(INFO) << "Building post-processing models.";
  GpuModel slice_model;   // Slice: Processes the last slice of prefill logits.
  GpuModel greedy_model;  // Greedy: Processes the decode output logits.

  switch (model_type_) {
    case ModelType::kQwen3: {
      Qwen3ModelBuilder builder(config_, env_->GetInfo(), create_info_,
                                nullptr);
      ABSL_RETURN_IF_ERROR(builder.BuildPostProcessGreedy(
          &slice_model, &slice_input_handle_, &slice_output_handle_));
      ABSL_RETURN_IF_ERROR(builder.BuildPostProcessGreedy(
          &greedy_model, &greedy_input_logits_handle_,
          &greedy_output_token_handle_));
      break;
    }
    case ModelType::kGemma3: {
      Gemma3ModelBuilder builder(config_, env_->GetInfo(), create_info_,
                                 nullptr);
      ABSL_RETURN_IF_ERROR(builder.BuildPostProcessGreedy(
          &slice_model, &slice_input_handle_, &slice_output_handle_));
      ABSL_RETURN_IF_ERROR(builder.BuildPostProcessGreedy(
          &greedy_model, &greedy_input_logits_handle_,
          &greedy_output_token_handle_));
      break;
    }
    case ModelType::kGemma4: {
      Gemma4ModelBuilder builder(config_, env_->GetInfo(), create_info_,
                                 nullptr);
      ABSL_RETURN_IF_ERROR(builder.BuildPostProcessGreedy(
          &slice_model, &slice_input_handle_, &slice_output_handle_));
      ABSL_RETURN_IF_ERROR(builder.BuildPostProcessGreedy(
          &greedy_model, &greedy_input_logits_handle_,
          &greedy_output_token_handle_));
      break;
    }
  }

  // Slice model tensor connections:
  // Reads prefill logits and writes the first selected token directly into
  // decode_input_tokens_ on GPU for zero-copy reuse by the first decode step.
  CreateGpuModelInfo slice_create_info = create_info_;
  slice_create_info.external_mutable_tensors.clear();
  slice_create_info.external_immutable_tensors.clear();
  slice_create_info.external_immutable_tensors[slice_input_handle_.id] =
      output_logits_.get();  // Prefill output logits --> Slice input
  slice_create_info.external_immutable_tensors[slice_output_handle_.id] =
      decode_input_tokens_.get();  // Slice output token --> Decode input
  slice_context_ = std::make_unique<InferenceContext>();
  ABSL_RETURN_IF_ERROR(
      slice_context_->InitFromGpuModel(*env_, slice_create_info, &slice_model));

  // Greedy model tensor connections:
  // Reads decode logits and writes the next selected token directly into
  // decode_input_tokens_ on GPU for zero-copy reuse by the next decode step.
  CreateGpuModelInfo greedy_create_info = create_info_;
  greedy_create_info.external_mutable_tensors.clear();
  greedy_create_info.external_immutable_tensors.clear();
  greedy_create_info
      .external_immutable_tensors[greedy_input_logits_handle_.id] =
      decode_output_logits_.get();  // Decode output logits --> Greedy input
  greedy_create_info
      .external_immutable_tensors[greedy_output_token_handle_.id] =
      decode_input_tokens_.get();  // Greedy output token --> Decode input
  greedy_context_ = std::make_unique<InferenceContext>();
  ABSL_RETURN_IF_ERROR(greedy_context_->InitFromGpuModel(
      *env_, greedy_create_info, &greedy_model));

  return absl::OkStatus();
}

absl::Status WebGpuLlmModelRunner::Prefill(
    const std::vector<int>& input_tokens_vec, bool download_token,
    int* output_token) {
  const int total_tokens = input_tokens_vec.size();
  const int chunk_size = config_.sequence_size;

  if (total_tokens < chunk_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input tokens size (", total_tokens,
        ") is less than prefill sequence size (", chunk_size, ")."));
  }

  const absl::Duration timeout = absl::Seconds(15);

  // Fast path: single chunk execution when total_tokens == chunk_size.
  if (total_tokens == chunk_size) {
    ml_drift::LlmRuntimeParams prefill_params;
    prefill_params.token_index_offset = 0;
    prefill_params.active_tokens = chunk_size;
    ABSL_RETURN_IF_ERROR(UpdateParams(prefill_params));

    Tensor<BHWC, DataType::kInt32> input_t;
    input_t.shape = BHWC(config_.batch_size, 1, chunk_size, 1);
    input_t.data = input_tokens_vec;
    ABSL_RETURN_IF_ERROR(
        prefill_context_->SetInputTensor(*env_, input_handle_.id, input_t));
    ABSL_RETURN_IF_ERROR(prefill_context_->Execute(*env_));

    ABSL_RETURN_IF_ERROR(slice_context_->Execute(*env_));
    ABSL_RETURN_IF_ERROR(
        webgpu::WaitUntilCompleted(env_->queue(), env_->device(), timeout));

    if (download_token && output_token != nullptr) {
      ABSL_RETURN_IF_ERROR(ReadTokenFromTensor(
          env_->device(), env_->queue(), *decode_input_tokens_, output_token));
    }
    return absl::OkStatus();
  }

  // Chunked path for total_tokens > chunk_size (e.g. S > kMaxPrefillChunkSize).
  Tensor<BHWC, DataType::kInt32> input_t;
  input_t.shape = BHWC(config_.batch_size, 1, chunk_size, 1);
  input_t.data.resize(chunk_size);

  int offset = 0;
  while (offset < total_tokens) {
    int chunk_offset = offset;
    int cur_len = total_tokens - offset;

    // If the remaining tokens are fewer than chunk_size, align to the end
    // of the input sequence. This avoids padding tokens, guarantees valid
    // KV cache entries, and ensures index (chunk_size - 1) is the exact
    // final prompt token.
    if (cur_len < chunk_size) {
      chunk_offset = total_tokens - chunk_size;
    }

    std::copy(input_tokens_vec.begin() + chunk_offset,
              input_tokens_vec.begin() + chunk_offset + chunk_size,
              input_t.data.begin());

    const int active_tokens = std::min(total_tokens, chunk_offset + chunk_size);
    const bool is_last_chunk = (chunk_offset + chunk_size >= total_tokens);

    ml_drift::LlmRuntimeParams prefill_params;
    prefill_params.token_index_offset = chunk_offset;
    prefill_params.active_tokens = active_tokens;
    ABSL_RETURN_IF_ERROR(UpdateParams(prefill_params));

    ABSL_RETURN_IF_ERROR(
        prefill_context_->SetInputTensor(*env_, input_handle_.id, input_t));
    ABSL_RETURN_IF_ERROR(prefill_context_->Execute(*env_));

    if (is_last_chunk) {
      ABSL_RETURN_IF_ERROR(slice_context_->Execute(*env_));
    }
    ABSL_RETURN_IF_ERROR(
        webgpu::WaitUntilCompleted(env_->queue(), env_->device(), timeout));

    if (is_last_chunk) {
      break;
    }
    offset += chunk_size;
  }

  if (download_token && output_token != nullptr) {
    ABSL_RETURN_IF_ERROR(ReadTokenFromTensor(
        env_->device(), env_->queue(), *decode_input_tokens_, output_token));
  }
  return absl::OkStatus();
}

absl::Status WebGpuLlmModelRunner::Decode(const DecodeOptions& options) {
  if (options.num_tokens <= 0) {
    return absl::OkStatus();
  }

  const absl::Duration timeout = absl::Seconds(15);
  ABSL_RETURN_IF_ERROR(EnsureParamBuffers(2));

  struct InflightStep {
    wgpu::Future future;
    wgpu::QueueWorkDoneStatus status = wgpu::QueueWorkDoneStatus::Success;
    bool is_active = false;
  };
  InflightStep inflight_steps[2];

  auto dispatch_step = [&](int step_idx) -> absl::Status {
    int buf_idx = step_idx % 2;
    int current_length = options.start_token_offset + step_idx;
    ml_drift::LlmRuntimeParams decode_params;
    decode_params.token_index_offset = current_length;
    decode_params.active_tokens = current_length + 1;
    ABSL_RETURN_IF_ERROR(UpdateParams(decode_params, buf_idx));
    ABSL_RETURN_IF_ERROR(decode_context_->SetTensor(
        params_i32_handle_.id, params_i32_[buf_idx].get()));

    ABSL_RETURN_IF_ERROR(decode_context_->Execute(*env_));
    ABSL_RETURN_IF_ERROR(greedy_context_->Execute(*env_));

    InflightStep& step_info = inflight_steps[buf_idx];
    step_info.future = env_->queue().OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [&step_info](wgpu::QueueWorkDoneStatus s, wgpu::StringView) {
          step_info.status = s;
        });
    step_info.is_active = true;
    return absl::OkStatus();
  };

  if (options.itl_ms != nullptr) {
    options.itl_ms->reserve(options.num_tokens);
  }

  // Prime the pipeline: dispatch decode step 0.
  ABSL_RETURN_IF_ERROR(dispatch_step(0));
  // Pre-dispatch decode step 1 so the GPU never idles when step 0 completes.
  if (options.num_tokens > 1) {
    ABSL_RETURN_IF_ERROR(dispatch_step(1));
  }

  auto prev_token_time = absl::Now();

  for (int step = 0; step < options.num_tokens; ++step) {
    int buf_idx = step % 2;
    InflightStep& step_info = inflight_steps[buf_idx];
    if (step_info.is_active) {
      ABSL_RETURN_IF_ERROR(
          webgpu::Instance::Wait(env_->device(), step_info.future, timeout));
      ABSL_RETURN_IF_ERROR(webgpu::VerifyQueueWorkDoneStatus(step_info.status));
      step_info.is_active = false;
    }

    if (options.itl_ms != nullptr) {
      auto cur_time = absl::Now();
      options.itl_ms->push_back(
          absl::ToDoubleMilliseconds(cur_time - prev_token_time));
      prev_token_time = cur_time;
    }

    int current_token = 0;
    if (options.download_token) {
      ABSL_RETURN_IF_ERROR(ReadTokenFromTensor(env_->device(), env_->queue(),
                                               *decode_input_tokens_,
                                               &current_token));

      if (options.check_eos && options.tokenizer != nullptr) {
        if (current_token == options.tokenizer->eos_id() &&
            (options.output_tokens == nullptr ||
             options.output_tokens->size() > 1)) {
          break;
        }
        if ((model_type_ == ModelType::kGemma3 ||
             model_type_ == ModelType::kGemma4) &&
            current_token == options.tokenizer->end_of_turn_id()) {
          break;
        }
      }

      if (options.output_tokens != nullptr) {
        options.output_tokens->push_back(current_token);
      }
    }

    // Buffer buf_idx is now free; pre-dispatch step + 2 to keep the GPU busy.
    if (step + 2 < options.num_tokens) {
      ABSL_RETURN_IF_ERROR(dispatch_step(step + 2));
    }

    if (options.download_token && options.tokenizer != nullptr) {
      std::cout << options.tokenizer->DecodeOutputTokens({current_token})
                << std::flush;
    }
  }

  // Drain any remaining GPU work.
  ABSL_RETURN_IF_ERROR(
      webgpu::WaitUntilCompleted(env_->queue(), env_->device(), timeout));

  // Restore decode_context_ to use params_i32_[0].
  ABSL_RETURN_IF_ERROR(
      decode_context_->SetTensor(params_i32_handle_.id, params_i32_[0].get()));

  return absl::OkStatus();
}

absl::Status WebGpuLlmModelRunner::PostProcessGreedy(
    Tokenizer& tokenizer, const std::vector<int>& input_tokens_vec,
    int num_gen_tokens, std::vector<int>& output_tokens_vec) {
  // 1. Prefill Step: Process all prompt input tokens on the GPU.
  ABSL_LOG(INFO) << "Running prefill...";
  int first_token = 0;
  ABSL_RETURN_IF_ERROR(
      Prefill(input_tokens_vec, /*download_token=*/true, &first_token))
      << "Error during prefill";
  output_tokens_vec.push_back(first_token);

  if (num_gen_tokens <= 1) {
    std::cout << tokenizer.DecodeOutputTokens({first_token}) << std::flush;
    std::cout << std::endl;
    return absl::OkStatus();
  }

  // 2. Decode Loop: Iteratively run Decode + Greedy on GPU with asynchronous
  // double-buffered pipelining to overlap CPU de-tokenization, EOS checks, and
  // terminal I/O with GPU transformer execution.
  ABSL_LOG(INFO) << "Running decode..." << std::flush;
  std::cout << tokenizer.DecodeOutputTokens({first_token}) << std::flush;
  DecodeOptions options;
  options.start_token_offset = input_tokens_vec.size();
  options.num_tokens = num_gen_tokens - 1;
  options.download_token = true;
  options.check_eos = true;
  options.tokenizer = &tokenizer;
  options.output_tokens = &output_tokens_vec;
  ABSL_RETURN_IF_ERROR(Decode(options));
  std::cout << std::endl;

  return absl::OkStatus();
}

absl::Status WebGpuLlmModelRunner::PostProcessGreedyBench(
    const std::vector<int>& input_tokens_vec, int num_gen_tokens) {
  if (input_tokens_vec.size() + num_gen_tokens > config_.cache_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Requested tokens exceed KV cache capacity: ",
        input_tokens_vec.size() + num_gen_tokens, " > ", config_.cache_size));
  }
  ABSL_LOG(INFO) << "Running LLM benchmark (context size: "
                 << config_.cache_size << ")...";

  // 1. Prefill Warmup
  ABSL_LOG(INFO) << "Warming up prefill...";
  ABSL_RETURN_IF_ERROR(Prefill(input_tokens_vec, /*download_token=*/false))
      << "Error during prefill warmup";

  // Prefill Benchmark (averaged over N runs, more runs for smaller input)
  const int kPrefillRuns = std::max(1, 1024 / (int)input_tokens_vec.size());
  ABSL_LOG(INFO) << "Running prefill benchmark (" << kPrefillRuns << " runs, "
                 << input_tokens_vec.size() << " tokens)...";
  double total_prefill_duration_s = 0.0;
  for (int i = 0; i < kPrefillRuns; ++i) {
    auto p_start_time = absl::Now();
    ABSL_RETURN_IF_ERROR(Prefill(input_tokens_vec, /*download_token=*/false))
        << "Error during prefill benchmark";
    auto p_end_time = absl::Now();
    total_prefill_duration_s +=
        absl::ToDoubleSeconds(p_end_time - p_start_time);
  }
  double avg_p_duration_s = total_prefill_duration_s / kPrefillRuns;
  ABSL_LOG(INFO) << "TTFT: " << avg_p_duration_s * 1000.0 << " ms";
  ABSL_LOG(INFO) << "\033[1;32mPrefill TPS: "
                 << (avg_p_duration_s > 0
                         ? input_tokens_vec.size() / avg_p_duration_s
                         : 0.0)
                 << "\033[0m";

  // 2. Decode Warmup
  constexpr int kDecodeWarmupTokens = 4;
  const int num_warmup_tokens = std::min(num_gen_tokens, kDecodeWarmupTokens);
  ABSL_LOG(INFO) << "Warming up decode (" << num_warmup_tokens << " tokens)...";
  DecodeOptions warmup_options;
  warmup_options.start_token_offset = input_tokens_vec.size();
  warmup_options.num_tokens = num_warmup_tokens;
  warmup_options.download_token = false;
  warmup_options.check_eos = false;
  ABSL_RETURN_IF_ERROR(Decode(warmup_options)) << "Error during decode warmup";

  // Re-run prefill to restore KV cache and reset decode input token to prompt
  // slice output.
  ABSL_RETURN_IF_ERROR(Prefill(input_tokens_vec, /*download_token=*/false))
      << "Error resetting prefill state";

  // 3. Decode Benchmark (pipelined autoregressive loop: Decode + Greedy)
  ABSL_LOG(INFO) << "Running decode benchmark (" << num_gen_tokens
                 << " tokens)...";
  std::vector<double> itl_ms;
  auto d_start_time = absl::Now();
  DecodeOptions bench_options;
  bench_options.start_token_offset = input_tokens_vec.size();
  bench_options.num_tokens = num_gen_tokens;
  bench_options.download_token = false;
  bench_options.check_eos = false;
  bench_options.itl_ms = &itl_ms;
  ABSL_RETURN_IF_ERROR(Decode(bench_options))
      << "Error during decode benchmark";
  auto d_end_time = absl::Now();

  double d_duration_s = absl::ToDoubleSeconds(d_end_time - d_start_time);
  double tps = d_duration_s > 0 ? num_gen_tokens / d_duration_s : 0.0;
  double avg_itl_ms =
      !itl_ms.empty() ? (d_duration_s * 1000.0) / num_gen_tokens : 0.0;

  std::vector<double> sorted_itl = itl_ms;
  std::sort(sorted_itl.begin(), sorted_itl.end());
  double p50 =
      !sorted_itl.empty() ? sorted_itl[sorted_itl.size() * 50 / 100] : 0.0;
  double p90 =
      !sorted_itl.empty() ? sorted_itl[sorted_itl.size() * 90 / 100] : 0.0;
  double p99 =
      !sorted_itl.empty() ? sorted_itl[sorted_itl.size() * 99 / 100] : 0.0;
  double min_itl = !sorted_itl.empty() ? sorted_itl.front() : 0.0;
  double max_itl = !sorted_itl.empty() ? sorted_itl.back() : 0.0;

  ABSL_LOG(INFO) << "\033[1;36mDecode TPS: " << tps << "\033[0m";
  ABSL_LOG(INFO) << "ITL (ms/tok) - Avg: " << avg_itl_ms << ", P50: " << p50
                 << ", P90: " << p90 << ", P99: " << p99 << ", Min: " << min_itl
                 << ", Max: " << max_itl;

  // Benchmark memory usage.
  {
    uint64_t weights_bytes = 0;
    for (const auto& tensor : model_tensors_) {
      weights_bytes += tensor->GetMemorySizeInBytes();
    }
    uint64_t kv_cache_bytes = 0;
    for (const auto& tensor : kv_cache_tensors_) {
      kv_cache_bytes += tensor->GetMemorySizeInBytes();
    }
    uint64_t activation_bytes = 0;
    if (prefill_context_) {
      activation_bytes +=
          prefill_context_->GetSizeOfMemoryAllocatedForConstTensors() +
          prefill_context_->GetSizeOfMemoryAllocatedForIntermediateTensors();
    }
    if (decode_context_) {
      activation_bytes +=
          decode_context_->GetSizeOfMemoryAllocatedForConstTensors() +
          decode_context_->GetSizeOfMemoryAllocatedForIntermediateTensors();
    }
    if (slice_context_) {
      activation_bytes +=
          slice_context_->GetSizeOfMemoryAllocatedForConstTensors() +
          slice_context_->GetSizeOfMemoryAllocatedForIntermediateTensors();
    }
    if (greedy_context_) {
      activation_bytes +=
          greedy_context_->GetSizeOfMemoryAllocatedForConstTensors() +
          greedy_context_->GetSizeOfMemoryAllocatedForIntermediateTensors();
    }

    double weights_mem = static_cast<double>(weights_bytes) / 1e9;
    double kv_cache_mem = static_cast<double>(kv_cache_bytes) / 1e9;
    double activation_mem = static_cast<double>(activation_bytes) / 1e9;
    double logical_mem = weights_mem + kv_cache_mem + activation_mem;

    ABSL_LOG(INFO) << "\033[1;35mMemory Usage (GB): "
                   << std::fixed << std::setprecision(3) << logical_mem
                   << " [Weights: " << weights_mem
                   << ", KVCache: " << kv_cache_mem
                   << ", Activation: " << activation_mem << "]\033[0m";
  }

  return absl::OkStatus();
}

std::unique_ptr<LlmModelRunner> CreateLlmModelRunner() {
  return std::make_unique<WebGpuLlmModelRunner>();
}

}  // namespace ml_drift
