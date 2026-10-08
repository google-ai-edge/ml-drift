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

#ifndef ML_DRIFT_SAMPLES_LLM_LLM_CONFIG_H_
#define ML_DRIFT_SAMPLES_LLM_LLM_CONFIG_H_

#include <algorithm>
#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ml_drift {

struct LlmRuntimeParams {
  int token_index_offset = 0;
  int active_tokens = 0;

  // indexes of values in linear int32 param buffer
  static constexpr int kTokenOffsetIndex = 0;
  static constexpr int kActiveTokensIndex = 1;
  static constexpr int kActiveTokensAlignedIndex = 2;
  static constexpr int kRingOffsetIndex = 3;

  static constexpr int kTotalParamsCount = 4;
};

enum class ModelType {
  kGemma3,
  kGemma4,
  kQwen3,
};

struct LlmConfig {
  int batch_size;
  int sequence_size;  // defined by input prompt length
  int model_dimension;
  int hidden_dimension;
  int head_dimension;
  int number_of_heads;
  int number_of_kv_heads;
  int vocabulary_size;
  int stack_size;
  int cache_size;  // determines maximum context length, configured in runner
  int num_local_layers_per_global;
  int sliding_window_size;
  std::optional<int> global_head_dimension;
  std::optional<int> global_number_of_kv_heads;
  std::optional<int> local_ring_cache_size;
  std::optional<float> global_rope_proportion;
};

inline int GetLocalRingCacheSize(const LlmConfig& config) {
  const int base_size = config.local_ring_cache_size.value_or(
      config.sliding_window_size + config.sequence_size);
  const int aligned_size = (base_size + 3) / 4 * 4;
  return std::min(config.cache_size, aligned_size);
}

inline bool UseRingBuffer(const LlmConfig& config) {
  return config.sliding_window_size > 0 &&
         config.num_local_layers_per_global > 0 &&
         config.cache_size > config.sliding_window_size;
}

inline int GetRingOffset(const LlmConfig& config) {
  const int ring_cache_size =
      UseRingBuffer(config) ? GetLocalRingCacheSize(config) : config.cache_size;
  return 1 - config.sliding_window_size + ring_cache_size;
}

inline int GetRingOffset(const LlmConfig& config, int token_offset) {
  if (config.sliding_window_size <= 0) {
    return 0;
  }
  return std::max(0, token_offset + 1 - config.sliding_window_size);
}

inline int GetCacheSize(const LlmConfig& config, bool is_global) {
  if (UseRingBuffer(config) && !is_global) {
    return GetLocalRingCacheSize(config);
  }
  return config.cache_size;
}

inline LlmConfig GetGemma4ModelConfig12B() {
  return LlmConfig{
      .batch_size = 1,
      .sequence_size = 1,
      .model_dimension = 3840,
      .hidden_dimension = 15360,
      .head_dimension = 256,
      .number_of_heads = 16,
      .number_of_kv_heads = 8,  // GQA
      .vocabulary_size = 262144,
      .stack_size = 48,
      .num_local_layers_per_global = 5,
      .sliding_window_size = 1024,
      .global_head_dimension = 512,
      .global_number_of_kv_heads = 1,
      .global_rope_proportion = 0.25f,
  };
}

inline LlmConfig GetGemma3ModelConfig270M() {
  return LlmConfig{
      .batch_size = 1,
      .sequence_size = 1,
      .model_dimension = 640,
      .hidden_dimension = 2048,
      .head_dimension = 256,
      .number_of_heads = 4,
      .number_of_kv_heads = 1,  // MQA
      .vocabulary_size = 262144,
      .stack_size = 18,
      .num_local_layers_per_global = 5,
      .sliding_window_size = 512,
  };
}

inline LlmConfig GetGemma3ModelConfig1B() {
  return LlmConfig{
      .batch_size = 1,
      .sequence_size = 1,
      .model_dimension = 1152,
      .hidden_dimension = 6 * 1152,
      .head_dimension = 256,
      .number_of_heads = 4,
      .number_of_kv_heads = 1,  // MQA
      .vocabulary_size = 262144,
      .stack_size = 26,
      .num_local_layers_per_global = 5,
      .sliding_window_size = 512,
  };
}

inline LlmConfig GetQwen3ModelConfig0_6B() {
  return LlmConfig{
      .batch_size = 1,
      .sequence_size = 1,
      .model_dimension = 1024,
      .hidden_dimension = 3072,
      .head_dimension = 128,
      .number_of_heads = 16,
      .number_of_kv_heads = 8,  // GQA
      .vocabulary_size = 151936,
      .stack_size = 28,
      .num_local_layers_per_global = 0,
      .sliding_window_size = 0,
  };
}

inline LlmConfig GetQwen3ModelConfig1_7B() {
  return LlmConfig{
      .batch_size = 1,
      .sequence_size = 1,
      .model_dimension = 2048,
      .hidden_dimension = 6144,
      .head_dimension = 128,
      .number_of_heads = 16,
      .number_of_kv_heads = 8,  // GQA
      .vocabulary_size = 151936,
      .stack_size = 28,
      .num_local_layers_per_global = 0,
      .sliding_window_size = 0,
  };
}

inline LlmConfig GetQwen3ModelConfig8B() {
  return LlmConfig{
      .batch_size = 1,
      .sequence_size = 1,
      .model_dimension = 4096,
      .hidden_dimension = 12288,
      .head_dimension = 128,
      .number_of_heads = 32,
      .number_of_kv_heads = 8,  // GQA
      .vocabulary_size = 151936,
      .stack_size = 36,
      .num_local_layers_per_global = 0,
      .sliding_window_size = 0,
  };
}

struct ModelInfo {
  ModelType type;
  LlmConfig config;
  bool force_fp32 = false;
};

inline const char* GetTokenizerFilename(ModelType model_type) {
  return (model_type == ModelType::kQwen3 || model_type == ModelType::kGemma4)
             ? "tokenizer.json"
             : "tokenizer.model";
}

inline absl::StatusOr<ModelInfo> GetModelInfo(absl::string_view model_name) {
  if (model_name == "gemma3:270m") {
    return ModelInfo{
        .type = ModelType::kGemma3,
        .config = GetGemma3ModelConfig270M(),
        .force_fp32 = true,
    };
  } else if (model_name == "gemma3:1b") {
    return ModelInfo{
        .type = ModelType::kGemma3,
        .config = GetGemma3ModelConfig1B(),
    };
  } else if (model_name == "gemma4:12b") {
    return ModelInfo{
        .type = ModelType::kGemma4,
        .config = GetGemma4ModelConfig12B(),
    };
  } else if (model_name == "qwen3:0.6b") {
    return ModelInfo{
        .type = ModelType::kQwen3,
        .config = GetQwen3ModelConfig0_6B(),
    };
  } else if (model_name == "qwen3:1.7b") {
    return ModelInfo{
        .type = ModelType::kQwen3,
        .config = GetQwen3ModelConfig1_7B(),
    };
  } else if (model_name == "qwen3:8b") {
    return ModelInfo{
        .type = ModelType::kQwen3,
        .config = GetQwen3ModelConfig8B(),
    };
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown model: ", model_name));
}

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_LLM_LLM_CONFIG_H_
