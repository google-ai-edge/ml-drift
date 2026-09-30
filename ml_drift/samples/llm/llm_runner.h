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

#ifndef THIRD_PARTY_ML_DRIFT_SAMPLES_LLM_LLM_RUNNER_H_
#define THIRD_PARTY_ML_DRIFT_SAMPLES_LLM_LLM_RUNNER_H_

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/samples/llm/llm_config.h"
#include "third_party/sentencepiece/src/sentencepiece_processor.h"
#include "third_party/tokenizers_cpp/include/tokenizers_cpp.h"

namespace ml_drift {

// Round up x to the nearest multiple of m.
inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

// Check if a file exists and can be opened for reading.
inline bool FileExists(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return file.is_open();
}

inline absl::StatusOr<std::string> LoadFileToString(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return absl::NotFoundError(absl::StrCat("Could not open file: ", path));
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}


class Tokenizer {
 public:
  explicit Tokenizer(ModelType model_type) : model_type_(model_type) {}

  explicit Tokenizer(const std::string& model_path, ModelType model_type)
      : model_type_(model_type) {
    ABSL_CHECK_OK(Init(model_path));
  }

  static absl::StatusOr<std::unique_ptr<Tokenizer>> Create(
      const std::string& model_path, ModelType model_type) {
    auto tokenizer = std::make_unique<Tokenizer>(model_type);
    auto status = tokenizer->Init(model_path);
    if (!status.ok()) {
      return status;
    }
    return tokenizer;
  }

  absl::Status Init(const std::string& model_path) {
    ABSL_LOG(INFO) << "Loading tokenizer model: " << model_path;
    switch (model_type_) {
      case ModelType::kQwen3: {
        auto model_blob = LoadFileToString(model_path);
        if (!model_blob.ok()) {
          return model_blob.status();
        }
        if (model_blob->empty()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Tokenizer file is empty: ", model_path));
        }
        tokenizer_json_ = tokenizers::Tokenizer::FromBlobJSON(*model_blob);
        if (!tokenizer_json_) {
          return absl::InternalError(absl::StrCat(
              "Failed to parse JSON tokenizer from: ", model_path));
        }
        // bos_id_ not used
        eos_id_ = tokenizer_json_->TokenToId("<|im_end|>");
        end_of_turn_id_ = tokenizer_json_->TokenToId("<|im_end|>");
        break;
      }
      case ModelType::kGemma4: {
        auto model_blob = LoadFileToString(model_path);
        if (!model_blob.ok()) {
          return model_blob.status();
        }
        if (model_blob->empty()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Tokenizer file is empty: ", model_path));
        }
        tokenizer_json_ = tokenizers::Tokenizer::FromBlobJSON(*model_blob);
        if (!tokenizer_json_) {
          return absl::InternalError(absl::StrCat(
              "Failed to parse JSON tokenizer from: ", model_path));
        }
        bos_id_ = tokenizer_json_->TokenToId("<bos>");
        eos_id_ = tokenizer_json_->TokenToId("<eos>");
        end_of_turn_id_ = tokenizer_json_->TokenToId("<turn|>");
        break;
      }
      case ModelType::kGemma3: {
        tokenizer_sp_ =
            std::make_unique<sentencepiece::SentencePieceProcessor>();
        auto sp_status = tokenizer_sp_->Load(model_path);
        if (!sp_status.ok()) {
          return absl::InternalError(
              absl::StrCat("Failed to load sentencepiece tokenizer from ",
                           model_path, ": ", sp_status.message()));
        }
        bos_id_ = tokenizer_sp_->bos_id();
        eos_id_ = tokenizer_sp_->eos_id();
        end_of_turn_id_ = tokenizer_sp_->PieceToId("<end_of_turn>");
        break;
      }
    }
    return absl::OkStatus();
  }

  std::vector<int> EncodeInputTokens(const std::string& input_text,
                                     bool apply_chat_template = true) {
    std::string templated_input = input_text;
    if (apply_chat_template) {
      switch (model_type_) {
        case ModelType::kQwen3:
          templated_input = absl::StrCat("<|im_start|>user\n", templated_input);
          absl::StrAppend(&templated_input, "<|im_end|>\n");
          absl::StrAppend(&templated_input, "<|im_start|>assistant\n");
          break;
        case ModelType::kGemma4:
          templated_input = absl::StrCat("<|turn>user\n", templated_input);
          absl::StrAppend(&templated_input, "<turn|>\n");
          absl::StrAppend(&templated_input, "<|turn>model\n");
          break;
        case ModelType::kGemma3:
          templated_input =
              absl::StrCat("<start_of_turn>user\n", templated_input);
          absl::StrAppend(&templated_input, "<end_of_turn>\n");
          absl::StrAppend(&templated_input, "<start_of_turn>model\n");
          break;
      }
    }

    std::vector<int> input_tokens;
    if (model_type_ == ModelType::kQwen3 || model_type_ == ModelType::kGemma4) {
      std::vector<int32_t> input_tokens_i32 =
          tokenizer_json_->Encode(templated_input);
      input_tokens.assign(input_tokens_i32.begin(), input_tokens_i32.end());
    } else {
      ABSL_CHECK_OK(tokenizer_sp_->Encode(templated_input, &input_tokens));
    }

    // Add bos_id as separate token, not as part of the template string.
    if (bos_id_ != -1) {
      input_tokens.insert(input_tokens.begin(), bos_id_);
    }
    return input_tokens;
  }

  std::string DecodeOutputTokens(const std::vector<int>& output_tokens_vec) {
    std::string decoded_text;
    switch (model_type_) {
      case ModelType::kQwen3:
      case ModelType::kGemma4: {
        std::vector<int32_t> output_tokens_i32(output_tokens_vec.begin(),
                                               output_tokens_vec.end());
        return tokenizer_json_->Decode(output_tokens_i32);
      }
      case ModelType::kGemma3: {
        ABSL_CHECK_OK(tokenizer_sp_->Decode(output_tokens_vec, &decoded_text));
        return decoded_text;
      }
    }
    return decoded_text;
  }

  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }
  int end_of_turn_id() const { return end_of_turn_id_; }

 private:
  ModelType model_type_;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_sp_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_json_;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int end_of_turn_id_ = -1;
};

// Abstract base class for the backend-specific LLM runner.
class LlmModelRunner {
 public:
  virtual ~LlmModelRunner() = default;

  virtual absl::Status Init(const std::string& weights_path,
                            int prefill_sequence_size, int context_size,
                            bool use_fp32 = false) = 0;

  virtual absl::Status InitSliceAndGreedy() = 0;

  virtual absl::Status PostProcessGreedy(
      Tokenizer& tokenizer, const std::vector<int>& input_tokens_vec,
      int num_gen_tokens, std::vector<int>& output_tokens_vec) = 0;

  virtual absl::Status PostProcessGreedyBench(
      const std::vector<int>& input_tokens_vec, int num_gen_tokens) {
    return absl::UnimplementedError(
        "PostProcessGreedyBench is not implemented.");
  }
};

// Factory function implemented by each backend.
std::unique_ptr<LlmModelRunner> CreateLlmModelRunner();

}  // namespace ml_drift

#endif  // THIRD_PARTY_ML_DRIFT_SAMPLES_LLM_LLM_RUNNER_H_
