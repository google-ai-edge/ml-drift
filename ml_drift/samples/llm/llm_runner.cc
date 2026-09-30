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

#include "ml_drift/samples/llm/llm_runner.h"

#include "ml_drift/samples/llm/llm_config.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "third_party/gloop/base/init_google.h"
#include "third_party/gloop/base/log_file_flags.h"

ABSL_FLAG(std::string, prompt, "", "The input prompt for the model.");
ABSL_FLAG(std::string, prompt_file, "",
          "Path to a text file containing the input prompt.");
ABSL_FLAG(int, max_gen_tokens, 512, "Max number of tokens to generate.");
ABSL_FLAG(std::string, model, "none",
          "Which model to run. gemma3:1b, gemma3:270m, gemma4:12b, qwen3:0.6b, "
          "qwen3:1.7b, qwen3:8b");
ABSL_FLAG(std::string, weights_path, "",
          "Path to the extracted weights (+tokenizer).");
ABSL_FLAG(std::string, tokenizer_path, "",
          "Path to the tokenizer file (tokenizer.json or tokenizer.model). If "
          "not set, defaults to <weights_path>/tokenizer.json or "
          "tokenizer.model depending on the model.");
ABSL_FLAG(int, context_window_size, 0,
          "The context window size (sets cache_size in LlmConfig). If not "
          "set, the context size is automatically calculated via: "
          "# input tokens + max decode tokens.");
ABSL_FLAG(bool, benchmark, false,
          "If true, overrides prompt and runs model with speed benchmarks, and "
          "shows timing info.");
ABSL_FLAG(int, benchmark_prefill_tokens, 512,
          "Number of prefill tokens to benchmark.");
ABSL_FLAG(int, benchmark_decode_tokens, 128,
          "Number of decode tokens to benchmark.");

int main(int argc, char** argv) {
  InitGoogle(argv[0], &argc, &argv, true);
  // Set log level to see all logs.
  absl::SetFlag(&FLAGS_alsologtostderr, true);

  // Load prompt and benchmark flags.
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  std::string prompt_file = absl::GetFlag(FLAGS_prompt_file);
  const bool load_file = !prompt_file.empty();
  if (load_file) {
    auto prompt_content = ml_drift::LoadFileToString(prompt_file);
    if (!prompt_content.ok()) {
      ABSL_LOG(ERROR) << "Failed to read prompt file: "
                      << prompt_content.status().message();
      return 1;
    }
    prompt = *std::move(prompt_content);
  }
  const bool benchmark = absl::GetFlag(FLAGS_benchmark);
  if (prompt.empty() && !benchmark) {
    ABSL_LOG(ERROR) << "Usage: " << argv[0]
                    << " --model=<model_name> --weights_path=<path_to_weights>"
                    << " --prompt=<your_prompt>";
    return 1;
  }
  int out_tokens = absl::GetFlag(FLAGS_max_gen_tokens);
  if (benchmark) {
    out_tokens = absl::GetFlag(FLAGS_benchmark_decode_tokens);
  }

  // Model selection.
  auto model_info = ml_drift::GetModelInfo(absl::GetFlag(FLAGS_model));
  if (!model_info.ok()) {
    ABSL_LOG(ERROR) << model_info.status().message();
    return 1;
  }
  const ml_drift::ModelType model_type = model_info->type;
  const bool use_fp32 = model_info->force_fp32;
  const bool use_chat_template = true;  // Default to use chat template.

  // Load weights and tokenizer.
  std::string weights_path = absl::GetFlag(FLAGS_weights_path);
  if (weights_path.empty()) {
    ABSL_LOG(ERROR) << "No weights path provided.";
    return 1;
  }
  if (!absl::EndsWith(weights_path, "/")) {
    absl::StrAppend(&weights_path, "/");
  }
  const std::string expected_tokenizer_filename =
      ml_drift::GetTokenizerFilename(model_type);
  std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer_path);
  if (tokenizer_path.empty()) {
    tokenizer_path = weights_path + expected_tokenizer_filename;
  }
  if (!ml_drift::FileExists(tokenizer_path)) {
    ABSL_LOG(ERROR) << "Tokenizer file not found: " << tokenizer_path;
    ABSL_LOG(ERROR) << "Please ensure '" << expected_tokenizer_filename
                    << "' is present in the weights directory (" << weights_path
                    << ") or specify --tokenizer_path.";
    return 1;
  }

  // Initialize the tokenizer and encode the prompt.
  auto tokenizer_or = ml_drift::Tokenizer::Create(tokenizer_path, model_type);
  if (!tokenizer_or.ok()) {
    ABSL_LOG(ERROR) << "Failed to load tokenizer from " << tokenizer_path
                    << ": " << tokenizer_or.status().message();
    return 1;
  }
  auto tokenizer = std::move(*tokenizer_or);
  std::vector<int> input_tokens_vec;
  if (benchmark) {
    const int benchmark_prefill_tokens =
        absl::GetFlag(FLAGS_benchmark_prefill_tokens);
    input_tokens_vec.resize(benchmark_prefill_tokens, 1);
    if (benchmark_prefill_tokens > 0 && tokenizer->bos_id() != -1) {
      input_tokens_vec[0] = tokenizer->bos_id();
    }
  } else {
    input_tokens_vec = tokenizer->EncodeInputTokens(prompt, use_chat_template);
  }
  ABSL_LOG(INFO) << "Number of input tokens: " << input_tokens_vec.size();

  // Determine context window size (cache_size).
  const int required_tokens = input_tokens_vec.size() + out_tokens;
  int context_size = absl::GetFlag(FLAGS_context_window_size);
  if (context_size <= 0) {
    // If not set, set context size to required tokens, rounded up by 256.
    context_size = ml_drift::round_up(required_tokens, 256);
  } else if (context_size < required_tokens) {
    ABSL_LOG(ERROR) << "context_window_size (" << context_size
                    << ") is less than required tokens (" << required_tokens
                    << ").";
    return 1;
  }

  // Initialize the model runner and post-processing.
  std::unique_ptr<ml_drift::LlmModelRunner> runner =
      ml_drift::CreateLlmModelRunner();
  auto status = runner->Init(weights_path, input_tokens_vec.size(),
                             context_size, use_fp32);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Error: " << status.message();
    return 1;
  }
  status = runner->InitSliceAndGreedy();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Error: " << status.message();
    return 1;
  }

  // Run Prefill and Decode (w/ post-process greedy).
  const int num_gen_tokens = out_tokens;
  if (benchmark) {
    status = runner->PostProcessGreedyBench(input_tokens_vec, num_gen_tokens);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Error: " << status.message();
      return 1;
    }
    ABSL_LOG(INFO) << "Finished.";
    return 0;
  }
  std::vector<int> output_tokens_vec;
  output_tokens_vec.reserve(num_gen_tokens);
  status = runner->PostProcessGreedy(*tokenizer, input_tokens_vec,
                                     num_gen_tokens, output_tokens_vec);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Error: " << status.message();
    return 1;
  }

  return 0;
}
