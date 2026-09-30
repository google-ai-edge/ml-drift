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

#include "ml_drift/samples/llm/qwen3_model_builder.h"

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/samples/llm/llm_config.h"
#include "ml_drift/samples/llm/llm_dummy_tensor_loader.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"

namespace ml_drift {
namespace {

TEST(Qwen3ModelBuilderTest, BuildSmallModelWithFakeWeights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  LlmConfig config{
      .batch_size = 1,
      .sequence_size = 4,
      .model_dimension = 64,
      .hidden_dimension = 128,
      .head_dimension = 16,
      .number_of_heads = 4,
      .number_of_kv_heads = 1,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
  };

  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> allocated_tensors;
  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>();
  dummy_loader->SetCreateTensorFn(
      [&allocated_tensors](
          const TensorDescriptor& desc) -> absl::StatusOr<GpuSpatialTensor*> {
        allocated_tensors.push_back(
            std::make_unique<DummyGpuSpatialTensor>(desc));
        return allocated_tensors.back().get();
      });

  auto tensor_loader = LlmTensorLoader::MakeCaching(std::move(dummy_loader));

  Qwen3ModelBuilder builder(config, gpu_info, create_info,
                            std::move(tensor_loader));

  GpuModel prefill_model;
  GpuModel decode_model;
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle decode_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle output_handle;
  GpuModelBuilder::TensorHandle decode_output_handle;
  std::vector<GpuModelBuilder::TensorHandle> k_caches_in_handles;
  std::vector<GpuModelBuilder::TensorHandle> v_caches_in_handles;

  absl::Status status = builder.Build(
      &prefill_model, &decode_model, &input_handle, &decode_input_handle,
      &params_i32_handle, &output_handle, &decode_output_handle,
      &k_caches_in_handles, &v_caches_in_handles);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(prefill_model.nodes.empty());
  EXPECT_FALSE(decode_model.nodes.empty());
  EXPECT_EQ(k_caches_in_handles.size(), 2);
  EXPECT_EQ(v_caches_in_handles.size(), 2);
}

TEST(Qwen3ModelBuilderTest, BuildSmallModelWithInt8Weights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  LlmConfig config{
      .batch_size = 1,
      .sequence_size = 4,
      .model_dimension = 64,
      .hidden_dimension = 128,
      .head_dimension = 16,
      .number_of_heads = 4,
      .number_of_kv_heads = 1,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
  };

  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> allocated_tensors;
  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>(
      /*quantization_i_group_size=*/-1, /*default_element_size_in_bits=*/8);
  dummy_loader->SetCreateTensorFn(
      [&allocated_tensors](
          const TensorDescriptor& desc) -> absl::StatusOr<GpuSpatialTensor*> {
        allocated_tensors.push_back(
            std::make_unique<DummyGpuSpatialTensor>(desc));
        return allocated_tensors.back().get();
      });

  auto tensor_loader = LlmTensorLoader::MakeCaching(std::move(dummy_loader));

  Qwen3ModelBuilder builder(config, gpu_info, create_info,
                            std::move(tensor_loader));

  GpuModel prefill_model;
  GpuModel decode_model;
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle decode_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle output_handle;
  GpuModelBuilder::TensorHandle decode_output_handle;
  std::vector<GpuModelBuilder::TensorHandle> k_caches_in_handles;
  std::vector<GpuModelBuilder::TensorHandle> v_caches_in_handles;

  absl::Status status = builder.Build(
      &prefill_model, &decode_model, &input_handle, &decode_input_handle,
      &params_i32_handle, &output_handle, &decode_output_handle,
      &k_caches_in_handles, &v_caches_in_handles);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(prefill_model.nodes.empty());
  EXPECT_FALSE(decode_model.nodes.empty());
  EXPECT_EQ(k_caches_in_handles.size(), 2);
  EXPECT_EQ(v_caches_in_handles.size(), 2);
}

TEST(Qwen3ModelBuilderTest, BuildSmallModelWithFloatWeights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  LlmConfig config{
      .batch_size = 1,
      .sequence_size = 4,
      .model_dimension = 64,
      .hidden_dimension = 128,
      .head_dimension = 16,
      .number_of_heads = 4,
      .number_of_kv_heads = 1,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
  };

  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> allocated_tensors;
  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>(
      /*quantization_i_group_size=*/-1, /*default_element_size_in_bits=*/32);
  dummy_loader->SetCreateTensorFn(
      [&allocated_tensors](
          const TensorDescriptor& desc) -> absl::StatusOr<GpuSpatialTensor*> {
        allocated_tensors.push_back(
            std::make_unique<DummyGpuSpatialTensor>(desc));
        return allocated_tensors.back().get();
      });

  auto tensor_loader = LlmTensorLoader::MakeCaching(std::move(dummy_loader));

  Qwen3ModelBuilder builder(config, gpu_info, create_info,
                            std::move(tensor_loader));

  GpuModel prefill_model;
  GpuModel decode_model;
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle decode_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle output_handle;
  GpuModelBuilder::TensorHandle decode_output_handle;
  std::vector<GpuModelBuilder::TensorHandle> k_caches_in_handles;
  std::vector<GpuModelBuilder::TensorHandle> v_caches_in_handles;

  absl::Status status = builder.Build(
      &prefill_model, &decode_model, &input_handle, &decode_input_handle,
      &params_i32_handle, &output_handle, &decode_output_handle,
      &k_caches_in_handles, &v_caches_in_handles);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(prefill_model.nodes.empty());
  EXPECT_FALSE(decode_model.nodes.empty());
  EXPECT_EQ(k_caches_in_handles.size(), 2);
  EXPECT_EQ(v_caches_in_handles.size(), 2);
}

TEST(Qwen3ModelBuilderTest, BuildSmallModelWithMixedInt8EmbedInt4Weights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  LlmConfig config{
      .batch_size = 1,
      .sequence_size = 4,
      .model_dimension = 64,
      .hidden_dimension = 128,
      .head_dimension = 16,
      .number_of_heads = 4,
      .number_of_kv_heads = 1,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
  };

  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> allocated_tensors;
  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>(
      /*quantization_i_group_size=*/-1, /*default_element_size_in_bits=*/4);
  dummy_loader->SetTensorElementSizeInBits("model.embed_tokens.w", 8);
  dummy_loader->SetCreateTensorFn(
      [&allocated_tensors](
          const TensorDescriptor& desc) -> absl::StatusOr<GpuSpatialTensor*> {
        allocated_tensors.push_back(
            std::make_unique<DummyGpuSpatialTensor>(desc));
        return allocated_tensors.back().get();
      });

  auto tensor_loader = LlmTensorLoader::MakeCaching(std::move(dummy_loader));

  Qwen3ModelBuilder builder(config, gpu_info, create_info,
                            std::move(tensor_loader));

  GpuModel prefill_model;
  GpuModel decode_model;
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle decode_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle output_handle;
  GpuModelBuilder::TensorHandle decode_output_handle;
  std::vector<GpuModelBuilder::TensorHandle> k_caches_in_handles;
  std::vector<GpuModelBuilder::TensorHandle> v_caches_in_handles;

  absl::Status status = builder.Build(
      &prefill_model, &decode_model, &input_handle, &decode_input_handle,
      &params_i32_handle, &output_handle, &decode_output_handle,
      &k_caches_in_handles, &v_caches_in_handles);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(prefill_model.nodes.empty());
  EXPECT_FALSE(decode_model.nodes.empty());
  EXPECT_EQ(k_caches_in_handles.size(), 2);
  EXPECT_EQ(v_caches_in_handles.size(), 2);
}

TEST(Qwen3ModelBuilderTest, BuildPostProcessGreedy) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  LlmConfig config{
      .batch_size = 1,
      .vocabulary_size = 100,
  };

  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>();
  Qwen3ModelBuilder builder(config, gpu_info, create_info,
                            std::move(dummy_loader));

  GpuModel greedy_model;
  GpuModelBuilder::TensorHandle input_logits_handle;
  GpuModelBuilder::TensorHandle output_token_handle;

  absl::Status status = builder.BuildPostProcessGreedy(
      &greedy_model, &input_logits_handle, &output_token_handle);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(greedy_model.nodes.empty());
}

TEST(Qwen3ModelBuilderTest, Qwen3ModelConfig0_6B) {
  LlmConfig config = GetQwen3ModelConfig0_6B();
  EXPECT_EQ(config.model_dimension, 1024);
  EXPECT_EQ(config.hidden_dimension, 3072);
  EXPECT_EQ(config.head_dimension, 128);
  EXPECT_EQ(config.number_of_heads, 16);
  EXPECT_EQ(config.number_of_kv_heads, 8);
  EXPECT_EQ(config.vocabulary_size, 151936);
  EXPECT_EQ(config.stack_size, 28);
}

TEST(Qwen3ModelBuilderTest, Qwen3ModelConfig1_7B) {
  LlmConfig config = GetQwen3ModelConfig1_7B();
  EXPECT_EQ(config.model_dimension, 2048);
  EXPECT_EQ(config.hidden_dimension, 6144);
  EXPECT_EQ(config.head_dimension, 128);
  EXPECT_EQ(config.number_of_heads, 16);
  EXPECT_EQ(config.number_of_kv_heads, 8);
  EXPECT_EQ(config.vocabulary_size, 151936);
  EXPECT_EQ(config.stack_size, 28);
}

TEST(Qwen3ModelBuilderTest, Qwen3ModelConfig8B) {
  LlmConfig config = GetQwen3ModelConfig8B();
  EXPECT_EQ(config.model_dimension, 4096);
  EXPECT_EQ(config.hidden_dimension, 12288);
  EXPECT_EQ(config.head_dimension, 128);
  EXPECT_EQ(config.number_of_heads, 32);
  EXPECT_EQ(config.number_of_kv_heads, 8);
  EXPECT_EQ(config.vocabulary_size, 151936);
  EXPECT_EQ(config.stack_size, 36);
}

TEST(Qwen3ModelBuilderTest, GetModelInfo) {
  auto qwen_info = GetModelInfo("qwen3:8b");
  ASSERT_TRUE(qwen_info.ok());
  if (qwen_info.ok()) {
    EXPECT_EQ(qwen_info->type, ModelType::kQwen3);
    EXPECT_EQ(qwen_info->config.model_dimension, 4096);
    EXPECT_FALSE(qwen_info->force_fp32);
    EXPECT_STREQ(GetTokenizerFilename(qwen_info->type), "tokenizer.json");
  }

  auto gemma3_info = GetModelInfo("gemma3:270m");
  ASSERT_TRUE(gemma3_info.ok());
  if (gemma3_info.ok()) {
    EXPECT_EQ(gemma3_info->type, ModelType::kGemma3);
    EXPECT_TRUE(gemma3_info->force_fp32);
    EXPECT_STREQ(GetTokenizerFilename(gemma3_info->type), "tokenizer.model");
  }

  auto gemma4_info = GetModelInfo("gemma4:12b");
  ASSERT_TRUE(gemma4_info.ok());
  if (gemma4_info.ok()) {
    EXPECT_EQ(gemma4_info->type, ModelType::kGemma4);
    EXPECT_FALSE(gemma4_info->force_fp32);
    EXPECT_STREQ(GetTokenizerFilename(gemma4_info->type), "tokenizer.json");
  }

  EXPECT_FALSE(GetModelInfo("unknown:model").ok());
}

TEST(Qwen3ModelBuilderTest, BuildModel8BStructure) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  LlmConfig config = GetQwen3ModelConfig8B();
  config.stack_size = 1;
  config.sequence_size = 2;
  config.cache_size = 8;

  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> allocated_tensors;
  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>();
  dummy_loader->SetCreateTensorFn(
      [&allocated_tensors](
          const TensorDescriptor& desc) -> absl::StatusOr<GpuSpatialTensor*> {
        allocated_tensors.push_back(
            std::make_unique<DummyGpuSpatialTensor>(desc));
        return allocated_tensors.back().get();
      });

  auto tensor_loader = LlmTensorLoader::MakeCaching(std::move(dummy_loader));

  Qwen3ModelBuilder builder(config, gpu_info, create_info,
                            std::move(tensor_loader));

  GpuModel prefill_model;
  GpuModel decode_model;
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle decode_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle output_handle;
  GpuModelBuilder::TensorHandle decode_output_handle;
  std::vector<GpuModelBuilder::TensorHandle> k_caches_in_handles;
  std::vector<GpuModelBuilder::TensorHandle> v_caches_in_handles;

  absl::Status status = builder.Build(
      &prefill_model, &decode_model, &input_handle, &decode_input_handle,
      &params_i32_handle, &output_handle, &decode_output_handle,
      &k_caches_in_handles, &v_caches_in_handles);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(prefill_model.nodes.empty());
  EXPECT_FALSE(decode_model.nodes.empty());
  EXPECT_EQ(k_caches_in_handles.size(), 1);
  EXPECT_EQ(v_caches_in_handles.size(), 1);
}

}  // namespace
}  // namespace ml_drift
