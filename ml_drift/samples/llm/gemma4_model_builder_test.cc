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

#include "ml_drift/samples/llm/gemma4_model_builder.h"

#include <memory>
#include <optional>
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

TEST(Gemma4ModelBuilderTest, BuildSmallModelWithFakeWeights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  // 2 layers: layer 0 (local), layer 1 (global)
  LlmConfig config{
      .batch_size = 1,
      .sequence_size = 4,
      .model_dimension = 64,
      .hidden_dimension = 128,
      .head_dimension = 16,
      .number_of_heads = 4,
      .number_of_kv_heads = 2,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
      .num_local_layers_per_global = 1,
      .sliding_window_size = 16,
      .global_head_dimension = 32,
      .global_number_of_kv_heads = 1,
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

  Gemma4ModelBuilder builder(config, gpu_info, create_info,
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

TEST(Gemma4ModelBuilderTest, BuildSmallModelWithInt8Weights) {
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
      .number_of_kv_heads = 2,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
      .num_local_layers_per_global = 1,
      .sliding_window_size = 16,
      .global_head_dimension = 32,
      .global_number_of_kv_heads = 1,
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

  Gemma4ModelBuilder builder(config, gpu_info, create_info,
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

TEST(Gemma4ModelBuilderTest, BuildSmallModelWithFloatWeights) {
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
      .number_of_kv_heads = 2,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
      .num_local_layers_per_global = 1,
      .sliding_window_size = 16,
      .global_head_dimension = 32,
      .global_number_of_kv_heads = 1,
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

  Gemma4ModelBuilder builder(config, gpu_info, create_info,
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

TEST(Gemma4ModelBuilderTest, BuildSmallModelWithMixedInt8EmbedInt4Weights) {
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
      .number_of_kv_heads = 2,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
      .num_local_layers_per_global = 1,
      .sliding_window_size = 16,
      .global_head_dimension = 32,
      .global_number_of_kv_heads = 1,
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

  Gemma4ModelBuilder builder(config, gpu_info, create_info,
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

TEST(Gemma4ModelBuilderTest, BuildSmallModelWithMixedFloatEmbedInt4Weights) {
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
      .number_of_kv_heads = 2,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 16,
      .num_local_layers_per_global = 1,
      .sliding_window_size = 16,
      .global_head_dimension = 32,
      .global_number_of_kv_heads = 1,
  };

  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> allocated_tensors;
  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>(
      /*quantization_i_group_size=*/-1, /*default_element_size_in_bits=*/4);
  dummy_loader->SetTensorElementSizeInBits("model.embed_tokens.w", 32);
  dummy_loader->SetCreateTensorFn(
      [&allocated_tensors](
          const TensorDescriptor& desc) -> absl::StatusOr<GpuSpatialTensor*> {
        allocated_tensors.push_back(
            std::make_unique<DummyGpuSpatialTensor>(desc));
        return allocated_tensors.back().get();
      });

  auto tensor_loader = LlmTensorLoader::MakeCaching(std::move(dummy_loader));

  Gemma4ModelBuilder builder(config, gpu_info, create_info,
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

TEST(Gemma4ModelBuilderTest, BuildPostProcessGreedy) {
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
  Gemma4ModelBuilder builder(config, gpu_info, create_info,
                             std::move(dummy_loader));

  GpuModel greedy_model;
  GpuModelBuilder::TensorHandle input_logits_handle;
  GpuModelBuilder::TensorHandle output_token_handle;

  absl::Status status = builder.BuildPostProcessGreedy(
      &greedy_model, &input_logits_handle, &output_token_handle);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(greedy_model.nodes.empty());
}

TEST(Gemma4ModelBuilderTest, RingBufferConfigHelpers) {
  LlmConfig config{
      .batch_size = 1,
      .sequence_size = 4,
      .stack_size = 4,
      .cache_size = 128,
      .num_local_layers_per_global = 3,
      .sliding_window_size = 32,
  };
  EXPECT_TRUE(UseRingBuffer(config));
  // Default local ring cache size = sliding_window_size + sequence_size = 32 +
  // 4 = 36.
  EXPECT_EQ(GetLocalRingCacheSize(config), 36);
  // For local layers:
  EXPECT_EQ(GetCacheSize(config, /*is_global=*/false), 36);
  // For global layers:
  EXPECT_EQ(GetCacheSize(config, /*is_global=*/true), 128);
  // Ring offset: 1 - 32 + 36 = 5.
  EXPECT_EQ(GetRingOffset(config), 5);

  // Explicit local ring cache size:
  config.local_ring_cache_size = 64;
  EXPECT_EQ(GetLocalRingCacheSize(config), 64);
  EXPECT_EQ(GetCacheSize(config, /*is_global=*/false), 64);
  EXPECT_EQ(GetRingOffset(config), 1 - 32 + 64);

  // When sliding_window_size is 0:
  config.sliding_window_size = 0;
  EXPECT_FALSE(UseRingBuffer(config));
  EXPECT_EQ(GetCacheSize(config, /*is_global=*/false), 128);

  // When cache_size <= sliding_window_size:
  config.sliding_window_size = 32;
  config.local_ring_cache_size = std::nullopt;
  config.cache_size = 32;
  EXPECT_FALSE(UseRingBuffer(config));
  EXPECT_EQ(GetCacheSize(config, /*is_global=*/false), 32);

  // When sliding_window_size < cache_size <= sliding_window_size +
  // sequence_size:
  config.cache_size = 34;
  EXPECT_TRUE(UseRingBuffer(config));
  EXPECT_EQ(GetLocalRingCacheSize(config), 34);  // min(34, 36) = 34
  EXPECT_EQ(GetCacheSize(config, /*is_global=*/false), 34);
  EXPECT_EQ(GetRingOffset(config), 1 - 32 + 34);  // 3
}

TEST(Gemma4ModelBuilderTest, BuildModelWithRingBufferKVCache) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  // 2 layers: layer 0 (local), layer 1 (global)
  // sliding_window_size = 16, cache_size = 64
  // Ring buffer is active for local layers because cache_size >
  // sliding_window_size.
  LlmConfig config{
      .batch_size = 1,
      .sequence_size = 4,
      .model_dimension = 64,
      .hidden_dimension = 128,
      .head_dimension = 16,
      .number_of_heads = 4,
      .number_of_kv_heads = 2,
      .vocabulary_size = 100,
      .stack_size = 2,
      .cache_size = 64,
      .num_local_layers_per_global = 1,
      .sliding_window_size = 16,
      .global_head_dimension = 32,
      .global_number_of_kv_heads = 1,
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
  Gemma4ModelBuilder builder(config, gpu_info, create_info,
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

  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(prefill_model.nodes.empty());
  EXPECT_FALSE(decode_model.nodes.empty());
  ASSERT_EQ(k_caches_in_handles.size(), 2);
  ASSERT_EQ(v_caches_in_handles.size(), 2);

  // Layer 0 is local, so KV cache tensor should be sized according to
  // GetCacheSize(config, false) = 16 + 4 = 20.
  const int local_ring_cache_size = GetCacheSize(config, /*is_global=*/false);
  EXPECT_EQ(local_ring_cache_size, 20);
  const int global_cache_size = GetCacheSize(config, /*is_global=*/true);
  EXPECT_EQ(global_cache_size, 64);

  const int local_k_elements =
      k_caches_in_handles[0].tensor_desc.GetBHWCShape().c;
  const int global_k_elements =
      k_caches_in_handles[1].tensor_desc.GetBHWCShape().c;
  EXPECT_LT(local_k_elements, global_k_elements);
}

TEST(Gemma4ModelBuilderTest, Gemma4ConfigHasProportionalRoPE) {
  LlmConfig config = GetGemma4ModelConfig12B();
  EXPECT_TRUE(config.global_rope_proportion.has_value());
  EXPECT_FLOAT_EQ(*config.global_rope_proportion, 0.25f);
  EXPECT_EQ(config.global_head_dimension.value_or(0), 512);
}

}  // namespace
}  // namespace ml_drift
