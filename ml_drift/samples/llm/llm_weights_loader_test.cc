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

#include "ml_drift/samples/llm/llm_weights_loader.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/samples/llm/llm_dummy_tensor_loader.h"

namespace ml_drift {
namespace {

TEST(LlmWeightsLoaderTest, LoadFloatWeights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilder builder(gpu_info, create_info.hints, create_info.precision,
                          create_info.storage_type);

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

  LlmWeightsLoader loader(builder, create_info, dummy_loader.get());
  OHWI shape(64, 1, 1, 32);

  auto [weights, type] = loader.GetWeights("layer.weight", shape);
  EXPECT_EQ(type, DataType::kFloat32);
  EXPECT_EQ(weights.shape.o, 64);
  EXPECT_EQ(weights.shape.i, 32);
  EXPECT_FALSE(weights.scale.has_value());
}

TEST(LlmWeightsLoaderTest, LoadInt8Weights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilder builder(gpu_info, create_info.hints, create_info.precision,
                          create_info.storage_type);

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

  LlmWeightsLoader loader(builder, create_info, dummy_loader.get());
  OHWI shape(64, 1, 1, 32);

  auto [weights, type] = loader.GetWeights("layer.weight", shape);
  EXPECT_EQ(type, DataType::kInt8);
  EXPECT_EQ(weights.shape.o, 64);
  EXPECT_EQ(weights.shape.i, 32);
  EXPECT_TRUE(weights.scale.has_value());
}

TEST(LlmWeightsLoaderTest, LoadInt4Weights) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuModelBuilder builder(gpu_info, create_info.hints, create_info.precision,
                          create_info.storage_type);

  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> allocated_tensors;
  auto dummy_loader = std::make_unique<LlmDummyTensorLoader>(
      /*quantization_i_group_size=*/-1, /*default_element_size_in_bits=*/4);
  dummy_loader->SetCreateTensorFn(
      [&allocated_tensors](
          const TensorDescriptor& desc) -> absl::StatusOr<GpuSpatialTensor*> {
        allocated_tensors.push_back(
            std::make_unique<DummyGpuSpatialTensor>(desc));
        return allocated_tensors.back().get();
      });

  LlmWeightsLoader loader(builder, create_info, dummy_loader.get());
  OHWI shape(64, 1, 1, 32);

  // Free function invocation test
  auto [weights, type] = GetWeights(loader, "layer.weight", shape);
  EXPECT_EQ(type, DataType::kInt4);
  EXPECT_EQ(weights.shape.o, 64);
  EXPECT_EQ(weights.shape.i, 32);
  EXPECT_TRUE(weights.scale.has_value());
}

}  // namespace
}  // namespace ml_drift
