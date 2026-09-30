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

#include "ml_drift/samples/llm/llm_dummy_tensor_loader.h"

#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"

namespace ml_drift {
namespace {

TEST(LlmDummyTensorLoaderTest, HasTensor) {
  LlmDummyTensorLoader loader;
  EXPECT_TRUE(loader.HasTensor("any_arbitrary_tensor_name"));
}

TEST(LlmDummyTensorLoaderTest, LoadFloat32) {
  LlmDummyTensorLoader loader;
  std::vector<float> values = loader.LoadFloat32("test_tensor", 10);
  EXPECT_EQ(values.size(), 10);
  for (float val : values) {
    EXPECT_GT(val, 0.0f);
  }
}

TEST(LlmDummyTensorLoaderTest, LoadScale) {
  LlmDummyTensorLoader loader;
  bool tensor_created = false;
  loader.SetCreateTensorFn([&tensor_created](const TensorDescriptor& desc)
                               -> absl::StatusOr<GpuSpatialTensor*> {
    tensor_created = true;
    return nullptr;
  });

  OHWI shape(2, 1, 1, 1);
  auto result = loader.LoadScale("test_scale", shape);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(tensor_created);
}

TEST(LlmDummyTensorLoaderTest, LoadScaleDefaultFallback) {
  LlmDummyTensorLoader loader;
  OHWI shape(2, 1, 1, 1);
  auto result = loader.LoadScale("test_scale", shape);
  ASSERT_TRUE(result.ok());
  EXPECT_NE(result.value(), nullptr);
}

TEST(LlmDummyTensorLoaderTest, LoadInt8Weights) {
  LlmDummyTensorLoader loader;
  int tensor_count = 0;
  loader.SetCreateTensorFn([&tensor_count](const TensorDescriptor& desc)
                               -> absl::StatusOr<GpuSpatialTensor*> {
    tensor_count++;
    return nullptr;
  });

  WeightsDescription weights_desc;
  weights_desc.type = DataType::kUint8;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(4, 1, 1, 4);

  auto status = loader.LoadInt8Weights("test_int8", weights_desc, shape, false);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(tensor_count, 2);  // weights and weights_sum_i
}

TEST(LlmDummyTensorLoaderTest, LoadInt4Weights) {
  LlmDummyTensorLoader loader;
  int tensor_count = 0;
  loader.SetCreateTensorFn([&tensor_count](const TensorDescriptor& desc)
                               -> absl::StatusOr<GpuSpatialTensor*> {
    tensor_count++;
    return nullptr;
  });

  WeightsDescription weights_desc;
  weights_desc.type = DataType::kUint4;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = 1;
  OHWI shape(8, 1, 1, 8);

  auto status = loader.LoadInt4Weights("test_int4", weights_desc, shape, false);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(tensor_count, 2);  // weights and weights_sum_i
}

}  // namespace
}  // namespace ml_drift
