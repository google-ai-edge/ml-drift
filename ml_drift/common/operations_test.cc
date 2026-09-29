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

#include "ml_drift/common/operations.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(DequantizeTensorTest, DequantizeINT8) {
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(1, 1, 1, 1);
  scale.data = {2.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(1, 1, 1, 1);
  zero_point.data = {1};
  Tensor<OHWI, DataType::kInt8> q_tensor;
  q_tensor.shape = OHWI(1, 2, 2, 1);
  q_tensor.data = {1, 2, 3, 4};
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  EXPECT_THAT(result.data, Pointwise(FloatNear(1e-5),
                                     std::vector<float>{0.0, 2.0, 4.0, 6.0}));
}

TEST(DequantizeTensorTest, DequantizeINT4) {
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(1, 1, 1, 1);
  scale.data = {2.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(1, 1, 1, 1);
  zero_point.data = {1};
  Tensor<OHWI, DataType::kInt4> q_tensor;
  q_tensor.shape = OHWI(1, 2, 2, 1);
  q_tensor.data.push_back(0x21);
  q_tensor.data.push_back(0x43);
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  EXPECT_THAT(result.data, Pointwise(FloatNear(1e-5),
                                     std::vector<float>{0.0, 2.0, 4.0, 6.0}));
}

TEST(DequantizeTensorTest, DequantizePerChannelINT8) {
  Tensor<OHWI, DataType::kInt8> q_tensor;
  q_tensor.shape = OHWI(2, 2, 1, 1);
  q_tensor.data = {1, 2, 3, 4};
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(2, 1, 1, 1);
  scale.data = {2.0f, 3.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(2, 1, 1, 1);
  zero_point.data = {1, 2};
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  EXPECT_THAT(result.data, Pointwise(FloatNear(1e-5),
                                     std::vector<float>{0.0, 2.0, 3.0, 6.0}));
}

TEST(DequantizeTensorTest, DequantizePerChannelINT4) {
  Tensor<OHWI, DataType::kInt4> q_tensor;
  q_tensor.shape = OHWI(2, 2, 1, 1);
  q_tensor.data.push_back(0x21);
  q_tensor.data.push_back(0x43);
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(2, 1, 1, 1);
  scale.data = {2.0f, 3.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(2, 1, 1, 1);
  zero_point.data = {1, 2};
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  EXPECT_THAT(result.data, Pointwise(FloatNear(1e-5),
                                     std::vector<float>{0.0, 2.0, 3.0, 6.0}));
}

TEST(DequantizeTensorTest, DequantizePerChannelScalePerTensorZeroPointINT8) {
  Tensor<OHWI, DataType::kInt8> q_tensor;
  q_tensor.shape = OHWI(2, 2, 1, 1);  // 2 channels, 2 elements per channel
  q_tensor.data = {1, 2, 3, 4};
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(2, 1, 1, 1);
  scale.data = {2.0f, 3.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(1, 1, 1, 1);  // Per-tensor zero point
  zero_point.data = {1};
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  // Channel 0: (1-1)*2.0=0.0, (2-1)*2.0=2.0
  // Channel 1: (3-1)*3.0=6.0, (4-1)*3.0=9.0
  EXPECT_THAT(result.data, Pointwise(FloatNear(1e-5),
                                     std::vector<float>{0.0, 2.0, 6.0, 9.0}));
}

TEST(DequantizeTensorTest, DequantizePerChannelScalePerTensorZeroPointINT4) {
  Tensor<OHWI, DataType::kInt4> q_tensor;
  q_tensor.shape = OHWI(2, 2, 1, 1);  // 2 channels, 2 elements per channel
  q_tensor.data.push_back(0x21);      // Unpacks to {1, 2}
  q_tensor.data.push_back(0x43);      // Unpacks to {3, 4}
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(2, 1, 1, 1);
  scale.data = {2.0f, 3.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(1, 1, 1, 1);  // Per-tensor zero point
  zero_point.data = {1};
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  // Channel 0: (1-1)*2.0=0.0, (2-1)*2.0=2.0
  // Channel 1: (3-1)*3.0=6.0, (4-1)*3.0=9.0
  EXPECT_THAT(result.data, Pointwise(FloatNear(1e-5),
                                     std::vector<float>{0.0, 2.0, 6.0, 9.0}));
}

TEST(DequantizeTensorTest, DequantizeINT4OddElements) {
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(1, 1, 1, 1);
  scale.data = {2.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(1, 1, 1, 1);
  zero_point.data = {1};
  Tensor<OHWI, DataType::kInt4> q_tensor;
  q_tensor.shape = OHWI(1, 1, 3, 1);  // 3 elements
  q_tensor.data.push_back(0x21);      // Unpacks to {1, 2}
  q_tensor.data.push_back(0x03);      // Unpacks to {3}
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  // (1-1)*2.0=0.0, (2-1)*2.0=2.0, (3-1)*2.0=4.0
  EXPECT_THAT(result.data,
              Pointwise(FloatNear(1e-5), std::vector<float>{0.0, 2.0, 4.0}));
}

TEST(DequantizeTensorTest, DequantizeINT2) {
  Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = OHWI(1, 1, 1, 1);
  scale.data = {2.0f};
  Tensor<OHWI, DataType::kInt32> zero_point;
  zero_point.shape = OHWI(1, 1, 1, 1);
  zero_point.data = {1};
  Tensor<OHWI, DataType::kInt2> q_tensor;
  q_tensor.shape = OHWI(1, 4, 1, 1);  // 4 elements
  q_tensor.data.push_back(0xE4);      // {0, 1, -2, -1}
  auto result = DequantizeTensor(q_tensor, scale, zero_point, false);
  EXPECT_THAT(
      result.data,
      Pointwise(FloatNear(1e-5), std::vector<float>{-2.0, 0.0, -6.0, -4.0}));
}

TEST(ToFloat32Test, ToFloat32INT4) {
  FullyConnectedInt4Attributes qattr;
  Tensor<OHWI, DataType::kInt4> int4_weights;
  int4_weights.shape = OHWI(1, 1, 2, 1);  // 2 elements
  int4_weights.data.push_back(0x21);      // {1, 2}
  qattr.weights = int4_weights;
  qattr.scale.shape = OHWI(1, 1, 1, 1);
  qattr.scale.data = {2.0f};
  qattr.zero_point.shape = OHWI(1, 1, 1, 1);
  qattr.zero_point.data = {1};
  qattr.op_name = "test_op";

  auto result = ToFloat32(qattr);
  EXPECT_EQ(result.op_name, "test_op");
  result.weights.data.resize(
      std::get<Tensor<OHWI, DataType::kInt4>>(qattr.weights)
          .shape.DimensionsProduct());
  EXPECT_THAT(result.weights.data,
              Pointwise(FloatNear(1e-5), std::vector<float>{0.0, 2.0}));
}
}  // namespace
}  // namespace ml_drift
