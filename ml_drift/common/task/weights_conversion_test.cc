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

#include "ml_drift/common/task/weights_conversion.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(ConvertF32F16, BaseTest) {
  const std::vector<float> src = {0.1, 0.2, 0.3, 0.4};
  std::unique_ptr<half[]> dst = ConvertF32F16(src);
  EXPECT_THAT(absl::MakeConstSpan(dst.get(), 4),
              ElementsAre(half(0.1), half(0.2), half(0.3), half(0.4)));
}

TEST(RearrangeWeightsUInt4Packed, kOSpatialIOGroupI4O4Base) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 15 -- packed
  for (uint8_t i = 0; i < 8; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 2 3      : 6
  // 4 5 6 7      : 4 5 6 7      : 22
  // 8 9 10 11    : -8 -7 -6 -5  : -26
  // 12 13 14 15  : -4 -3 -2 -1  : -10

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/8u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x8
  std::vector<uint8_t> expected_output = {0, 4, 8,  12, 1, 5, 9,  13,
                                          2, 6, 10, 14, 3, 7, 11, 15};
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        expected_output[2 * i] | expected_output[2 * i + 1] << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {6, 22, -26, -10};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt4Packed, kOSpatialIOGroupI4O4ZeroWeights) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 15 -- packed
  for (uint8_t i = 0; i < 8; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 2 3      : 6
  // 4 5 6 7      : 4 5 6 7      : 22
  // 8 9 10 11    : -8 -7 -6 -5  : -26
  // 12 13 14 15  : -4 -3 -2 -1  : -10

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/8u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x8
  std::vector<uint8_t> expected_output = {0, 4, 8,  12, 1, 5, 9,  13,
                                          2, 6, 10, 14, 3, 7, 11, 15};
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        expected_output[2 * i] | expected_output[2 * i + 1] << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {6, 22, -26, -10};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt4Packed, k2DYIsSpatialIOAndXIsOGroupI4O4ZeroWeights) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 15 -- packed
  for (uint8_t i = 0; i < 8; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 2 3      : 6
  // 4 5 6 7      : 4 5 6 7      : 22
  // 8 9 10 11    : -8 -7 -6 -5  : -26
  // 12 13 14 15  : -4 -3 -2 -1  : -10

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/8u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x8
  std::vector<uint8_t> expected_output = {0, 4, 8,  12, 1, 5, 9,  13,
                                          2, 6, 10, 14, 3, 7, 11, 15};
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        expected_output[2 * i] | expected_output[2 * i + 1] << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {6, 22, -26, -10};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

uint8_t Int4AsUint4(int8_t value) { return value >= 0 ? value : (value + 16); }

TEST(RearrangeWeightsUInt4Packed, kOSpatialIOGroupI4O4BaseVersusRef) {
  Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(4, 1, 1, 4);
  weights_i8.data = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  std::vector<uint8_t> weights_i8_packed(weights_i8.data.size() / 2);
  for (int i = 0; i < weights_i8.data.size() / 2; ++i) {
    uint8_t part0 = Int4AsUint4(weights_i8.data[i * 2]);
    uint8_t part1 = Int4AsUint4(weights_i8.data[i * 2 + 1]);
    weights_i8_packed[i] = (part1 << 4) | part0;
  }

  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/8u, /*swap_dims=*/false));

  std::vector<uint8_t> data_ref(weights_i8.shape.DimensionsProduct() / 2);
  RearrangeWeightsInt8AsUint4(weights_i8, dst_weight_desc,
                              absl::MakeSpan(data_ref), 8, 8u);
  EXPECT_THAT(output, ElementsAreArray(data_ref));
  auto expected_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i.data));
}

TEST(RearrangeWeightsUInt4Packed,
     k2DYIsSpatialIOAndXIsOGroupI4O4BaseVersusRef) {
  Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(16, 2, 1, 8);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = (i % 16) - 8;
  }

  std::vector<uint8_t> weights_i8_packed(weights_i8.data.size() / 2);
  for (int i = 0; i < weights_i8.data.size() / 2; ++i) {
    uint8_t part0 = Int4AsUint4(weights_i8.data[i * 2]);
    uint8_t part1 = Int4AsUint4(weights_i8.data[i * 2 + 1]);
    weights_i8_packed[i] = (part1 << 4) | part0;
  }

  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(16, 2, 1, 8);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 2;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape));
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/8u, /*swap_dims=*/false));

  std::vector<uint8_t> data_ref(output.size());
  RearrangeWeightsInt8AsUint4(weights_i8, dst_weight_desc,
                              absl::MakeSpan(data_ref), 8, 8u);
  EXPECT_THAT(output, ElementsAreArray(data_ref));
  auto expected_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i.data));
}

TEST(RearrangeWeightsUInt4Packed, k2DYIsSpatialIOAndXIsOGroupI4O4Base) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 15 -- packed
  for (uint8_t i = 0; i < 8; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 2 3      : 6
  // 4 5 6 7      : 4 5 6 7      : 22
  // 8 9 10 11    : -8 -7 -6 -5  : -26
  // 12 13 14 15  : -4 -3 -2 -1  : -10

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/8u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x8
  std::vector<uint8_t> expected_output = {0, 4, 8,  12, 1, 5, 9,  13,
                                          2, 6, 10, 14, 3, 7, 11, 15};
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        expected_output[2 * i] | expected_output[2 * i + 1] << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {6, 22, -26, -10};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt4Packed, kOSpatialIOGroupI4O4Pad) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(3, 1, 1, 4);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 12 -- packed
  for (uint8_t i = 0; i < 6; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 2 3      : 6
  // 4 5 6 7      : 4 5 6 7      : 22
  // 8 9 10 11    : -8 -7 -6 -5  : -26

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  uint8_t pad = 8u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i), pad,
                                        /*swap_dims=*/false));

  // Should result in filling to be 4x4 with pad_value, then bit flip 0x8
  pad ^= 0x8;  // Flip now to cancel out later flip
  std::vector<uint8_t> expected_output = {0, 4, 8,  pad, 1, 5, 9,  pad,
                                          2, 6, 10, pad, 3, 7, 11, pad};
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        expected_output[2 * i] | expected_output[2 * i + 1] << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {6, 22, -26, 0};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt4Packed, k2DYIsSpatialIOAndXIsOGroupI4O4Pad) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(3, 1, 1, 4);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 12 -- packed
  for (uint8_t i = 0; i < 6; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 2 3      : 6
  // 4 5 6 7      : 4 5 6 7      : 22
  // 8 9 10 11    : -8 -7 -6 -5  : -26

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  uint8_t pad = 8u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i), pad,
                                        /*swap_dims=*/false));

  // Should result in filling to be 4x4 with pad_value, then bit flip 0x8
  pad ^= 0x8;  // Flip now to cancel out later flip
  std::vector<uint8_t> expected_output = {0, 4, 8,  pad, 1, 5, 9,  pad,
                                          2, 6, 10, pad, 3, 7, 11, pad};
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        expected_output[2 * i] | expected_output[2 * i + 1] << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {6, 22, -26, 0};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt4Packed, DoublePad) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 2);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 8 -- packed
  for (uint8_t i = 0; i < 4; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1          : 0 1          : 1
  // 2 3          : 2 3          : 5
  // 4 5          : 4 5          : 9
  // 6 7          : 6 7          : 13

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  uint8_t pad = 8u;
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i), pad,
                                        /*swap_dims=*/false));

  // Here we test the "double_pad"
  pad ^= 0x8;
  std::vector<uint8_t> expected_output = {
      0, 2, 4, 6, 1, 3, 5, 7, pad, pad, pad, pad, pad, pad, pad, pad};
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        expected_output[2 * i] | expected_output[2 * i + 1] << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {1, 5, 9, 13};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt4Packed, SwapDims) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data;
  // Fill weights with 0 to 15 -- packed
  for (uint8_t i = 0; i < 8; ++i) {
    weights_data.push_back(((2 * i + 1) << 4) | 2 * i);
  }
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 4 8 12     : 0 4 -8 -4    : -8
  // 1 5 9 13     : 1 5 -7 -3    : -4
  // 2 6 10 14    : 2 6 -6 -2    : 0
  // 3 7 11 15    : 3 7 -5 -1    : 4

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/8u, /*swap_dims=*/true));

  // Swap_dims cancels transpose; just bit flip 0x8
  std::vector<uint8_t> expected_output_packed(8);
  for (int i = 0; i < 8; ++i) {
    expected_output_packed[i] =
        (2 * i) | (2 * i + 1) << 4;
    expected_output_packed[i] ^= 0x88;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {-8, -4, 0, 4};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

std::vector<uint8_t> PackInt2IntoUint8(std::vector<uint8_t> input) {
  std::vector<uint8_t> output;
  output.reserve((input.size() + 3) / 4);
  for (int i = 0; i < input.size(); i += 4) {
    output.push_back(input[i] | input[i + 1] << 2 | input[i + 2] << 4 |
                     input[i + 3] << 6);
  }
  return output;
}

TEST(RearrangeWeightsUInt2Packed, kOSpatialIOGroupI4O4Base) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2, 1, 3, 0, 1, 2});
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 -2 -1    : -2
  // 3 1 2 3      : -1 1 -2 -1   : -3
  // 0 1 2 1      : 0 1 -2 1     : 0
  // 3 0 1 2      : -1 0 1 -2    : -2
  weights.data = std::move(weights_data);

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/2u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x2
  std::vector<uint8_t> expected_unpacked_output = {0, 3, 0, 3, 1, 1, 1, 0,
                                                   2, 2, 2, 1, 3, 3, 1, 2};
  std::vector<uint8_t> expected_output_packed =
      PackInt2IntoUint8(expected_unpacked_output);
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {-2, -3, 0, -2};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt2Packed, kOSpatialIOGroupI4O4ZeroWeights) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2, 1, 3, 0, 1, 2});
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 -2 -1    : -2
  // 3 1 2 3      : -1 1 -2 -1   : -3
  // 0 1 2 1      : 0 1 -2 1     : 0
  // 3 0 1 2      : -1 0 1 -2    : -2

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/2u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x2
  std::vector<uint8_t> expected_unpacked_output = {0, 3, 0, 3, 1, 1, 1, 0,
                                                   2, 2, 2, 1, 3, 3, 1, 2};
  std::vector<uint8_t> expected_output_packed =
      PackInt2IntoUint8(expected_unpacked_output);
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {-2, -3, 0, -2};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt2Packed, k2DYIsSpatialIOAndXIsOGroupI4O4ZeroWeights) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2, 1, 3, 0, 1, 2});
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 -2 -1    : -2
  // 3 1 2 3      : -1 1 -2 -1   : -3
  // 0 1 2 1      : 0 1 -2 1     : 0
  // 3 0 1 2      : -1 0 1 -2    : -2

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/2u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x8
  std::vector<uint8_t> expected_unpacked_output = {0, 3, 0, 3, 1, 1, 1, 0,
                                                   2, 2, 2, 1, 3, 3, 1, 2};
  std::vector<uint8_t> expected_output_packed =
      PackInt2IntoUint8(expected_unpacked_output);
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {-2, -3, 0, -2};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

uint8_t Int2AsUint2(int8_t value) {
  return value == 0 ? 0 : value == 1 ? 1 : value == -1 ? 3 : 2;
}

TEST(RearrangeWeightsUInt2Packed, kOSpatialIOGroupI4O4BaseVersusRef) {
  Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(4, 1, 1, 4);
  weights_i8.data = {0, 1, -2, -1, 1, -2, -1, 0, -2, -1, 0, 1, -1, 0, 1, -2};

  std::vector<uint8_t> weights_i8_unpacked;
  for (auto d : weights_i8.data) {
    weights_i8_unpacked.push_back(Int2AsUint2(d));
  }
  std::vector<uint8_t> weights_i8_packed =
      PackInt2IntoUint8(weights_i8_unpacked);

  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/2u, /*swap_dims=*/false));

  std::vector<uint8_t> data_ref(weights_i8.shape.DimensionsProduct() / 4);
  RearrangeWeightsInt8AsUint2(weights_i8, dst_weight_desc,
                              absl::MakeSpan(data_ref), 2, 2u);
  EXPECT_THAT(output, ElementsAreArray(data_ref));
  auto expected_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i.data));
}

TEST(RearrangeWeightsUInt2Packed,
     k2DYIsSpatialIOAndXIsOGroupI4O4BaseVersusRef) {
  Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(16, 2, 1, 8);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = (i % 4) - 2;
  }

  std::vector<uint8_t> weights_i8_unpacked;
  for (auto d : weights_i8.data) {
    weights_i8_unpacked.push_back(Int2AsUint2(d));
  }
  std::vector<uint8_t> weights_i8_packed =
      PackInt2IntoUint8(weights_i8_unpacked);

  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(16, 2, 1, 8);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 2;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape));
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/2u, /*swap_dims=*/false));

  std::vector<uint8_t> data_ref(output.size());
  RearrangeWeightsInt8AsUint2(weights_i8, dst_weight_desc,
                              absl::MakeSpan(data_ref), 2, 2u);
  EXPECT_THAT(output, ElementsAreArray(data_ref));
  auto expected_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i.data));
}

TEST(RearrangeWeightsUInt2Packed, k2DYIsSpatialIOAndXIsOGroupI4O4Base) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2, 1, 3, 0, 1, 2});
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 -2 -1    : -2
  // 3 1 2 3      : -1 1 -2 -1   : -3
  // 0 1 2 1      : 0 1 -2 1     : 0
  // 3 0 1 2      : -1 0 1 -2    : -2

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/2u, /*swap_dims=*/false));

  // Should result in transpose, and bit flip 0x8
  std::vector<uint8_t> expected_unpacked_output = {0, 3, 0, 3, 1, 1, 1, 0,
                                                   2, 2, 2, 1, 3, 3, 1, 2};
  std::vector<uint8_t> expected_output_packed =
      PackInt2IntoUint8(expected_unpacked_output);
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {-2, -3, 0, -2};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt2Packed, kOSpatialIOGroupI4O4Pad) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(3, 1, 1, 4);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2, 1});
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 -2 -1    : -2
  // 3 1 2 3      : -1 1 -2 -1   : -3
  // 0 1 2 1      : 0 1 -2 1     : 0

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  uint8_t pad = 2u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i), pad,
                                        /*swap_dims=*/false));

  // Should result in filling to be 4x4 with pad_value, then bit flip 0x2
  pad ^= 0x2;  // Flip now to cancel out later flip
  std::vector<uint8_t> expected_unpacked_output = {0, 3, 0, pad, 1, 1, 1, pad,
                                                   2, 2, 2, pad, 3, 3, 1, pad};
  std::vector<uint8_t> expected_output_packed =
      PackInt2IntoUint8(expected_unpacked_output);
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {-2, -3, 0, 0};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt2Packed, k2DYIsSpatialIOAndXIsOGroupI4O4Pad) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(3, 1, 1, 4);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2, 1});
  weights.data = std::move(weights_data);
  // weights OI   : signed       : sum_i
  // 0 1 2 3      : 0 1 -2 -1    : -2
  // 3 1 2 3      : -1 1 -2 -1   : -3
  // 0 1 2 1      : 0 1 -2 1     : 0

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  uint8_t pad = 2u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i), pad,
                                        /*swap_dims=*/false));

  // Should result in filling to be 4x4 with pad_value, then bit flip 0x2
  pad ^= 0x2;  // Flip now to cancel out later flip
  std::vector<uint8_t> expected_unpacked_output = {0, 3, 0, pad, 1, 1, 1, pad,
                                                   2, 2, 2, pad, 3, 3, 1, pad};
  std::vector<uint8_t> expected_output_packed =
      PackInt2IntoUint8(expected_unpacked_output);
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {-2, -3, 0, 0};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt2Packed, DoublePad) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 2);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 0, 1, 2, 3});
  weights.data = std::move(weights_data);
  // weights OI   : signed:' sum_i
  // 0 1          : 0 1  : 1
  // 2 3          : -2 -1: -3
  // 0 1          : 0 1  : 1
  // 2 3          : -2 -1: -3

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  uint8_t pad = 2u;
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i), pad,
                                        /*swap_dims=*/false));

  // Here we test the "quad_pad"
  pad ^= 0x2;
  std::vector<uint8_t> expected_unpacked_output = {
      0, 2, 0, 2, 1, 3, 1, 3, pad, pad, pad, pad, pad, pad, pad, pad};
  std::vector<uint8_t> expected_output_packed =
      PackInt2IntoUint8(expected_unpacked_output);
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {1, -3, 1, -3};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(RearrangeWeightsUInt2Packed, SwapDims) {
  // Prep weights
  Tensor<OHWI, DataType::kUint8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  std::vector<uint8_t> weights_data =
      PackInt2IntoUint8({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
  weights.data = std::move(weights_data);
  // weights OI   : signed sum_i
  // 0 0 0 0      : 0 0 0 0 : 0
  // 1 1 1 1      : 1 1 1 1 : 4
  // 2 2 2 2      :-2-2-2-2 : -8
  // 3 3 3 3      :-1-1-1-1 : -4

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::kUint8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  ABSL_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
                                        absl::MakeSpan(output),
                                        absl::MakeSpan(weights_sum_i),
                                        /*pad_value=*/2u, /*swap_dims=*/true));

  // Swap_dims cancels transpose; just bit flip 0x2
  std::vector<uint8_t> expected_output_packed = weights.data;
  for (int i = 0; i < expected_output_packed.size(); ++i) {
    expected_output_packed[i] ^= 0xAA;
  }
  EXPECT_THAT(output, ElementsAreArray(expected_output_packed));
  std::vector<int32_t> expected_sum_i = {0, 4, -8, -4};
  EXPECT_THAT(weights_sum_i, ElementsAreArray(expected_sum_i));
}

TEST(ISpatialOI4O4UnalignedIO, NonIterativeIndexCalculationNoSpatial) {
  const OHWI weights_shape(7, 1, 1, 9);
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  int counter = 0;
  for (int src_s = 0; src_s < src_slices; ++src_s) {
    for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
      const int src_ch_count = std::min(4, weights_shape.i - src_s * 4);
      const int dst_ch_count = std::min(4, weights_shape.o - dst_s * 4);
      for (int j = 0; j < src_ch_count; ++j) {
        for (int i = 0; i < dst_ch_count; ++i) {
          int index_in_block = j * dst_ch_count + i;
          int ind = src_s * weights_shape.o * 4 + dst_s * 4 * src_ch_count +
                    index_in_block;
          EXPECT_EQ(counter, ind);
          counter += 1;
        }
      }
    }
  }
}

TEST(ISpatialOI4O4UnalignedIO, NonIterativeIndexCalculation) {
  const OHWI weights_shape(7, 1, 3, 6);
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  const int dst_x4 = weights_shape.o / 4;
  const int dst_last_block_size = weights_shape.o - dst_x4 * 4;
  int counter = 0;
  for (int src_s = 0; src_s < src_slices; ++src_s) {
    for (int y = 0; y < weights_shape.h; ++y) {
      for (int x = 0; x < weights_shape.w; ++x) {
        for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
          const int src_ch_count = std::min(4, weights_shape.i - src_s * 4);
          const int dst_ch_count = std::min(4, weights_shape.o - dst_s * 4);
          for (int j = 0; j < src_ch_count; ++j) {
            for (int i = 0; i < dst_ch_count; ++i) {
              int index_in_block = j * dst_ch_count + i;
              int ind = src_s * weights_shape.w * weights_shape.o * 4 +
                        (x * dst_x4 + dst_s) * 4 * src_ch_count +
                        x * dst_last_block_size * src_ch_count + index_in_block;
              EXPECT_EQ(counter, ind);
              counter += 1;
            }
          }
        }
      }
    }
  }
}

TEST(ISpatialOI4O4UnalignedIO, CalculateWeightsCoordsFromIndexNoSpatial) {
  const OHWI weights_shape(7, 1, 1, 11);
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  const int src_x4 = weights_shape.i / 4;
  const int dst_x4 = weights_shape.o / 4;
  const int src_last_block_size = weights_shape.i - src_x4 * 4;
  const int dst_last_block_size = weights_shape.o - dst_x4 * 4;
  int counter = 0;
  for (int src_s = 0; src_s < src_slices; ++src_s) {
    for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
      const int src_ch_count = std::min(4, weights_shape.i - src_s * 4);
      const int dst_ch_count = std::min(4, weights_shape.o - dst_s * 4);
      for (int j = 0; j < src_ch_count; ++j) {
        for (int i = 0; i < dst_ch_count; ++i) {
          const int s_ch = src_s * 4 + j;
          const int d_ch = dst_s * 4 + i;
          int index_in_block = j * dst_ch_count + i;
          int ind = src_s * weights_shape.o * 4 + dst_s * 4 * src_ch_count +
                    index_in_block;
          EXPECT_EQ(counter, ind);
          counter += 1;
          int ich = -1;
          int och = -1;
          if (ind >= src_x4 * weights_shape.o * 4) {
            ind = ind - src_x4 * weights_shape.o * 4;
            // ind = dst_s * 4 * src_ch_count + j * dst_ch_count + i
            if (ind >= dst_x4 * 4 * src_last_block_size) {
              ind = ind - dst_x4 * 4 * src_last_block_size;
              // ind = j * dst_ch_count + i
              int i0 = ind % dst_last_block_size;
              int j0 = ind / dst_last_block_size;
              och = dst_x4 * 4 + i0;
              ich = src_x4 * 4 + j0;
            } else {
              int block_size = src_last_block_size * 4;
              int dst_s = ind / block_size;
              int block_id = ind % block_size;
              int i0 = block_id % 4;
              int j0 = block_id / 4;
              och = dst_s * 4 + i0;
              ich = src_x4 * 4 + j0;
            }
          } else {
            int src_s = ind / (weights_shape.o * 4);
            ind = ind % (weights_shape.o * 4);
            if (ind >= dst_x4 * 16) {
              ind = ind - dst_x4 * 16;
              // ind = j * dst_ch_count + i
              int i0 = ind % dst_last_block_size;
              int j0 = ind / dst_last_block_size;
              och = dst_x4 * 4 + i0;
              ich = src_s * 4 + j0;
            } else {
              int dst_s = ind / 16;
              int block_id = ind % 16;
              int i0 = block_id % 4;
              int j0 = block_id / 4;
              och = dst_s * 4 + i0;
              ich = src_s * 4 + j0;
            }
          }
          EXPECT_EQ(ich, s_ch);
          EXPECT_EQ(och, d_ch);
        }
      }
    }
  }
}

TEST(ISpatialOI4O4UnalignedIO, CalculateWeightsCoordsFromIndex) {
  const OHWI weights_shape(7, 1, 3, 5);
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  const int src_x4 = weights_shape.i / 4;
  const int dst_x4 = weights_shape.o / 4;
  const int src_last_block_size = weights_shape.i - src_x4 * 4;
  const int dst_last_block_size = weights_shape.o - dst_x4 * 4;
  int counter = 0;
  for (int src_s = 0; src_s < src_slices; ++src_s) {
    for (int y = 0; y < weights_shape.h; ++y) {
      for (int x = 0; x < weights_shape.w; ++x) {
        for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
          const int src_ch_count = std::min(4, weights_shape.i - src_s * 4);
          const int dst_ch_count = std::min(4, weights_shape.o - dst_s * 4);
          for (int j = 0; j < src_ch_count; ++j) {
            for (int i = 0; i < dst_ch_count; ++i) {
              const int s_ch = src_s * 4 + j;
              const int d_ch = dst_s * 4 + i;
              int index_in_block = j * dst_ch_count + i;
              int ind = src_s * weights_shape.w * weights_shape.o * 4 +
                        (x * dst_x4 + dst_s) * 4 * src_ch_count +
                        x * dst_last_block_size * src_ch_count + index_in_block;
              EXPECT_EQ(counter, ind);
              counter += 1;
              int ich = -1;
              int och = -1;
              int xc = -1;
              if (ind >= src_x4 * weights_shape.o * weights_shape.w * 4) {
                ind = ind - src_x4 * weights_shape.o * weights_shape.w * 4;
                // ind = (x * dst_x4 + dst_s) * 4 * src_ch_count +
                //   x * dst_last_block_size * src_ch_count + j * dst_ch_count
                //   + i
                xc = ind / (weights_shape.o * src_last_block_size);
                ind = ind % (weights_shape.o * src_last_block_size);
                if (ind >= dst_x4 * 4 * src_last_block_size) {
                  ind = ind - dst_x4 * 4 * src_last_block_size;
                  // ind = j * dst_ch_count + i
                  int i0 = ind % dst_last_block_size;
                  int j0 = ind / dst_last_block_size;
                  och = dst_x4 * 4 + i0;
                  ich = src_x4 * 4 + j0;
                } else {
                  int block_size = src_last_block_size * 4;
                  int dst_s = ind / block_size;
                  int block_id = ind % block_size;
                  int i0 = block_id % 4;
                  int j0 = block_id / 4;
                  och = dst_s * 4 + i0;
                  ich = src_x4 * 4 + j0;
                }
              } else {
                int src_s = ind / (weights_shape.o * weights_shape.w * 4);
                ind = ind % (weights_shape.o * weights_shape.w * 4);
                xc = ind / (weights_shape.o * 4);
                ind = ind % (weights_shape.o * 4);
                if (ind >= dst_x4 * 16) {
                  ind = ind - dst_x4 * 16;
                  // ind = j * dst_ch_count + i
                  int i0 = ind % dst_last_block_size;
                  int j0 = ind / dst_last_block_size;
                  och = dst_x4 * 4 + i0;
                  ich = src_s * 4 + j0;
                } else {
                  int dst_s = ind / 16;
                  int block_id = ind % 16;
                  int i0 = block_id % 4;
                  int j0 = block_id / 4;
                  och = dst_s * 4 + i0;
                  ich = src_s * 4 + j0;
                }
              }
              EXPECT_EQ(ich, s_ch);
              EXPECT_EQ(och, d_ch);
              EXPECT_EQ(xc, x);
            }
          }
        }
      }
    }
  }
}

TEST(CustomGroups, TotalSizeOIO4I4) {
  const OHWI weights_shape(17, 1, 1, 31);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {{Axis::kInputChannels, 4},
                             {Axis::kOutputChannels, 4},
                             {Axis::kInputChannels, 0},
                             {Axis::kOutputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, weights_shape),
            AlignByN(weights_shape.o, 4) * AlignByN(weights_shape.i, 4));
}

TEST(CustomGroups, TotalSizeIOI4O3I2) {
  const OHWI weights_shape(17, 1, 1, 31);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {{Axis::kInputChannels, 2},
                             {Axis::kOutputChannels, 3},
                             {Axis::kInputChannels, 4},
                             {Axis::kOutputChannels, 0},
                             {Axis::kInputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, weights_shape),
            AlignByN(weights_shape.o, 3) * AlignByN(weights_shape.i, 8));
}

TEST(CustomGroups, TotalSizeIOI4SpatialO2) {
  const OHWI weights_shape(17, 2, 3, 31);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {
      {Axis::kOutputChannels, 2}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},         {Axis::kInputChannels, 4},
      {Axis::kOutputChannels, 0}, {Axis::kInputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, weights_shape),
            AlignByN(weights_shape.o, 2) * AlignByN(weights_shape.i, 4) *
                weights_shape.h * weights_shape.w);
}

namespace {
void RunConversions(const OHWI& weights_shape,
                    const WeightsDescription& weight_desc,
                    const WeightsDescription& weight_desc_ref) {
  const int total_elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_shape);
  const int total_elements_count_ref =
      GetTotalElementsCountForLayout(weight_desc_ref, weights_shape);

  Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = i;
  }

  std::vector<float> output(total_elements_count, 0.0f);
  RearrangeWeights(weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(output.data()),
                                  output.size() * sizeof(float)));

  std::vector<float> output_ref(total_elements_count_ref, 0.0f);
  RearrangeWeights(weights, weight_desc_ref,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(output_ref.data()),
                                  output_ref.size() * sizeof(float)));

  EXPECT_EQ(output, output_ref);
}

void RunConversions(const OHWDI& weights_shape,
                    const WeightsDescription& weight_desc,
                    const WeightsDescription& weight_desc_ref) {
  const int total_elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_shape);
  const int total_elements_count_ref =
      GetTotalElementsCountForLayout(weight_desc_ref, weights_shape);

  Tensor<OHWDI, DataType::kFloat32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = i;
  }

  std::vector<float> output(total_elements_count, 0.0f);
  RearrangeWeights(weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(output.data()),
                                  output.size() * sizeof(float)));

  std::vector<float> output_ref(total_elements_count_ref, 0.0f);
  RearrangeWeights(weights, weight_desc_ref,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(output_ref.data()),
                                  output_ref.size() * sizeof(float)));

  EXPECT_EQ(output, output_ref);
}
}  // namespace

TEST(CustomGroups, ConversionOHWIToOSpatialIOGroupI4O4) {
  const OHWI weights_shape(39, 2, 3, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {
      {Axis::kOutputChannels, 4}, {Axis::kInputChannels, 4},
      {Axis::kOutputChannels, 4}, {Axis::kInputChannels, 0},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kOutputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::kFloat32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWDIToOSpatialIOGroupI4O4) {
  const OHWDI weights_shape(39, 2, 3, 4, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {
      {Axis::kOutputChannels, 4}, {Axis::kInputChannels, 4},
      {Axis::kOutputChannels, 4}, {Axis::kInputChannels, 0},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kDepth, 0},          {Axis::kOutputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::kFloat32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWIToOSpatialIOGroupO4I4) {
  const OHWI weights_shape(39, 2, 3, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 4},  {Axis::kOutputChannels, 4},
      {Axis::kOutputChannels, 4}, {Axis::kInputChannels, 0},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kOutputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::kFloat32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupO4I4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWDIToOSpatialIOGroupO4I4) {
  const OHWDI weights_shape(39, 2, 3, 4, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 4},  {Axis::kOutputChannels, 4},
      {Axis::kOutputChannels, 4}, {Axis::kInputChannels, 0},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kDepth, 0},          {Axis::kOutputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::kFloat32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupO4I4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

void RearrangeWeightsToOSpatialIOGroupITileOTile(
    const Tensor<OHWI, DataType::kFloat32>& weights, int i_tile_size,
    int o_tile_size, int out_group_size, absl::Span<float> dst,
    float pad_value) {
  const int dst_slices = DivideRoundUp(weights.shape.o, o_tile_size);
  const int src_slices = DivideRoundUp(weights.shape.i, i_tile_size);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  const auto* src = weights.Data();
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d_group = 0; d_group < out_group_size; ++d_group) {
            for (int i = 0; i < i_tile_size; ++i) {
              for (int j = 0; j < o_tile_size; ++j) {
                const int s_ch = s * i_tile_size + i;
                const int d_ch =
                    (d * out_group_size + d_group) * o_tile_size + j;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                  const int f_index =
                      weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  dst[counter++] = src[f_index];
                } else {
                  dst[counter++] = pad_value;
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(CustomGroups, ConversionOHWIToOSpatialIOGroupITileOTile) {
  const OHWI weights_shape(139, 2, 3, 123);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {
      {Axis::kOutputChannels, 8}, {Axis::kInputChannels, 8},
      {Axis::kOutputChannels, 2}, {Axis::kInputChannels, 0},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kOutputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  const int total_elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_shape);

  Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = i;
  }

  std::vector<float> output(total_elements_count, 0.0f);
  RearrangeWeights(weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(output.data()),
                                  output.size() * sizeof(float)));

  std::vector<float> output_ref(total_elements_count, 0.0f);
  RearrangeWeightsToOSpatialIOGroupITileOTile(weights, 8, 8, 2,
                                              absl::MakeSpan(output_ref), 0.0f);

  EXPECT_EQ(output, output_ref);
}

void RearrangeWeightsToOICustomSpatialI4O4(
    const Tensor<OHWI, DataType::kFloat32>& weights,
    const std::vector<int>& spatial_remap, absl::Span<float> dst) {
  const float pad_value = 0.0f;
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  const auto* src = weights.Data();
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int y = 0; y < weights.shape.h; ++y) {
        for (int x = 0; x < weights.shape.w; ++x) {
          const int kernel_index = spatial_remap[y * weights.shape.w + x];
          const int kernel_index_x = kernel_index % weights.shape.w;
          const int kernel_index_y = kernel_index / weights.shape.w;
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              const int s_ch = s * 4 + i;
              const int d_ch = d * 4 + j;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index = weights.shape.LinearIndex(
                    {d_ch, kernel_index_y, kernel_index_x, s_ch});
                dst[counter++] = src[f_index];
              } else {
                dst[counter++] = pad_value;
              }
            }
          }
        }
      }
    }
  }
}

TEST(CustomGroups, ConversionOHWITokOICustomSpatialI4O4) {
  const OHWI weights_shape(139, 3, 3, 123);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.group_sizes = {
      {Axis::kOutputChannels, 4}, {Axis::kInputChannels, 4},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kInputChannels, 0},  {Axis::kOutputChannels, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;
  weight_desc.spatial_remap = {4, 5, 3, 7, 1, 8, 6, 2, 0};

  const int total_elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_shape);

  Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = i;
  }

  std::vector<float> output(total_elements_count, 0.0f);
  RearrangeWeights(weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(output.data()),
                                  output.size() * sizeof(float)));

  std::vector<float> output_ref(total_elements_count, 0.0f);
  RearrangeWeightsToOICustomSpatialI4O4(weights, weight_desc.spatial_remap,
                                        absl::MakeSpan(output_ref));

  EXPECT_EQ(output, output_ref);
}

TEST(CustomGroupsI2gr0O8I2gr1O2, TotalSize) {
  const OHWI ohwi_shape(39, 2, 3, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.layout = WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2;
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 3},  {Axis::kOutputChannels, 2},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kOutputChannels, 0}, {Axis::kInputChannels, 0}};
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, ohwi_shape),
            AlignByN(ohwi_shape.o, 16 * 2) * AlignByN(ohwi_shape.i, 4 * 3) *
                ohwi_shape.h * ohwi_shape.w);

  const OHWDI ohwdi_shape(39, 2, 3, 5, 23);
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 3}, {Axis::kOutputChannels, 2},
      {Axis::kWidth, 0},         {Axis::kHeight, 0},
      {Axis::kDepth, 0},         {Axis::kOutputChannels, 0},
      {Axis::kInputChannels, 0}};
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, ohwdi_shape),
            AlignByN(ohwdi_shape.o, 16 * 2) * AlignByN(ohwdi_shape.i, 4 * 3) *
                ohwdi_shape.h * ohwdi_shape.w * ohwdi_shape.d);
}

TEST(CustomGroupsI2gr0O8I2gr1O2, ConversionOHWIAllTypes) {
  const OHWI weights_shape(39, 2, 3, 23);
  constexpr int kOutputGroupSize = 2;
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.layout = WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2;
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 2}, {Axis::kOutputChannels, kOutputGroupSize},
      {Axis::kInputChannels, 0}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},        {Axis::kOutputChannels, 0}};

  const int elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_shape);
  const int src_groups = DivideRoundUp(weights_shape.i, 8);
  const int dst_groups = DivideRoundUp(weights_shape.o, 16 * kOutputGroupSize);

  auto build_ref = [&](const auto& src_data, auto pad_val) {
    using ElemT = decltype(pad_val);
    std::vector<ElemT> ref(elements_count, pad_val);
    int counter = 0;
    for (int dst_group = 0; dst_group < dst_groups; ++dst_group) {
      for (int ky = 0; ky < weights_shape.h; ++ky) {
        for (int kx = 0; kx < weights_shape.w; ++kx) {
          for (int src_group = 0; src_group < src_groups; ++src_group) {
            for (int dst_block = 0; dst_block < kOutputGroupSize; ++dst_block) {
              const int i_offset = src_group * 8;
              const int o_offset =
                  (dst_group * kOutputGroupSize + dst_block) * 16;
              for (int i2 = 0; i2 < 2; ++i2) {
                for (int i1 = 0; i1 < 2; ++i1) {
                  for (int o1 = 0; o1 < 8; ++o1) {
                    for (int i0 = 0; i0 < 2; ++i0) {
                      for (int o0 = 0; o0 < 2; ++o0) {
                        const int i_ch = i_offset + (i2 * 2 + i0) * 2 + i1;
                        const int o_ch = o_offset + o1 * 2 + o0;
                        if (o_ch < weights_shape.o && i_ch < weights_shape.i) {
                          ref[counter++] = static_cast<ElemT>(
                              src_data[weights_shape.LinearIndex(
                                  {o_ch, ky, kx, i_ch})]);
                        } else {
                          ref[counter++] = pad_val;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return ref;
  };

  // Float32 and Float16 conversions.
  Tensor<OHWI, DataType::kFloat32> f32_weights;
  f32_weights.shape = weights_shape;
  f32_weights.data.resize(weights_shape.DimensionsProduct() +
                          XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < f32_weights.data.size(); ++i) {
    f32_weights.data[i] = static_cast<float>(i % 31);
  }
  std::vector<float> f32_out(elements_count, -1.0f);
  RearrangeWeights(f32_weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(f32_out.data()),
                                  f32_out.size() * sizeof(float)));
  EXPECT_EQ(f32_out, build_ref(f32_weights.data, 0.0f));

  weight_desc.type = DataType::kFloat16;
  std::vector<half> f16_out(elements_count, half(-1.0f));
  RearrangeWeights(f32_weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(f16_out.data()),
                                  f16_out.size() * sizeof(half)));
  EXPECT_EQ(f16_out, build_ref(f32_weights.data, half(0.0f)));

  // Int8 conversion.
  weight_desc.type = DataType::kInt8;
  Tensor<OHWI, DataType::kInt8> i8_weights;
  i8_weights.shape = weights_shape;
  i8_weights.data.resize(weights_shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < i8_weights.data.size(); ++i) {
    i8_weights.data[i] = static_cast<int8_t>((i % 127) - 63);
  }
  std::vector<int8_t> i8_out(elements_count, 1);
  RearrangeWeights(
      i8_weights, weight_desc,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(i8_out.data()), i8_out.size()));
  EXPECT_EQ(i8_out, build_ref(i8_weights.data, static_cast<int8_t>(0)));

  // Uint8 conversion with non-zero pad_value.
  weight_desc.type = DataType::kUint8;
  Tensor<OHWI, DataType::kUint8> u8_weights;
  u8_weights.shape = weights_shape;
  u8_weights.data.resize(weights_shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));
  for (int i = 0; i < u8_weights.data.size(); ++i) {
    u8_weights.data[i] = static_cast<uint8_t>(i % 251);
  }
  std::vector<uint8_t> u8_out(elements_count, 0);
  RearrangeWeights(u8_weights, weight_desc, absl::MakeSpan(u8_out),
                   /*pad_value=*/128);
  EXPECT_EQ(u8_out, build_ref(u8_weights.data, static_cast<uint8_t>(128)));
}

TEST(CustomGroupsI2gr0O8I2gr1O2, ConversionOHWDI) {
  const OHWI ohwi_shape(39, 2, 3, 23);
  const OHWDI ohwdi_1_shape(39, 2, 3, 1, 23);
  const OHWDI ohwdi_shape(39, 2, 3, 4, 23);
  constexpr int kOutputGroupSize = 2;

  WeightsDescription ohwi_desc;
  ohwi_desc.type = DataType::kFloat32;
  ohwi_desc.layout = WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2;
  ohwi_desc.group_sizes = {
      {Axis::kInputChannels, 2}, {Axis::kOutputChannels, kOutputGroupSize},
      {Axis::kInputChannels, 0}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},        {Axis::kOutputChannels, 0}};

  WeightsDescription ohwdi_desc = ohwi_desc;
  ohwdi_desc.group_sizes = {
      {Axis::kInputChannels, 2}, {Axis::kOutputChannels, kOutputGroupSize},
      {Axis::kInputChannels, 0}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},        {Axis::kDepth, 0},
      {Axis::kOutputChannels, 0}};

  Tensor<OHWI, DataType::kFloat32> ohwi_weights;
  ohwi_weights.shape = ohwi_shape;
  ohwi_weights.data.resize(ohwi_shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < ohwi_weights.data.size(); ++i) {
    ohwi_weights.data[i] = static_cast<float>(i);
  }

  Tensor<OHWDI, DataType::kFloat32> ohwdi_1_weights;
  ohwdi_1_weights.shape = ohwdi_1_shape;
  ohwdi_1_weights.data = ohwi_weights.data;

  const int ohwi_elements =
      GetTotalElementsCountForLayout(ohwi_desc, ohwi_shape);
  ASSERT_EQ(ohwi_elements,
            GetTotalElementsCountForLayout(ohwdi_desc, ohwdi_1_shape));

  std::vector<float> ohwi_out(ohwi_elements, -1.0f);
  RearrangeWeights(ohwi_weights, ohwi_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(ohwi_out.data()),
                                  ohwi_out.size() * sizeof(float)));

  std::vector<float> ohwdi_1_out(ohwi_elements, -2.0f);
  RearrangeWeights(
      ohwdi_1_weights, ohwdi_desc,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(ohwdi_1_out.data()),
                     ohwdi_1_out.size() * sizeof(float)));
  EXPECT_EQ(ohwi_out, ohwdi_1_out);

  Tensor<OHWDI, DataType::kFloat32> ohwdi_weights;
  ohwdi_weights.shape = ohwdi_shape;
  ohwdi_weights.data.resize(ohwdi_shape.DimensionsProduct() +
                            XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < ohwdi_weights.data.size(); ++i) {
    ohwdi_weights.data[i] = static_cast<float>(i % 31);
  }

  const int ohwdi_elements =
      GetTotalElementsCountForLayout(ohwdi_desc, ohwdi_shape);
  std::vector<float> ohwdi_out(ohwdi_elements, -1.0f);
  RearrangeWeights(ohwdi_weights, ohwdi_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(ohwdi_out.data()),
                                  ohwdi_out.size() * sizeof(float)));

  std::vector<float> ohwdi_ref(ohwdi_elements, -2.0f);
  const int src_groups = DivideRoundUp(ohwdi_shape.i, 8);
  const int dst_groups = DivideRoundUp(ohwdi_shape.o, 16 * kOutputGroupSize);
  int counter = 0;
  for (int dst_group = 0; dst_group < dst_groups; ++dst_group) {
    for (int kz = 0; kz < ohwdi_shape.d; ++kz) {
      for (int ky = 0; ky < ohwdi_shape.h; ++ky) {
        for (int kx = 0; kx < ohwdi_shape.w; ++kx) {
          for (int src_group = 0; src_group < src_groups; ++src_group) {
            for (int dst_block = 0; dst_block < kOutputGroupSize; ++dst_block) {
              const int i_offset = src_group * 8;
              const int o_offset =
                  (dst_group * kOutputGroupSize + dst_block) * 16;
              for (int i2 = 0; i2 < 2; ++i2) {
                for (int i1 = 0; i1 < 2; ++i1) {
                  for (int o1 = 0; o1 < 8; ++o1) {
                    for (int i0 = 0; i0 < 2; ++i0) {
                      for (int o0 = 0; o0 < 2; ++o0) {
                        const int i_ch = i_offset + (i2 * 2 + i0) * 2 + i1;
                        const int o_ch = o_offset + o1 * 2 + o0;
                        if (o_ch < ohwdi_shape.o && i_ch < ohwdi_shape.i) {
                          ohwdi_ref[counter++] =
                              ohwdi_weights.data[ohwdi_shape.LinearIndex(
                                  {o_ch, ky, kx, kz, i_ch})];
                        } else {
                          ohwdi_ref[counter++] = 0.0f;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(ohwdi_out, ohwdi_ref);
}

TEST(CustomGroupsI2gr0O8I2gr1O2, ConversionArbitraryOrderAndSpatialRemap) {
  // Innermost-to-outermost outside I2gr0O8I2gr1O2:
  // o_tile (2) -> h (2) -> w (3) -> o_outer -> i_outer, with spatial_remap.
  const OHWI ohwi_shape(35, 2, 3, 19);
  WeightsDescription ohwi_desc;
  ohwi_desc.type = DataType::kFloat32;
  ohwi_desc.layout = WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2;
  ohwi_desc.group_sizes = {{Axis::kOutputChannels, 2},
                           {Axis::kHeight, 0},
                           {Axis::kWidth, 0},
                           {Axis::kOutputChannels, 0},
                           {Axis::kInputChannels, 0}};
  ohwi_desc.spatial_remap = {5, 1, 3, 0, 4, 2};

  Tensor<OHWI, DataType::kFloat32> ohwi_weights;
  ohwi_weights.shape = ohwi_shape;
  ohwi_weights.data.resize(ohwi_shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < ohwi_weights.data.size(); ++i) {
    ohwi_weights.data[i] = static_cast<float>(i + 1);
  }

  const int ohwi_elements =
      GetTotalElementsCountForLayout(ohwi_desc, ohwi_shape);
  std::vector<float> ohwi_out(ohwi_elements, -1.0f);
  RearrangeWeights(ohwi_weights, ohwi_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(ohwi_out.data()),
                                  ohwi_out.size() * sizeof(float)));

  std::vector<float> ohwi_ref(ohwi_elements, -2.0f);
  const int i_blocks = DivideRoundUp(ohwi_shape.i, 4);
  const int o_outer_blocks = DivideRoundUp(ohwi_shape.o, 16 * 2);
  int counter = 0;
  for (int i_blk = 0; i_blk < i_blocks; ++i_blk) {
    for (int o_outer = 0; o_outer < o_outer_blocks; ++o_outer) {
      for (int x = 0; x < ohwi_shape.w; ++x) {
        for (int y = 0; y < ohwi_shape.h; ++y) {
          const int remapped = ohwi_desc.spatial_remap[y * ohwi_shape.w + x];
          const int rx = remapped % ohwi_shape.w;
          const int ry = remapped / ohwi_shape.w;
          for (int o_tile = 0; o_tile < 2; ++o_tile) {
            const int i_base = i_blk * 4;
            const int o_base = (o_outer * 2 + o_tile) * 16;
            for (int i1 = 0; i1 < 2; ++i1) {
              for (int o1 = 0; o1 < 8; ++o1) {
                for (int i0 = 0; i0 < 2; ++i0) {
                  for (int o0 = 0; o0 < 2; ++o0) {
                    const int i_ch = i_base + i0 * 2 + i1;
                    const int o_ch = o_base + o1 * 2 + o0;
                    if (o_ch < ohwi_shape.o && i_ch < ohwi_shape.i) {
                      ohwi_ref[counter++] =
                          ohwi_weights.data[ohwi_shape.LinearIndex(
                              {o_ch, ry, rx, i_ch})];
                    } else {
                      ohwi_ref[counter++] = 0.0f;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(ohwi_out, ohwi_ref);
}

TEST(CustomGroupsOgr1Ogr2Ogr0, TotalSize) {
  const OHWI ohwi_shape(39, 2, 3, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.layout = WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0;
  weight_desc.params0 = {4, 2, 2};
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 3},  {Axis::kOutputChannels, 2},
      {Axis::kWidth, 0},          {Axis::kHeight, 0},
      {Axis::kOutputChannels, 0}, {Axis::kInputChannels, 0}};
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, ohwi_shape),
            AlignByN(ohwi_shape.o, 16 * 2) * AlignByN(ohwi_shape.i, 3) *
                ohwi_shape.h * ohwi_shape.w);

  const OHWDI ohwdi_shape(39, 2, 3, 5, 23);
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 3}, {Axis::kOutputChannels, 2},
      {Axis::kWidth, 0},         {Axis::kHeight, 0},
      {Axis::kDepth, 0},         {Axis::kOutputChannels, 0},
      {Axis::kInputChannels, 0}};
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, ohwdi_shape),
            AlignByN(ohwdi_shape.o, 16 * 2) * AlignByN(ohwdi_shape.i, 3) *
                ohwdi_shape.h * ohwdi_shape.w * ohwdi_shape.d);
}

TEST(CustomGroupsOgr1Ogr2Ogr0, MatchesOSpatialIOGroupI4O4WhenGr1Gr2Are1) {
  WeightsDescription weight_desc;
  weight_desc.type = DataType::kFloat32;
  weight_desc.layout = WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0;
  weight_desc.params0 = {4, 1, 1};
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 4}, {Axis::kOutputChannels, 4},
      {Axis::kInputChannels, 0}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},        {Axis::kOutputChannels, 0}};

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::kFloat32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  RunConversions(OHWI(39, 2, 3, 23), weight_desc, weight_desc_ref);

  weight_desc.group_sizes = {
      {Axis::kInputChannels, 4}, {Axis::kOutputChannels, 4},
      {Axis::kInputChannels, 0}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},        {Axis::kDepth, 0},
      {Axis::kOutputChannels, 0}};
  RunConversions(OHWDI(39, 2, 3, 4, 23), weight_desc, weight_desc_ref);
}

TEST(CustomGroupsOgr1Ogr2Ogr0, RearrangesOHWITransposedGroupsAndPadding) {
  const OHWI weights_shape(6, 1, 1, 3);
  WeightsDescription weight_desc;
  weight_desc.layout = WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0;
  weight_desc.params0 = {2, 2, 2};
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 2}, {Axis::kOutputChannels, 1},
      {Axis::kInputChannels, 0}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},        {Axis::kOutputChannels, 0}};

  ASSERT_EQ(GetTotalElementsCountForLayout(weight_desc, weights_shape), 32u);

  Tensor<OHWI, DataType::kFloat32> weights_f32;
  weights_f32.shape = weights_shape;
  weights_f32.data.resize(weights_shape.DimensionsProduct() +
                          XNN_EXTRA_BYTES / sizeof(float));
  Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = weights_shape;
  weights_i8.data.resize(weights_shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  Tensor<OHWI, DataType::kUint8> weights_u8;
  weights_u8.shape = weights_shape;
  weights_u8.data.resize(weights_shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));
  for (int o = 0; o < weights_shape.o; ++o) {
    for (int i = 0; i < weights_shape.i; ++i) {
      const int idx = weights_shape.LinearIndex({o, 0, 0, i});
      const int val = o * 10 + i + 1;
      weights_f32.data[idx] = static_cast<float>(val);
      weights_i8.data[idx] = static_cast<int8_t>(val);
      weights_u8.data[idx] = static_cast<uint8_t>(val);
    }
  }

  weight_desc.type = DataType::kFloat32;
  std::vector<float> out_f32(32, -1.0f);
  RearrangeWeights(weights_f32, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(out_f32.data()),
                                  out_f32.size() * sizeof(float)));
  EXPECT_THAT(out_f32, ElementsAre(1, 11, 41, 51, 21, 31, 0, 0,  //
                                   2, 12, 42, 52, 22, 32, 0, 0,  //
                                   3, 13, 43, 53, 23, 33, 0, 0,  //
                                   0, 0, 0, 0, 0, 0, 0, 0));

  weight_desc.type = DataType::kFloat16;
  std::vector<half> out_f16(32, half(-1.0f));
  RearrangeWeights(weights_f32, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(out_f16.data()),
                                  out_f16.size() * sizeof(half)));
  const std::vector<half> expected_f16 = {
      half(1),  half(11), half(41), half(51), half(21), half(31), half(0),
      half(0),  half(2),  half(12), half(42), half(52), half(22), half(32),
      half(0),  half(0),  half(3),  half(13), half(43), half(53), half(23),
      half(33), half(0),  half(0),  half(0),  half(0),  half(0),  half(0),
      half(0),  half(0),  half(0),  half(0)};
  EXPECT_THAT(out_f16, ElementsAreArray(expected_f16));

  weight_desc.type = DataType::kInt8;
  std::vector<int8_t> out_i8(32, -1);
  RearrangeWeights(
      weights_i8, weight_desc,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(out_i8.data()), out_i8.size()));
  EXPECT_THAT(out_i8, ElementsAre(1, 11, 41, 51, 21, 31, 0, 0,  //
                                  2, 12, 42, 52, 22, 32, 0, 0,  //
                                  3, 13, 43, 53, 23, 33, 0, 0,  //
                                  0, 0, 0, 0, 0, 0, 0, 0));

  weight_desc.type = DataType::kUint8;
  std::vector<uint8_t> out_u8(32, 0);
  RearrangeWeights(weights_u8, weight_desc, absl::MakeSpan(out_u8),
                   /*pad_value=*/99);
  EXPECT_THAT(out_u8, ElementsAre(1, 11, 41, 51, 21, 31, 99, 99,  //
                                  2, 12, 42, 52, 22, 32, 99, 99,  //
                                  3, 13, 43, 53, 23, 33, 99, 99,  //
                                  99, 99, 99, 99, 99, 99, 99, 99));
}

TEST(CustomGroupsOgr1Ogr2Ogr0, RearrangesOHWDITransposedGroupsAndPadding) {
  const OHWDI weights_shape(5, 1, 2, 2, 2);
  WeightsDescription weight_desc;
  weight_desc.layout = WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0;
  weight_desc.params0 = {1, 2, 2};
  weight_desc.group_sizes = {
      {Axis::kInputChannels, 2}, {Axis::kOutputChannels, 1},
      {Axis::kInputChannels, 0}, {Axis::kWidth, 0},
      {Axis::kHeight, 0},        {Axis::kDepth, 0},
      {Axis::kOutputChannels, 0}};

  ASSERT_EQ(GetTotalElementsCountForLayout(weight_desc, weights_shape), 64u);

  Tensor<OHWDI, DataType::kFloat32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int o = 0; o < weights_shape.o; ++o) {
    for (int kx = 0; kx < weights_shape.w; ++kx) {
      for (int kz = 0; kz < weights_shape.d; ++kz) {
        for (int i = 0; i < weights_shape.i; ++i) {
          weights.data[weights_shape.LinearIndex({o, 0, kx, kz, i})] =
              static_cast<float>(100 * o + 10 * kx + 2 * kz + i + 1);
        }
      }
    }
  }

  weight_desc.type = DataType::kFloat32;
  std::vector<float> out_f32(64, -1.0f);
  RearrangeWeights(weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(out_f32.data()),
                                  out_f32.size() * sizeof(float)));
  const std::vector<float> expected = {
      1,   201, 101, 301, 2,   202, 102, 302,  // dst_group=0, kz=0, kx=0
      11,  211, 111, 311, 12,  212, 112, 312,  // dst_group=0, kz=0, kx=1
      3,   203, 103, 303, 4,   204, 104, 304,  // dst_group=0, kz=1, kx=0
      13,  213, 113, 313, 14,  214, 114, 314,  // dst_group=0, kz=1, kx=1
      401, 0,   0,   0,   402, 0,   0,   0,    // dst_group=1, kz=0, kx=0
      411, 0,   0,   0,   412, 0,   0,   0,    // dst_group=1, kz=0, kx=1
      403, 0,   0,   0,   404, 0,   0,   0,    // dst_group=1, kz=1, kx=0
      413, 0,   0,   0,   414, 0,   0,   0};   // dst_group=1, kz=1, kx=1
  EXPECT_THAT(out_f32, ElementsAreArray(expected));

  weight_desc.type = DataType::kFloat16;
  std::vector<half> out_f16(64, half(-1.0f));
  RearrangeWeights(weights, weight_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(out_f16.data()),
                                  out_f16.size() * sizeof(half)));
  std::vector<half> expected_f16;
  expected_f16.reserve(expected.size());
  for (float v : expected) {
    expected_f16.push_back(half(v));
  }
  EXPECT_THAT(out_f16, ElementsAreArray(expected_f16));
}

TEST(CustomGroupsOgr1Ogr2Ogr0, ConversionArbitraryOrderAndSpatialRemap) {
  const OHWI ohwi_shape(35, 2, 3, 19);
  WeightsDescription ohwi_desc;
  ohwi_desc.type = DataType::kFloat32;
  ohwi_desc.layout = WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0;
  ohwi_desc.params0 = {4, 2, 2};
  ohwi_desc.group_sizes = {{Axis::kInputChannels, 4},
                           {Axis::kOutputChannels, 2},
                           {Axis::kHeight, 0},
                           {Axis::kWidth, 0},
                           {Axis::kOutputChannels, 0},
                           {Axis::kInputChannels, 0}};
  ohwi_desc.spatial_remap = {5, 1, 3, 0, 4, 2};

  Tensor<OHWI, DataType::kFloat32> ohwi_weights;
  ohwi_weights.shape = ohwi_shape;
  ohwi_weights.data.resize(ohwi_shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < ohwi_weights.data.size(); ++i) {
    ohwi_weights.data[i] = static_cast<float>(i + 1);
  }

  const int ohwi_elements =
      GetTotalElementsCountForLayout(ohwi_desc, ohwi_shape);
  std::vector<float> ohwi_out(ohwi_elements, -1.0f);
  RearrangeWeights(ohwi_weights, ohwi_desc,
                   absl::MakeSpan(reinterpret_cast<uint8_t*>(ohwi_out.data()),
                                  ohwi_out.size() * sizeof(float)));

  std::vector<float> ohwi_ref(ohwi_elements, -2.0f);
  const int i_blocks = DivideRoundUp(ohwi_shape.i, 4);
  const int o_outer_blocks = DivideRoundUp(ohwi_shape.o, 16 * 2);
  int counter = 0;
  for (int i_blk = 0; i_blk < i_blocks; ++i_blk) {
    for (int o_outer = 0; o_outer < o_outer_blocks; ++o_outer) {
      for (int x = 0; x < ohwi_shape.w; ++x) {
        for (int y = 0; y < ohwi_shape.h; ++y) {
          const int remapped = ohwi_desc.spatial_remap[y * ohwi_shape.w + x];
          const int rx = remapped % ohwi_shape.w;
          const int ry = remapped / ohwi_shape.w;
          for (int o_tile = 0; o_tile < 2; ++o_tile) {
            for (int i0 = 0; i0 < 4; ++i0) {
              const int i_ch = i_blk * 4 + i0;
              const int o_base = (o_outer * 2 + o_tile) * 16;
              for (int o_gr1 = 0; o_gr1 < 2; ++o_gr1) {
                for (int o_gr2 = 0; o_gr2 < 2; ++o_gr2) {
                  for (int o_gr0 = 0; o_gr0 < 4; ++o_gr0) {
                    const int o_ch = o_base + (o_gr2 * 2 + o_gr1) * 4 + o_gr0;
                    if (o_ch < ohwi_shape.o && i_ch < ohwi_shape.i) {
                      ohwi_ref[counter++] =
                          ohwi_weights.data[ohwi_shape.LinearIndex(
                              {o_ch, ry, rx, i_ch})];
                    } else {
                      ohwi_ref[counter++] = 0.0f;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(ohwi_out, ohwi_ref);
}

}  // namespace
}  // namespace ml_drift
