// Copyright 2024 The ML Drift Authors.
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
#include "ml_drift/common/default/status_matchers.h"
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(4, 1, 1, 4);
  weights_i8.data = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  std::vector<uint8_t> weights_i8_packed(weights_i8.data.size() / 2);
  for (int i = 0; i < weights_i8.data.size() / 2; ++i) {
    uint8_t part0 = Int4AsUint4(weights_i8.data[i * 2]);
    uint8_t part1 = Int4AsUint4(weights_i8.data[i * 2 + 1]);
    weights_i8_packed[i] = (part1 << 4) | part0;
  }

  // Prep weights
  Tensor<OHWI, DataType::UINT8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::INT8> weights_i8;
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
  Tensor<OHWI, DataType::UINT8> weights;
  weights.shape = OHWI(16, 2, 1, 8);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 2;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape));
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  uint8_t pad = 8u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  uint8_t pad = 8u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(8);
  uint8_t pad = 8u;
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(8);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt4Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  weights_sum_i[0] = 1;  // This will be zeroed out
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(4, 1, 1, 4);
  weights_i8.data = {0, 1, -2, -1, 1, -2, -1, 0, -2, -1, 0, 1, -1, 0, 1, -2};

  std::vector<uint8_t> weights_i8_unpacked;
  for (auto d : weights_i8.data) {
    weights_i8_unpacked.push_back(Int2AsUint2(d));
  }
  std::vector<uint8_t> weights_i8_packed =
      PackInt2IntoUint8(weights_i8_unpacked);

  // Prep weights
  Tensor<OHWI, DataType::UINT8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::INT8> weights_i8;
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
  Tensor<OHWI, DataType::UINT8> weights;
  weights.shape = OHWI(16, 2, 1, 8);
  weights.data = weights_i8_packed;

  // Prep WeightsDescription
  WeightsDescription dst_weight_desc;
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 2;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape));
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  uint8_t pad = 2u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  uint8_t pad = 2u;
  std::vector<int32_t> weights_sum_i(AlignByN(weights.shape.o, 4));
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;

  std::vector<uint8_t> output(4);
  uint8_t pad = 2u;
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  Tensor<OHWI, DataType::UINT8> weights;
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
  dst_weight_desc.type = DataType::UINT8;
  dst_weight_desc.output_group_size = 1;
  dst_weight_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  std::vector<uint8_t> output(4);
  std::vector<int32_t> weights_sum_i(weights.shape.o);
  MLD_ASSERT_OK(RearrangeWeightsUInt2Packed(weights, dst_weight_desc,
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
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {{Axis::INPUT_CHANNELS, 4},
                             {Axis::OUTPUT_CHANNELS, 4},
                             {Axis::INPUT_CHANNELS, 0},
                             {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, weights_shape),
            AlignByN(weights_shape.o, 4) * AlignByN(weights_shape.i, 4));
}

TEST(CustomGroups, TotalSizeIOI4O3I2) {
  const OHWI weights_shape(17, 1, 1, 31);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {{Axis::INPUT_CHANNELS, 2},
                             {Axis::OUTPUT_CHANNELS, 3},
                             {Axis::INPUT_CHANNELS, 4},
                             {Axis::OUTPUT_CHANNELS, 0},
                             {Axis::INPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;
  EXPECT_EQ(GetTotalElementsCountForLayout(weight_desc, weights_shape),
            AlignByN(weights_shape.o, 3) * AlignByN(weights_shape.i, 8));
}

TEST(CustomGroups, TotalSizeIOI4SpatialO2) {
  const OHWI weights_shape(17, 2, 3, 31);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 2}, {Axis::WIDTH, 0},
      {Axis::HEIGHT, 0},          {Axis::INPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 0}, {Axis::INPUT_CHANNELS, 0}};
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

  Tensor<OHWI, DataType::FLOAT32> weights;
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

  Tensor<OHWDI, DataType::FLOAT32> weights;
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
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 0},
      {Axis::WIDTH, 0},           {Axis::HEIGHT, 0},
      {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWDIToOSpatialIOGroupI4O4) {
  const OHWDI weights_shape(39, 2, 3, 4, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 0},
      {Axis::WIDTH, 0},           {Axis::HEIGHT, 0},
      {Axis::DEPTH, 0},           {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupI4O4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWIToOSpatialIOGroupO4I4) {
  const OHWI weights_shape(39, 2, 3, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::INPUT_CHANNELS, 4},  {Axis::OUTPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 0},
      {Axis::WIDTH, 0},           {Axis::HEIGHT, 0},
      {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupO4I4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWDIToOSpatialIOGroupO4I4) {
  const OHWDI weights_shape(39, 2, 3, 4, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::INPUT_CHANNELS, 4},  {Axis::OUTPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 0},
      {Axis::WIDTH, 0},           {Axis::HEIGHT, 0},
      {Axis::DEPTH, 0},           {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOSpatialIOGroupO4I4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWIToOISpatialOGroupI4O4) {
  const OHWI weights_shape(39, 2, 3, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::WIDTH, 0},
      {Axis::HEIGHT, 0},          {Axis::INPUT_CHANNELS, 0},
      {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOISpatialOGroupI4O4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWDIToOISpatialOGroupI4O4) {
  const OHWDI weights_shape(39, 2, 3, 4, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::WIDTH, 0},
      {Axis::HEIGHT, 0},          {Axis::DEPTH, 0},
      {Axis::INPUT_CHANNELS, 0},  {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOISpatialOGroupI4O4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWIToOISpatialOGroupO4I4) {
  const OHWI weights_shape(39, 2, 3, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::INPUT_CHANNELS, 4},  {Axis::OUTPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::WIDTH, 0},
      {Axis::HEIGHT, 0},          {Axis::INPUT_CHANNELS, 0},
      {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOISpatialOGroupO4I4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

TEST(CustomGroups, ConversionOHWDIToOISpatialOGroupO4I4) {
  const OHWDI weights_shape(39, 2, 3, 4, 23);
  WeightsDescription weight_desc;
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::INPUT_CHANNELS, 4},  {Axis::OUTPUT_CHANNELS, 4},
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::WIDTH, 0},
      {Axis::HEIGHT, 0},          {Axis::DEPTH, 0},
      {Axis::INPUT_CHANNELS, 0},  {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  WeightsDescription weight_desc_ref;
  weight_desc_ref.type = DataType::FLOAT32;
  weight_desc_ref.output_group_size = 4;
  weight_desc_ref.layout = WeightsLayout::kOISpatialOGroupO4I4;

  RunConversions(weights_shape, weight_desc, weight_desc_ref);
}

void RearrangeWeightsToOSpatialIOGroupITileOTile(
    const Tensor<OHWI, DataType::FLOAT32>& weights, int i_tile_size,
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
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 8}, {Axis::INPUT_CHANNELS, 8},
      {Axis::OUTPUT_CHANNELS, 2}, {Axis::INPUT_CHANNELS, 0},
      {Axis::WIDTH, 0},           {Axis::HEIGHT, 0},
      {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;

  const int total_elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_shape);

  Tensor<OHWI, DataType::FLOAT32> weights;
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
  RearrangeWeightsToOSpatialIOGroupITileOTile(
      weights, 8, 8, 2, absl::MakeSpan(output_ref.data(), output_ref.size()),
      0.0f);

  EXPECT_EQ(output, output_ref);
}

void RearrangeWeightsToOICustomSpatialI4O4(
    const Tensor<OHWI, DataType::FLOAT32>& weights,
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
  weight_desc.type = DataType::FLOAT32;
  weight_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 4},
      {Axis::WIDTH, 0},           {Axis::HEIGHT, 0},
      {Axis::INPUT_CHANNELS, 0},  {Axis::OUTPUT_CHANNELS, 0}};
  weight_desc.layout = WeightsLayout::kCustomGroups;
  weight_desc.spatial_remap = {4, 5, 3, 7, 1, 8, 6, 2, 0};

  const int total_elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_shape);

  Tensor<OHWI, DataType::FLOAT32> weights;
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
  RearrangeWeightsToOICustomSpatialI4O4(
      weights, weight_desc.spatial_remap,
      absl::MakeSpan(output_ref.data(), output_ref.size()));

  EXPECT_EQ(output, output_ref);
}

}  // namespace
}  // namespace ml_drift
