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

#include "ml_drift/samples/stable_diffusion/util.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"

using ::testing::FloatEq;
using ::testing::Pointwise;

namespace ml_drift {
namespace {

std::vector<half> ConvertFp32ToFp16(const std::vector<float>& fp32s) {
  std::vector<half> fp16s;
  fp16s.reserve(fp32s.size());
  for (const float fp32 : fp32s) fp16s.emplace_back(fp32);
  return fp16s;
}

TEST(UtilTest, ConvertsLinearFp16ToFp32Empty) {  // num_elements == 0
  const std::vector<float> kNumbers = {};
  const std::vector<half> input = ConvertFp32ToFp16(kNumbers);
  EXPECT_EQ(internal::ConvertLinearFp16ToFp32(input), kNumbers);
}

TEST(UtilTest, ConvertsLinearFp16ToFp32WithoutNeon) {  // num_elements < 8
  const std::vector<float> kNumbers = {0, 1, 2, 3};
  const std::vector<half> input = ConvertFp32ToFp16(kNumbers);
  EXPECT_EQ(internal::ConvertLinearFp16ToFp32(input), kNumbers);
}

TEST(UtilTest, ConvertsLinearFp16ToFp32WithNeon) {  // num_elements == 8
  const std::vector<float> kNumbers = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<half> input = ConvertFp32ToFp16(kNumbers);
  EXPECT_EQ(internal::ConvertLinearFp16ToFp32(input), kNumbers);
}

TEST(UtilTest, ConvertsLinearFp16ToFp32General) {  // num_elements > 8 & % 8 > 0
  const std::vector<float> kNumbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const std::vector<half> input = ConvertFp32ToFp16(kNumbers);
  EXPECT_EQ(internal::ConvertLinearFp16ToFp32(input), kNumbers);
}

TEST(UtilTest, ConvertsOihwFp16ToOhwiFp32WithoutNeon) {  // hw_stride < 8 rows
  const auto input = ConvertFp32ToFp16({
      0,  1,  2,   // o = 0, i = 0, h = 0, w = 0..2
      3,  4,  5,   // o = 0, i = 0, h = 1, w = 0..2
      6,  7,  8,   // o = 0, i = 1, h = 0, w = 0..2
      9,  10, 11,  // o = 0, i = 1, h = 1, w = 0..2
      12, 13, 14,  // o = 0, i = 2, h = 0, w = 0..2
      15, 16, 17,  // o = 0, i = 2, h = 1, w = 0..2
      18, 19, 20,  // o = 0, i = 3, h = 0, w = 0..2
      21, 22, 23,  // o = 0, i = 3, h = 1, w = 0..2
  });
  const OHWI kShape(1, 2, 3, 4);
  const auto actual = internal::ConvertOihwFp16ToOhwiFp32(input, kShape);

  const std::vector<float> kExpected = {
      0, 6,  12, 18,  // o = 0, h = 0, w = 0, i = 0..3
      1, 7,  13, 19,  // o = 0, h = 0, w = 1, i = 0..3
      2, 8,  14, 20,  // o = 0, h = 0, w = 2, i = 0..3
      3, 9,  15, 21,  // o = 0, h = 1, w = 0, i = 0..3
      4, 10, 16, 22,  // o = 0, h = 1, w = 1, i = 0..3
      5, 11, 17, 23,  // o = 0, h = 1, w = 2, i = 0..3
  };
  EXPECT_EQ(actual, kExpected);
}

TEST(UtilTest, ConvertsOihwFp16ToOhwiFp32General) {  // hw_stride > 8 rows
  const auto input = ConvertFp32ToFp16({
      0,  1,  2,  3,   // o = 0, i = 0, h = 0, w = 0..3
      4,  5,  6,  7,   // o = 0, i = 0, h = 1, w = 0..3
      8,  9,  10, 11,  // o = 0, i = 0, h = 2, w = 0..3
      12, 13, 14, 15,  // o = 0, i = 1, h = 0, w = 0..3
      16, 17, 18, 19,  // o = 0, i = 1, h = 1, w = 0..3
      20, 21, 22, 23,  // o = 0, i = 1, h = 2, w = 0..3
  });
  const OHWI kShape(1, 3, 4, 2);
  const auto actual = internal::ConvertOihwFp16ToOhwiFp32(input, kShape);

  const std::vector<float> kExpected = {
      0,  12, 1,  13,  // o = 0, h = 0, w = 0..1, i = 0..1
      2,  14, 3,  15,  // o = 0, h = 0, w = 2..3, i = 0..1
      4,  16, 5,  17,  // o = 0, h = 1, w = 0..1, i = 0..1
      6,  18, 7,  19,  // o = 0, h = 1, w = 2..3, i = 0..1
      8,  20, 9,  21,  // o = 0, h = 2, w = 0..1, i = 0..1
      10, 22, 11, 23,  // o = 0, h = 2, w = 2..3, i = 0..1
  };
  EXPECT_EQ(actual, kExpected);
}

TEST(UtilTest, B345188985) {
  std::vector<float> fp32s;
  fp32s.reserve(320 * 3 * 3 * 17);
  for (int i = 0; i < 320 * 3 * 3 * 17; ++i) fp32s.push_back(i % 100);
  const std::vector<half> fp16s = ConvertFp32ToFp16(fp32s);
  const OHWI kShape(320, 3, 3, 17);
  const std::vector<float> actual =
      internal::ConvertOihwFp16ToOhwiFp32(fp16s, kShape);
  ASSERT_EQ(actual.size(), 320 * 3 * 3 * 17);
  std::vector<float> expected;
  expected.reserve(320 * 3 * 3 * 17);
  for (int o = 0; o < 320; ++o) {
    for (int h = 0; h < 3; ++h) {
      for (int w = 0; w < 3; ++w) {
        for (int i = 0; i < 17; ++i) {
          expected.push_back((153 * o + 9 * i + 3 * h + w) % 100);
        }
      }
    }
  }
  EXPECT_THAT(actual, Pointwise(FloatEq(), expected));
}

}  // namespace
}  // namespace ml_drift
