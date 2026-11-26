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

#include "ml_drift/common/kernels/fully_connected_util.h"

#include "gtest/gtest.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace fc {
namespace {

TEST(IsQuantizedTest, FloatTypesAreNotQuantized) {
  EXPECT_FALSE(IsQuantized(DataType::kFloat32));
  EXPECT_FALSE(IsQuantized(DataType::kFloat16));
}

TEST(IsQuantizedTest, IntegerTypesAreQuantized) {
  EXPECT_TRUE(IsQuantized(DataType::kInt8));
  EXPECT_TRUE(IsQuantized(DataType::kUint8));
  EXPECT_TRUE(IsQuantized(DataType::kInt4));
  EXPECT_TRUE(IsQuantized(DataType::kUint4));
  EXPECT_TRUE(IsQuantized(DataType::kInt2));
}

TEST(IsScalarQuantizedTest, ScalarQuantized) {
  // scale_zp_shape with product == 1
  EXPECT_TRUE(IsScalarQuantized(DataType::kInt8, OHWI(1, 1, 1, 1)));
}

TEST(IsScalarQuantizedTest, NotScalarQuantizedIfFloat) {
  EXPECT_FALSE(IsScalarQuantized(DataType::kFloat32, OHWI(1, 1, 1, 1)));
}

TEST(IsScalarQuantizedTest, NotScalarQuantizedIfMultipleElements) {
  EXPECT_FALSE(IsScalarQuantized(DataType::kInt8, OHWI(4, 1, 1, 1)));
}

TEST(IsLinearQuantizedTest, LinearQuantized) {
  // o != 1 and i == 1
  EXPECT_TRUE(IsLinearQuantized(DataType::kInt8, OHWI(4, 1, 1, 1)));
}

TEST(IsLinearQuantizedTest, NotLinearIfFloat) {
  EXPECT_FALSE(IsLinearQuantized(DataType::kFloat32, OHWI(4, 1, 1, 1)));
}

TEST(IsLinearQuantizedTest, NotLinearIfINotOne) {
  EXPECT_FALSE(IsLinearQuantized(DataType::kInt8, OHWI(4, 1, 1, 2)));
}

TEST(IsBlockQuantizedTest, BlockQuantized) {
  // o != 1 and i != 1
  EXPECT_TRUE(IsBlockQuantized(DataType::kInt8, OHWI(4, 1, 1, 2)));
}

TEST(IsBlockQuantizedTest, NotBlockIfFloat) {
  EXPECT_FALSE(IsBlockQuantized(DataType::kFloat32, OHWI(4, 1, 1, 2)));
}

TEST(IsBlockQuantizedTest, NotBlockIfIIsOne) {
  EXPECT_FALSE(IsBlockQuantized(DataType::kInt8, OHWI(4, 1, 1, 1)));
}

TEST(GetDataTypeForWeightsTest, UnsignedToSigned) {
  EXPECT_EQ(GetDataTypeForWeights(DataType::kUint8), DataType::kInt8);
  EXPECT_EQ(GetDataTypeForWeights(DataType::kUint4), DataType::kInt4);
  EXPECT_EQ(GetDataTypeForWeights(DataType::kUint2), DataType::kInt2);
}

TEST(GetDataTypeForWeightsTest, OtherTypesPassThrough) {
  EXPECT_EQ(GetDataTypeForWeights(DataType::kInt8), DataType::kInt8);
  EXPECT_EQ(GetDataTypeForWeights(DataType::kFloat32), DataType::kFloat32);
  EXPECT_EQ(GetDataTypeForWeights(DataType::kFloat16), DataType::kFloat16);
}

TEST(GetBlockSpatialCoordsTest, BasicDecomposition) {
  // shape = BHWC(2, 3, 4, 1)
  // linear_spatial decomposes as: b = linear % B, then x = (linear/B) % W,
  // then y = (linear/B/W) % H
  BHWC shape(2, 3, 4, 1);
  int3 coords = GetBlockSpatialCoords(0, shape);
  EXPECT_EQ(coords.x, 0);  // b
  EXPECT_EQ(coords.y, 0);  // y
  EXPECT_EQ(coords.z, 0);  // x
}

TEST(GetBlockSpatialCoordsTest, NonZeroCoords) {
  BHWC shape(2, 3, 4, 1);
  // linear = 5: b = 5%2=1, rem=2, x=2%4=2, rem=0, y=0%3=0
  int3 coords = GetBlockSpatialCoords(5, shape);
  EXPECT_EQ(coords.x, 1);  // b
  EXPECT_EQ(coords.y, 0);  // y
  EXPECT_EQ(coords.z, 2);  // x
}

TEST(GetBlockSizeTest, NullShapeReturnsOnes) {
  BHWC result = GetBlockSize(nullptr, false);
  EXPECT_EQ(result.b, 1);
  EXPECT_EQ(result.h, 1);
  EXPECT_EQ(result.w, 1);
  EXPECT_EQ(result.c, 1);
}

TEST(GetBlockSizeTest, BasicShape) {
  BHWC shape(2, 3, 4, 8);
  BHWC result = GetBlockSize(&shape, false);
  // c is always set to 1
  EXPECT_EQ(result.c, 1);
  EXPECT_EQ(result.b, 2);
  EXPECT_EQ(result.h, 3);
  EXPECT_EQ(result.w, 4);
}

TEST(GetBlockSizeTest, BatchedWeightsSetsHToOne) {
  BHWC shape(2, 3, 4, 8);
  BHWC result = GetBlockSize(&shape, true);
  EXPECT_EQ(result.c, 1);
  EXPECT_EQ(result.h, 1);
  EXPECT_EQ(result.w, 4);
}

}  // namespace
}  // namespace fc
}  // namespace ml_drift
