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

#include "ml_drift/common/util.h"

#include <cfloat>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"

namespace ml_drift {
namespace {

using ::testing::Eq;
using ::testing::FloatEq;

TEST(UtilTest, DivideRoundUp) {
  EXPECT_THAT(DivideRoundUp(0, 256), Eq(0));
  EXPECT_THAT(DivideRoundUp(2u, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(2, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(255u, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(255, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(256u, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(256, 256), Eq(1));
  EXPECT_THAT(DivideRoundUp(257u, 256), Eq(2));
  EXPECT_THAT(DivideRoundUp(257, 256), Eq(2));
}

TEST(UtilTest, AlignByN) {
  EXPECT_THAT(AlignByN(0u, 256), Eq(0));
  EXPECT_THAT(AlignByN(1u, 256), Eq(256));
  EXPECT_THAT(AlignByN(255u, 256), Eq(256));
  EXPECT_THAT(AlignByN(256u, 256), Eq(256));
  EXPECT_THAT(AlignByN(257u, 256), Eq(512));

  EXPECT_THAT(AlignByN(1, 4), Eq(4));
  EXPECT_THAT(AlignByN(80, 4), Eq(80));
  EXPECT_THAT(AlignByN(81, 4), Eq(84));
}

TEST(UtilTest, GetEpsilon) {
  constexpr float kHalfEpsilon = 9.7656e-4;
  GpuInfo gpu_info;
  gpu_info.opencl_info.supports_fp16_rtn = false;
  gpu_info.opencl_info.supports_fp32_rtn = false;
  EXPECT_THAT(GetEpsilon(CalculationsPrecision::F32, gpu_info),
              FloatEq(FLT_EPSILON * 4.0f));
  EXPECT_THAT(GetEpsilon(CalculationsPrecision::F32_F16, gpu_info),
              FloatEq(kHalfEpsilon * 4.0f));
  EXPECT_THAT(GetEpsilon(CalculationsPrecision::F16, gpu_info),
              FloatEq(kHalfEpsilon * 4.0f));
  gpu_info.opencl_info.supports_fp16_rtn = true;
  gpu_info.opencl_info.supports_fp32_rtn = true;
  EXPECT_THAT(GetEpsilon(CalculationsPrecision::F32, gpu_info),
              FloatEq(FLT_EPSILON));
  EXPECT_THAT(GetEpsilon(CalculationsPrecision::F32_F16, gpu_info),
              FloatEq(kHalfEpsilon));
  EXPECT_THAT(GetEpsilon(CalculationsPrecision::F16, gpu_info),
              FloatEq(kHalfEpsilon));
}

}  // namespace
}  // namespace ml_drift
