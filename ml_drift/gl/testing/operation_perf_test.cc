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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/gl/testing/perf_util.h"

namespace ml_drift {
namespace gl {

TEST(OpenGlOperationTest, Performance) {
  ABSL_ASSERT_OK(ConvolutionPerfTest(CalculationsPrecision::F16,
                                BHWC(1, 32, 32, 1024), 1024, HW(1, 1)));
  // ABSL_ASSERT_OK(DepthwiseConvPerfTest(BHWC(1, 16, 16, 768), HW(5, 5)));
}

}  // namespace gl
}  // namespace ml_drift
