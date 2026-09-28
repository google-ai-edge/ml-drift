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
#include "ml_drift/common/kernels/special/tests/experts_remap_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"

namespace ml_drift {

class BaseTest : public testing::Test {};

TEST_F(BaseTest, ExpertsRemapTest) { ABSL_ASSERT_OK(ExpertsRemapTest(exec_env)); }

TEST_F(BaseTest, ExpertsRemap1kTest) {
  ABSL_ASSERT_OK(ExpertsRemapBigTest(exec_env, /*seq_size=*/1024,
                                /*num_active_experts=*/8,
                                /*num_experts=*/128));
}

TEST_F(BaseTest, ExpertsRemap4kTest) {
  ABSL_ASSERT_OK(ExpertsRemapBigTest(exec_env, /*seq_size=*/4096,
                                /*num_active_experts=*/16,
                                /*num_experts=*/256));
}

TEST_F(BaseTest, OffsetsTest) { ABSL_ASSERT_OK(OffsetsTest(exec_env)); }
TEST_F(BaseTest, Offsets1kTest) { ABSL_ASSERT_OK(OffsetsBigTest(exec_env, 1024)); }
TEST_F(BaseTest, Offsets4kTest) { ABSL_ASSERT_OK(OffsetsBigTest(exec_env, 4096)); }

TEST_F(BaseTest, LinearizeMapTest) {
  ABSL_ASSERT_OK(LinearizeMapTest(exec_env, /*seq_size=*/16,
                             /*num_active_experts=*/4,
                             /*num_experts=*/8));
}

TEST_F(BaseTest, LinearizeMap1kTest) {
  ABSL_ASSERT_OK(LinearizeMapTest(exec_env, /*seq_size=*/1024,
                             /*num_active_experts=*/8,
                             /*num_experts=*/128));
}

TEST_F(BaseTest, LinearizeMap4kTest) {
  ABSL_ASSERT_OK(LinearizeMapTest(exec_env, /*seq_size=*/4096,
                             /*num_active_experts=*/16,
                             /*num_experts=*/256));
}

TEST_F(BaseTest, ExpertsRemapToTest) {
  ABSL_ASSERT_OK(ExpertsRemapToTest(exec_env, /*seq_size=*/16,
                               /*num_active_experts=*/4,
                               /*num_experts=*/8, /*channels=*/32));
}

TEST_F(BaseTest, ExpertsRemapTo1kTest) {
  ABSL_ASSERT_OK(ExpertsRemapToTest(exec_env, /*seq_size=*/1024,
                               /*num_active_experts=*/8,
                               /*num_experts=*/128, /*channels=*/16));
}

TEST_F(BaseTest, ExpertsRemapTo4kTest) {
  ABSL_ASSERT_OK(ExpertsRemapToTest(exec_env, /*seq_size=*/4096,
                               /*num_active_experts=*/16,
                               /*num_experts=*/256, /*channels=*/8));
}

TEST_F(BaseTest, ExpertsRemapFromTest) {
  ABSL_ASSERT_OK(ExpertsRemapFromTest(exec_env, /*seq_size=*/16,
                                 /*num_active_experts=*/4,
                                 /*num_experts=*/8, /*channels=*/32));
}

TEST_F(BaseTest, ExpertsRemapFrom1kTest) {
  ABSL_ASSERT_OK(ExpertsRemapFromTest(exec_env, /*seq_size=*/1024,
                                 /*num_active_experts=*/8,
                                 /*num_experts=*/128, /*channels=*/16));
}

TEST_F(BaseTest, ExpertsRemapFrom4kTest) {
  ABSL_ASSERT_OK(ExpertsRemapFromTest(exec_env, /*seq_size=*/4096,
                                 /*num_active_experts=*/16,
                                 /*num_experts=*/256, /*channels=*/8));
}

}  // namespace ml_drift
