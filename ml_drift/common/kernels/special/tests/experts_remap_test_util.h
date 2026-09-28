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

#ifndef ML_DRIFT_COMMON_KERNELS_SPECIAL_TESTS_EXPERTS_REMAP_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_SPECIAL_TESTS_EXPERTS_REMAP_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ExpertsRemapTest(TestExecutionEnvironment* env);
absl::Status ExpertsRemapBigTest(TestExecutionEnvironment* env, int seq_size,
                                 int num_active_experts, int num_experts);

absl::Status OffsetsTest(TestExecutionEnvironment* env);
absl::Status OffsetsBigTest(TestExecutionEnvironment* env, int size);

absl::Status LinearizeMapTest(TestExecutionEnvironment* env, int seq_size,
                              int num_active_experts, int num_experts);

absl::Status ExpertsRemapToTest(TestExecutionEnvironment* env, int seq_size,
                                int num_active_experts, int num_experts,
                                int channels);

absl::Status ExpertsRemapFromTest(TestExecutionEnvironment* env, int seq_size,
                                  int num_active_experts, int num_experts,
                                  int channels);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPECIAL_TESTS_EXPERTS_REMAP_TEST_UTIL_H_
