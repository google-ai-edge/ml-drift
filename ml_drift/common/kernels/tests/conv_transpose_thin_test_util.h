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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONV_TRANSPOSE_THIN_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONV_TRANSPOSE_THIN_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ConvolutionTransposedThinSimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposedThinTest(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage);
absl::Status ConvolutionTransposedThinBigTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage);
absl::Status ConvolutionTransposedThinBatchedBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

absl::Status ConvolutionTransposed3x3ThinSimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed3x3ThinTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage);
absl::Status ConvolutionTransposed3x3ThinBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed3x3ThinBatchedBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed3x3ThinExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONV_TRANSPOSE_THIN_TEST_UTIL_H_
