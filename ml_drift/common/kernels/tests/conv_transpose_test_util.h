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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONV_TRANSPOSE_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONV_TRANSPOSE_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

// Base tests
absl::Status ConvTransposedSimpleWeightsTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage);
absl::Status ConvTransposedTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage);
absl::Status ConvTransposedBigTest(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage);
absl::Status ConvTransposedBatchedBigTest(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
absl::Status ConvTransposed3DBigTest(TestExecutionEnvironment& env,
                                     CalculationsPrecision precision,
                                     TensorStorageType storage);
absl::Status ConvTransposed3DBatchedBigTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage);
absl::Status ConvTransposedExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

// 2x2 tests
absl::Status ConvolutionTransposed2x2Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
absl::Status ConvolutionTransposed2x2BatchedTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed2x2DynamicWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

// 3x3 tests
absl::Status ConvolutionTransposed3x3SimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed3x3Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
absl::Status ConvolutionTransposed3x3BatchedTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed3x3ExternalWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

// 4x4 tests
absl::Status ConvolutionTransposed4x4SimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed4x4Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
absl::Status ConvolutionTransposed4x4BatchedTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvolutionTransposed4x4ExternalWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONV_TRANSPOSE_TEST_UTIL_H_
