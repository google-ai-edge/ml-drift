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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_DEPTHWISE_CONV_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_DEPTHWISE_CONV_TEST_UTIL_H_

#include "ml_drift/common/precision.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status DepthwiseConvSimpleWeightsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage);
absl::Status DepthwiseConvNoMultiplierTest(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage);
absl::Status DepthwiseConvMultiplier2Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
absl::Status DepthwiseConvBigTest(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage);
absl::Status DepthwiseConvBatchedBigTest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage);
absl::Status DepthwiseConvExternalWeightsTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage);
absl::Status DepthwiseConv3DTest(TestExecutionEnvironment& env,
                                 CalculationsPrecision precision,
                                 TensorStorageType storage);
absl::Status DepthwiseConv3DBatchedTest(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage);
absl::Status DepthwiseConv3DNoMultiplierTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage);
// 3x3
absl::Status DepthwiseConv3x3SimpleWeightsTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage);
absl::Status DepthwiseConv3x3Test(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage);
absl::Status DepthwiseConv3x3BigTest(TestExecutionEnvironment& env,
                                     CalculationsPrecision precision,
                                     TensorStorageType storage);
absl::Status DepthwiseConv3x3BatchedBigTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage);
// tiled
absl::Status DepthwiseConvTiledSimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status DepthwiseConvTiledTest(TestExecutionEnvironment& env,
                                    CalculationsPrecision precision,
                                    TensorStorageType storage);
absl::Status DepthwiseConvTiledBigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);
absl::Status DepthwiseConvTiledBatchedBigTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage);
absl::Status DepthwiseConvTiledWithDilationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_DEPTHWISE_CONV_TEST_UTIL_H_
