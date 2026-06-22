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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONV_GENERIC_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONV_GENERIC_TEST_UTIL_H_

#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ConvGeneric1x1SimpleWeightsTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage);
absl::Status ConvGeneric1x1Test(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage);
absl::Status ConvGenericSimpleWeightsTest(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
absl::Status ConvGenericTest(TestExecutionEnvironment& env,
                             CalculationsPrecision precision,
                             TensorStorageType storage);
absl::Status ConvGenericGroupedTest(TestExecutionEnvironment& env,
                                    CalculationsPrecision precision,
                                    TensorStorageType storage);
absl::Status ConvGenericWinograd3x3TileNxNTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage,
                                               int tile_size);
absl::Status ConvGeneric1x1Int8SymmetricTest(TestExecutionEnvironment& env,
                                             TensorStorageType src_storage,
                                             TensorStorageType dst_storage);

// Big tests:
// Big test vs ref implementation.
absl::Status ConvGeneric1x1BigTest(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage);
// Big test vs ref implementation.
absl::Status ConvGeneric1x1BatchedBigTest(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
// Big test vs ref implementation.
absl::Status ConvGenericBigTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage);
// Big test vs ref implementation.
absl::Status ConvGenericBatchedBigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);
// Big test vs ref implementation.
absl::Status ConvGenericGroupedBigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);

absl::Status ConvGenericExternalWfloatTest(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage,
                                           const BHWC& src_shape,
                                           int dst_channels,
                                           bool batched_weights = false);

absl::Status ConvGenericExternalWi8Test(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage,
                                        const BHWC& src_shape, int dst_channels,
                                        bool batched_weights = false,
                                        int group_size = -1);

absl::Status ConvGenericExternalWi4Test(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage,
                                        const BHWC& src_shape, int dst_channels,
                                        bool batched_weights = false,
                                        int group_size = -1);

absl::Status ConvGenericExternalWi2Test(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage,
                                        const BHWC& src_shape, int dst_channels,
                                        bool batched_weights = false,
                                        int group_size = -1);

absl::Status ConvGeneric3d1x1x1BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);
absl::Status ConvGeneric3d1x1x1BatchedBigTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage);
absl::Status ConvGeneric3dBigTest(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage);
absl::Status ConvGeneric3dBatchedBigTest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage);

// Big test vs ref implementation.
absl::Status ConvGenericInt8BigTest(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage);
// Big test vs ref implementation.
absl::Status ConvGenericInt8ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage);
// Big test vs ref implementation.
absl::Status ConvGenericInt8WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage);
// Big test vs ref implementation. Converts weights to int8 before using conv.
absl::Status ConvGenericInt8WeightsInt4WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage);

// Big test vs ref implementation.
absl::Status ConvGenericInt4BigTest(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage,
                                    const BHWC& src_shape);
// Big test vs ref implementation.
absl::Status ConvGenericInt4ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage, const BHWC& src_shape);
// Big test vs ref implementation.
absl::Status ConvGenericInt4WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage, const BHWC& src_shape);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONV_GENERIC_TEST_UTIL_H_
