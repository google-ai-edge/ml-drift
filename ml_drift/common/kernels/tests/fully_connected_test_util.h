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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_FULLY_CONNECTED_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_FULLY_CONNECTED_TEST_UTIL_H_

#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status FullyConnectedTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage);
absl::Status FullyConnectedLargeTest(TestExecutionEnvironment& env,
                                     CalculationsPrecision precision,
                                     TensorStorageType storage);
absl::Status FullyConnectedExtraLargeTest(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);
absl::Status FullyConnectedInt8Test(TestExecutionEnvironment& env,
                                    CalculationsPrecision precision,
                                    TensorStorageType storage);
absl::Status FullyConnectedInt8BlockwiseAttributesTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt8BlockwiseAttributesWithZeroPointsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedWeightsAsSpatialTensorTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const OHWI& weights_shape,
    const BHWC& src_shape);

absl::Status FullyConnectedInt4Sparse2x4Test(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage,
                                             const BHWC& src_shape,
                                             int dst_channels);

absl::Status FullyConnectedBigTest(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage);
absl::Status FullyConnectedWidth3Height2BigTest(TestExecutionEnvironment& env,
                                                CalculationsPrecision precision,
                                                TensorStorageType storage);
absl::Status FullyConnectedExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedBatchedWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedRingedOTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);
absl::Status FullyConnectedRingedITest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);

absl::Status FullyConnectedInt8BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);
absl::Status FullyConnectedInt8Width2Batch2BigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt8GroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt8ExternalBigTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage);
absl::Status FullyConnectedInt8BatchedWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt8BatchedWeightsIdsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape,
    const BHWC& batch_ids_shape);
absl::Status FullyConnectedInt8ExternalGroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedSi8Wi8BigTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage);

absl::Status FullyConnectedInt4BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);
absl::Status FullyConnectedInt4BlockwiseTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage);
absl::Status FullyConnectedInt4WidthIs4BigTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage);
absl::Status FullyConnectedInt4GroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt4ExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt4ExternalGroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedSi8Wi4BigTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage);

absl::Status FullyConnectedInt2BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage);
absl::Status FullyConnectedInt2BlockwiseTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage);
absl::Status FullyConnectedInt2WidthIs4BigTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage);
absl::Status FullyConnectedInt2GroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt2ExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedInt2ExternalGroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status FullyConnectedSi8Wi2BigTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage);

absl::Status FullyConnectedPackedGroupsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_FULLY_CONNECTED_TEST_UTIL_H_
