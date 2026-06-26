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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WAVE_MATRIX_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WAVE_MATRIX_TEST_UTIL_H_

#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ConvWaveMatrix1x1Test(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage);
absl::Status ConvWaveMatrixTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage);
absl::Status ConvWaveMatrix1x1BatchTest(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage);
absl::Status ConvWaveMatrix1x1ExternalWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvWaveMatrixExternalWeightsTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage);
absl::Status ConvWaveMatrixExternalBatchedWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

absl::Status ConvWaveMatrixPackedGroupsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage,
                                            const BHWC& src_shape,
                                            int dst_channels);

absl::Status ConvWaveMatrixExternalWfloatTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage,
                                              const BHWC& src_shape,
                                              int dst_channels,
                                              bool batched_weights = false);
absl::Status ConvWaveMatrixExternalWi8Test(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape, int dst_channels,
    bool batched_weights = false, int group_size = -1, int scale_zp_batch = -1);
absl::Status ConvWaveMatrixExternalWi4Test(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape, int dst_channels,
    bool batched_weights = false, int group_size = -1);
absl::Status ConvWaveMatrixExternalWi2Test(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape, int dst_channels,
    bool batched_weights = false, int group_size = -1);

absl::Status ConvWaveMatrixWinograd4x4To6x6Test(TestExecutionEnvironment& env,
                                                CalculationsPrecision precision,
                                                TensorStorageType storage);

absl::Status ConvWaveMatrixRuntimeSrcEndChannelsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvWaveMatrixRuntimeDstEndChannelsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

absl::Status ConvWaveMatrixInt8Test(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage);
absl::Status ConvWaveMatrixInt8ExternalWeightsTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage);
absl::Status ConvWaveMatrixInt8WithSrcQuantizationTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WAVE_MATRIX_TEST_UTIL_H_
