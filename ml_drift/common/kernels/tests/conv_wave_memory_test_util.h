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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WAVE_MEMORY_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WAVE_MEMORY_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {
absl::Status ConvWaveMemoryTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage, const BHWC& shape,
                                int dst_channels);

absl::Status ConvWaveMemoryGroupedX3Test(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage);
absl::Status ConvWaveMemoryGroupedX7Test(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage);

absl::Status ConvWaveMemoryExternalWeightsTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage);
absl::Status ConvWaveMemoryExternalBatchedWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

absl::Status ConvWaveMemoryWinograd4x4To6x6Test(TestExecutionEnvironment& env,
                                                CalculationsPrecision precision,
                                                TensorStorageType storage);

absl::Status ConvWaveMemoryBatchedMatMulTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage);

absl::Status ConvWaveMemoryPackedGroupsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage,
                                            const BHWC& src_shape,
                                            int dst_channels);

absl::Status ConvWaveMemoryInt8Test(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage);
absl::Status ConvWaveMemoryInt8ExternalWeightsTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage);
absl::Status ConvWaveMemoryInt8WithSrcQuantizationTest(
    TestExecutionEnvironment& env, TensorStorageType int_storage,
    TensorStorageType float_storage, DataType float_type);

absl::Status ConvWaveMemoryRuntimeSrcEndChannelsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
absl::Status ConvWaveMemoryRuntimeDstEndChannelsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);
}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WAVE_MEMORY_TEST_UTIL_H_
