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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONV_APPLE_MPP_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONV_APPLE_MPP_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ConvAppleMPPBigTest(TestExecutionEnvironment& env,
                                 TensorStorageType dst_storage,
                                 const BHWC& src_shape, int dst_channels);

absl::Status ConvAppleMPPExternalWeightsTest(TestExecutionEnvironment& env,
                                             TensorStorageType dst_storage,
                                             const BHWC& src_shape,
                                             int dst_channels);

absl::Status ConvAppleMPPExternalBatchedWi4Test(TestExecutionEnvironment& env,
                                                TensorStorageType storage,
                                                const BHWC& src_shape,
                                                int dst_channels);
absl::Status ConvAppleMPPExternalBatchedGroupedWi4Test(
    TestExecutionEnvironment& env, TensorStorageType dst_storage,
    const BHWC& src_shape, int dst_channels, int group_size);

absl::Status ConvAppleMPPBatchedMatMulTest(TestExecutionEnvironment& env,
                                           TensorStorageType dst_storage,
                                           const BHWC& left_shape,
                                           const BHWC& right_shape);

absl::Status ConvAppleMPPRuntimeSrcEndChannelsTest(
    TestExecutionEnvironment& env, TensorStorageType storage);
absl::Status ConvAppleMPPRuntimeDstEndChannelsTest(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status ConvAppleMPPInt8BigTest(TestExecutionEnvironment& env,
                                     TensorStorageType dst_storage,
                                     const BHWC& src_shape, int dst_channels);

absl::Status ConvAppleMPPInt8ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType dst_storage,
    const BHWC& src_shape, int dst_channels);

absl::Status ConvAppleMPPInt8ExternalBatchedWi4Test(
    TestExecutionEnvironment& env, TensorStorageType dst_storage,
    const BHWC& src_shape, int dst_channels);

absl::Status ConvAppleMPPInt8WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType int_storage,
    TensorStorageType float_storage, DataType float_type, const BHWC& src_shape,
    int dst_channels);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONV_APPLE_MPP_TEST_UTIL_H_
