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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_ADD_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_ADD_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

template <DataType data_type>
absl::Status AddTwoEqualIntTensorsTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage);

template <DataType data_type>
absl::Status AddTwoEqualUintTensorsTest(TestExecutionEnvironment& env,
                                        TensorStorageType storage);

absl::Status AddTwoEqualTensorsBFloatTest(TestExecutionEnvironment& env,
                                          TensorStorageType storage);

absl::Status AddTwoEqualTensorsTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage);

absl::Status AddFirstTensorHasMoreChannelsThanSecondTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);

absl::Status AddFirstTensorHasLessChannelsThanSecondTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);

absl::Status AddBigTest(TestExecutionEnvironment& env, DataType data_type,
                        TensorStorageType storage);
absl::Status AddBatchedBigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage);
absl::Status AddNotEqualBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status AddNotEqualBatchedBigTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage);
absl::Status AddNotEqualFirstTensorBigTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage);
absl::Status AddNotEqualFirstTensorBatchedBigTest(TestExecutionEnvironment& env,
                                                  DataType data_type,
                                                  TensorStorageType storage);

absl::Status AddBroadcast5DTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_ADD_TEST_UTIL_H_
