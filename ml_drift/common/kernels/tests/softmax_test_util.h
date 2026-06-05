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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_SOFTMAX_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_SOFTMAX_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status SoftmaxTest(TestExecutionEnvironment& env, DataType data_type,
                         TensorStorageType storage);
absl::Status SoftmaxWGTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage);
absl::Status SoftmaxReduceTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage);
absl::Status SoftmaxBigNumberTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage);
absl::Status SoftmaxRuntimeChannelsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);
absl::Status SoftmaxBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage);
absl::Status SoftmaxBatchedBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status SoftmaxReduceBigTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage);
absl::Status SoftmaxReduceRuntimeChannelsBigTest(TestExecutionEnvironment& env,
                                                 DataType data_type,
                                                 TensorStorageType storage);
absl::Status SoftmaxRuntimeChannelsBigTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage);

absl::Status Softmax1x1Test(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage);
absl::Status Softmax1x1BigNumberTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage);
absl::Status Softmax1x1RuntimeChannelsTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage);
absl::Status Softmax1x1Custom1Test(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status Softmax1x1BigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage);
absl::Status Softmax1x1BatchedBigTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage);
absl::Status Softmax1x1ReduceBigTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage);
absl::Status Softmax1x1ReduceRuntimeChannelsBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status Softmax1x1RuntimeChannelsBigTest(TestExecutionEnvironment& env,
                                              DataType data_type,
                                              TensorStorageType storage);

absl::Status Softmax5DTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage);

absl::Status Softmax1x15DTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_SOFTMAX_TEST_UTIL_H_
