// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_PADDING_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_PADDING_TEST_UTIL_H_

#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status PaddingAppendWidthTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage);
absl::Status PaddingAppendWidthConstValuesTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage);
absl::Status PaddingPrependWidthTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage);
absl::Status PaddingAppendHeightTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage);
absl::Status PaddingPrependHeightTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage);
absl::Status PaddingAppendChannelsTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage);
absl::Status PaddingPrependChannelsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);
absl::Status PaddingPrependChannelsX4Test(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage);
absl::Status PaddingComplexTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status PaddingReflectWidthTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage);
absl::Status PaddingReflectChannelsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);
absl::Status PaddingBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage);
absl::Status PaddingBatchedBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_PADDING_TEST_UTIL_H_
