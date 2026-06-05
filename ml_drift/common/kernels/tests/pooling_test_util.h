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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_POOLING_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_POOLING_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status AveragePoolingTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status AveragePoolingNonEmptyPaddingTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage);
absl::Status MaxPoolingTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage);
absl::Status MaxPoolingIndicesTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);

absl::Status AveragePoolingBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status AveragePoolingBatchedBigTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage);
absl::Status AveragePoolingNonEmptyPaddingBigTest(TestExecutionEnvironment& env,
                                                  DataType data_type,
                                                  TensorStorageType storage);
absl::Status AveragePooling3DBigTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage);
absl::Status AveragePooling3DBatchedBigTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage);
absl::Status MaxPoolingBigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage);
absl::Status MaxPoolingBatchedBigTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage);
absl::Status MaxPooling3DBigTest(TestExecutionEnvironment& env,
                                 DataType data_type, TensorStorageType storage);
absl::Status MaxPooling3DBatchedBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);
absl::Status MaxPoolingIndicesBigTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage);
absl::Status MaxPoolingIndicesBatchedBigTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage);
absl::Status MaxPoolingIndices3DBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);
absl::Status MaxPoolingIndices3DBatchedBigTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_POOLING_TEST_UTIL_H_
