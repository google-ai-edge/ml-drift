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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_REDUCE_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_REDUCE_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

template <DataType data_type>
absl::Status ReduceSumChannelsIntTest(TestExecutionEnvironment& env,
                                      TensorStorageType storage);
template <DataType data_type>
absl::Status ReduceProductChannelsUIntTest(TestExecutionEnvironment& env,
                                           TensorStorageType storage);
absl::Status ReduceAllTest(TestExecutionEnvironment& env,
                           TensorStorageType storage);
absl::Status ReduceAnyTest(TestExecutionEnvironment& env,
                           TensorStorageType storage);

absl::Status MeanHWTest(TestExecutionEnvironment& env, DataType data_type,
                        TensorStorageType storage);
absl::Status ReduceSumChannelsTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status ReduceProductChannelsTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage);
absl::Status ReduceMaxChannelsTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status ReduceMinChannelsTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status ReduceMaxIndChannelsTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage);
absl::Status ReduceMaxIndHeightTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage);

absl::Status ReduceHWBigTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage, OperationType op_type);
absl::Status ReduceHWBatchedBigTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage,
                                    OperationType op_type);
absl::Status ReduceHBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage, OperationType op_type);
absl::Status ReduceBHBigTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage, OperationType op_type);
absl::Status ReduceCBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage, OperationType op_type);
absl::Status ReduceCx4BigTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage, OperationType op_type);
absl::Status ReduceHWCBigTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage, OperationType op_type);
absl::Status ReduceBHDBigTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage, OperationType op_type);
absl::Status ReduceBHWDBigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage,
                               OperationType op_type);
}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_REDUCE_TEST_UTIL_H_
