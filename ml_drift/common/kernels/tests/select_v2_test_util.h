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

#ifndef ML_DRIFT_COMMON_KERNELS_SELECT_V2_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_SELECT_V2_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

template <DataType data_type>
absl::Status IfTest(TestExecutionEnvironment& env, DataType src_data_type,
                    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2Test(TestExecutionEnvironment& env, DataType data_type,
                          TensorStorageType storage,
                          TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2Scalar4DTest(TestExecutionEnvironment& env,
                                  DataType data_type, TensorStorageType storage,
                                  TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2TrueValueTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage,
                                   TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2FalseValueTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage,
                                    TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BatchTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage,
                               TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2ChannelsTest(TestExecutionEnvironment& env,
                                  DataType data_type, TensorStorageType storage,
                                  TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2ChannelsBatchTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage,
                                       TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BroadcastTrueTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage,
                                       TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BroadcastFalseTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage,
                                        TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BroadcastBothTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage,
                                       TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2ChannelsBroadcastFalseTest(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage,
                                                TensorStorageType cond_storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SELECT_V2_TEST_UTIL_H_
