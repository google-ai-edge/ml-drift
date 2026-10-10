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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONCAT_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONCAT_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ConcatChannelsBoolTest(TestExecutionEnvironment& env,
                                    TensorStorageType storage);
template <DataType data_type>
absl::Status ConcatIntTest(TestExecutionEnvironment& env,
                           TensorStorageType storage);
absl::Status ConcatWidthTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage);
absl::Status ConcatHeightTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage);
absl::Status ConcatChannelsTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status ConcatChannelsAlignedx4Test(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage);

// Concat sources need not share a data type. The delegate folds fp16 weight
// DEQUANTIZEs away, so under CalculationsPrecision::F32 -- where per-value
// types are preserved -- an f16 constant can be concatenated with an f32
// computed value. These cover both kernels, and both directions of narrowing,
// with the destination type stated explicitly rather than inferred.
absl::Status ConcatChannelsMixedTypesTest(TestExecutionEnvironment& env,
                                          TensorStorageType storage);
absl::Status ConcatWidthMixedTypesTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage);
absl::Status ConcatChannelsMixedTypesF16DstTest(TestExecutionEnvironment& env,
                                                TensorStorageType storage);

absl::Status ConcatWidthBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status ConcatWidthBatchedBigTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage);
absl::Status ConcatHeightBigTest(TestExecutionEnvironment& env,
                                 DataType data_type, TensorStorageType storage);
absl::Status ConcatHeightBatchedBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);
absl::Status ConcatBatchBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status ConcatDepthBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status ConcatDepthBatchedBigTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage);
absl::Status ConcatChannelsBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status ConcatChannelsBatchedBigTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage);
absl::Status ConcatChannelsx4BigTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage);
absl::Status ConcatChannelsx4BatchedBigTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage);
absl::Status ConcatChannelsBHWDCBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONCAT_TEST_UTIL_H_
