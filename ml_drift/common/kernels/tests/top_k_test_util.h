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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_TOP_K_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_TOP_K_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status TopKTest(TestExecutionEnvironment& env, DataType data_type,
                      TensorStorageType storage);
absl::Status TopKPartialReductionTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage);
absl::Status TopKBigTest(TestExecutionEnvironment& env, DataType data_type,
                         TensorStorageType storage);
absl::Status TopKBig2StepTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage);
absl::Status TopKBig2StepFirstStepNoWgReductionTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status TopKIterativeTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage);
absl::Status TopKIterativeBigTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage);
absl::Status TopKIterative2StepBigTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_TOP_K_TEST_UTIL_H_
