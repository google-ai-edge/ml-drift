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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_TRANSPOSE_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_TRANSPOSE_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

template <DataType T>
absl::Status TransposeIntTest(TestExecutionEnvironment& env,
                              TensorStorageType storage);

template <DataType T>
absl::Status TransposeUintTest(TestExecutionEnvironment& env,
                               TensorStorageType storage);

absl::Status TransposeTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage);

absl::Status TransposeTest(TestExecutionEnvironment& exec_env,
                           const TransposeAttributes& attr,
                           const BHWC& src_shape, DataType data_type,
                           TensorStorageType storage);

absl::Status Transpose3DTest(TestExecutionEnvironment& exec_env,
                             const Transpose3DAttributes& attr,
                             const BHWDC& src_shape, DataType data_type,
                             TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_TRANSPOSE_TEST_UTIL_H_
