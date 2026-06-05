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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_WINOGRAD_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_WINOGRAD_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status Winograd3x3ForwardTiledTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage,
                                         const BHWC& src_shape,
                                         int tile_size = 6);
absl::Status Winograd3x3BackwardTiledTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage,
                                          const BHWC& initial_shape,
                                          int tile_size = 6);
absl::Status Winograd4x4To36Test(TestExecutionEnvironment& env,
                                 DataType data_type, TensorStorageType storage,
                                 int tile_size = 6);
absl::Status Winograd4x4To36BatchTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage,
                                      int tile_size = 6);
absl::Status Winograd36To4x4Test(TestExecutionEnvironment& env,
                                 DataType data_type, TensorStorageType storage,
                                 int tile_size = 6);

absl::Status Winograd3x3To36Test(TestExecutionEnvironment& env);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_WINOGRAD_TEST_UTIL_H_
