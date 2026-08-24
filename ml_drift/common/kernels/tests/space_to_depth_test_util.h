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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_SPACE_TO_DEPTH_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_SPACE_TO_DEPTH_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status SpaceToDepthTensorShape1x2x2x1BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status SpaceToDepthTensorShape1x2x2x2BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status SpaceToDepthTensorShape1x2x2x3BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status SpaceToDepthTensorShape1x4x4x1BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);

absl::Status DepthToSpaceFrom1x1x1x4To1x2x2x1Test(TestExecutionEnvironment& env,
                                                  DataType data_type,
                                                  TensorStorageType storage);
absl::Status DepthToSpaceFrom1x1x1x16To1x2x2x4Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status DepthToSpaceFrom1x2x2x16To1x4x4x4Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_SPACE_TO_DEPTH_TEST_UTIL_H_
