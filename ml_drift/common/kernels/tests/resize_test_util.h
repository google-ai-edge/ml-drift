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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_RESIZE_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_RESIZE_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ResizeBilinearAlignedTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage);
absl::Status ResizeBilinearNonAlignedTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage);
absl::Status ResizeBilinearWithoutHalfPixelTest(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage);
absl::Status ResizeBilinearWithHalfPixelTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage);
absl::Status ResizeNearestTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage);
absl::Status ResizeNearestAlignCornersTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage);
absl::Status ResizeNearestHalfPixelCentersTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage);

absl::Status ResizeBilinearAlignedBigTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage);
absl::Status ResizeBilinearNonAlignedBigTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage);
absl::Status ResizeBilinearAlignedBatchedBigTest(TestExecutionEnvironment& env,
                                                 DataType data_type,
                                                 TensorStorageType storage);
absl::Status ResizeBilinearNonAlignedBatchedBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status ResizeNearestBigTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage);
absl::Status ResizeNearestBatchedBigTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage);
absl::Status ResizeBilinear3DAlignedBigTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage);
absl::Status ResizeBilinear3DNonAlignedBigTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage);
absl::Status ResizeBilinear3DAlignedBatchedBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status ResizeBilinear3DNonAlignedBatchedBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status ResizeNearest3DBigTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage);
absl::Status ResizeNearest3DBatchedBigTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_RESIZE_TEST_UTIL_H_
