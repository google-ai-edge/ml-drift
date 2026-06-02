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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_STRIDED_SLICE_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_STRIDED_SLICE_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status StridedSliceTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage);
absl::Status StridedSliceBoolTest(TestExecutionEnvironment& env,
                                  TensorStorageType storage);
absl::Status StridedSliceBigTest(TestExecutionEnvironment& exec_env,
                                 const SliceAttributes& attr,
                                 const BHWC& src_shape, DataType data_type,
                                 TensorStorageType storage);

absl::Status StridedSlice3DTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage);
absl::Status StridedSlice3DPaddedGridTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage);
absl::Status StridedSlice3DBigTest(TestExecutionEnvironment& exec_env,
                                   const Slice3DAttributes& attr,
                                   const BHWDC& src_shape, DataType data_type,
                                   TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_STRIDED_SLICE_TEST_UTIL_H_
