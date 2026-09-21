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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_FULLY_CONNECTED_OI_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_FULLY_CONNECTED_OI_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status FullyConnectedOITest(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage, bool isI4O4);
absl::Status FullyConnectedOIWi8(TestExecutionEnvironment& env,
                                 CalculationsPrecision precision,
                                 TensorStorageType storage, bool isI4O4);
absl::Status FullyConnectedOIWi4(TestExecutionEnvironment& env,
                                 CalculationsPrecision precision,
                                 TensorStorageType storage);
absl::Status FullyConnectedOIWi4Sparse2x4(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage);

absl::Status FullyConnectedOIRuntimeSrcTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage);
absl::Status FullyConnectedOIRuntimeDstTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage);

absl::Status FullyConnectedOIPackedGroupsTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage);

absl::Status FullyConnectedOIRingedOTest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage);
absl::Status FullyConnectedOIRingedITest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_FULLY_CONNECTED_OI_TEST_UTIL_H_
