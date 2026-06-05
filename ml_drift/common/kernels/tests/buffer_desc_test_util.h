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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_BUFFER_DESC_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_BUFFER_DESC_TEST_UTIL_H_

#include "ml_drift/common/status.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status ReadAsI16SelectorTest(TestExecutionEnvironment& env);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_BUFFER_DESC_TEST_UTIL_H_
