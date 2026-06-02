// Copyright 2025 The ML Drift Authors.
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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_KERNEL_TEST_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_KERNEL_TEST_H_

#include "ml_drift/common/task/testing_util.h"

// Global variable to be overwritten by {backend}_kernel_test.cc.
extern ml_drift::TestExecutionEnvironment* exec_env;

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_KERNEL_TEST_H_
