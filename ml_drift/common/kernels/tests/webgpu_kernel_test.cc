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

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/webgpu/testing/webgpu_test.h"
// Global variable declared in kernel_test.h
ml_drift::TestExecutionEnvironment* exec_env;
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ml_drift::webgpu::WebGpuOperationTestEnvironment* env =
      new ml_drift::webgpu::WebGpuOperationTestEnvironment();
  ::testing::AddGlobalTestEnvironment(env);
  exec_env = &env->exec_env_;
#ifdef __EMSCRIPTEN__
  // Asyncify stack unwinding doesn't play nicely with onExit handlers from
  // main(), even with the necessary `-sEXIT_RUNTIME=1` linkopt. Therefore, in
  // that case, we manually force an exit call once we have our status.
  // See https://github.com/emscripten-core/emscripten/issues/14940 for more
  // details.
  if (emscripten_has_asyncify()) {
    auto res = RUN_ALL_TESTS();
    exit(res);
  }
#endif
  return RUN_ALL_TESTS();
}
