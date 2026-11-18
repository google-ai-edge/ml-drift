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

#include "ml_drift/cl/cl_command_buffer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/cl/opencl_wrapper.h"

namespace ml_drift {
namespace cl {
namespace {

using ::absl_testing::StatusIs;

TEST(CLCommandBufferTest, InitReturnsUnavailableWhenFunctionPointerIsNull) {
  PFN_clCreateCommandBufferKHR saved_fn = clCreateCommandBufferKHR;
  absl::Cleanup restore_fn = [&] { clCreateCommandBufferKHR = saved_fn; };
  clCreateCommandBufferKHR = nullptr;

  CLCommandBuffer cb;
  EXPECT_THAT(cb.Init(/*queue=*/nullptr, /*simultaneous_use=*/false),
              StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(cb.Init(/*queue=*/nullptr, /*simultaneous_use=*/true),
              StatusIs(absl::StatusCode::kUnavailable));
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
