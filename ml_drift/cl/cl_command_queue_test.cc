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

#include "ml_drift/cl/cl_command_queue.h"

#include <cstddef>
#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_kernel.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

// The fake OpenCL entry points below write this handle to the output event,
// on success and on failure, so the tests can see whether CLCommandQueue kept
// it. A real driver leaves the output event unspecified on failure.
cl_event FakeEvent() { return reinterpret_cast<cl_event>(uintptr_t{0x1234}); }

cl_command_queue FakeQueue() {
  return reinterpret_cast<cl_command_queue>(uintptr_t{0x5678});
}

int released_events = 0;

cl_int CL_API_CALL FakeReleaseEvent(cl_event event) {
  if (event == FakeEvent()) ++released_events;
  return CL_SUCCESS;
}

cl_int CL_API_CALL FakeEnqueueNDRangeKernelOk(
    cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
    const size_t*, cl_uint, const cl_event*, cl_event* event) {
  if (event) *event = FakeEvent();
  return CL_SUCCESS;
}

cl_int CL_API_CALL FakeEnqueueNDRangeKernelFails(
    cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
    const size_t*, cl_uint, const cl_event*, cl_event* event) {
  if (event) *event = FakeEvent();
  return CL_INVALID_WORK_GROUP_SIZE;
}

cl_int CL_API_CALL FakeEnqueueMarkerOk(cl_command_queue, cl_event* event) {
  *event = FakeEvent();
  return CL_SUCCESS;
}

cl_int CL_API_CALL FakeEnqueueMarkerFails(cl_command_queue, cl_event* event) {
  *event = FakeEvent();
  return CL_OUT_OF_RESOURCES;
}

// Replaces the OpenCL entry points used by CLCommandQueue and CLEvent with
// fakes for the duration of a test, so no OpenCL device is needed.
class CLCommandQueueEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_enqueue_nd_range_kernel_ = clEnqueueNDRangeKernel;
    saved_enqueue_marker_ = clEnqueueMarker;
    saved_release_event_ = clReleaseEvent;
    clReleaseEvent = &FakeReleaseEvent;
    released_events = 0;
  }

  void TearDown() override {
    clEnqueueNDRangeKernel = saved_enqueue_nd_range_kernel_;
    clEnqueueMarker = saved_enqueue_marker_;
    clReleaseEvent = saved_release_event_;
  }

  PFN_clEnqueueNDRangeKernel saved_enqueue_nd_range_kernel_ = nullptr;
  PFN_clEnqueueMarker saved_enqueue_marker_ = nullptr;
  PFN_clReleaseEvent saved_release_event_ = nullptr;
};

TEST_F(CLCommandQueueEventTest, DispatchKeepsEventOnSuccess) {
  clEnqueueNDRangeKernel = &FakeEnqueueNDRangeKernelOk;
  CLCommandQueue queue(FakeQueue(), /*has_ownership=*/false);
  CLKernel kernel;

  absl::Status status;
  cl_event kept_event = nullptr;
  {
    CLEvent event;
    status = queue.Dispatch(kernel, int3(1, 1, 1), int3(1, 1, 1), &event);
    kept_event = event.event();
  }

  EXPECT_THAT(status, IsOk());
  EXPECT_EQ(kept_event, FakeEvent());
  EXPECT_EQ(released_events, 1);
}

TEST_F(CLCommandQueueEventTest, DispatchDoesNotKeepEventOnFailure) {
  clEnqueueNDRangeKernel = &FakeEnqueueNDRangeKernelFails;
  CLCommandQueue queue(FakeQueue(), /*has_ownership=*/false);
  CLKernel kernel;

  absl::Status status;
  bool event_kept = true;
  {
    CLEvent event;
    status = queue.Dispatch(kernel, int3(1, 1, 1), int3(1, 1, 1), &event);
    event_kept = event.is_valid();
  }

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnknown));
  EXPECT_FALSE(event_kept);
  EXPECT_EQ(released_events, 0);
}

TEST_F(CLCommandQueueEventTest, EnqueueEventKeepsEventOnSuccess) {
  clEnqueueMarker = &FakeEnqueueMarkerOk;
  CLCommandQueue queue(FakeQueue(), /*has_ownership=*/false);

  absl::Status status;
  cl_event kept_event = nullptr;
  {
    CLEvent event;
    status = queue.EnqueueEvent(&event);
    kept_event = event.event();
  }

  EXPECT_THAT(status, IsOk());
  EXPECT_EQ(kept_event, FakeEvent());
  EXPECT_EQ(released_events, 1);
}

TEST_F(CLCommandQueueEventTest, EnqueueEventDoesNotKeepEventOnFailure) {
  clEnqueueMarker = &FakeEnqueueMarkerFails;
  CLCommandQueue queue(FakeQueue(), /*has_ownership=*/false);

  absl::Status status;
  bool event_kept = true;
  {
    CLEvent event;
    status = queue.EnqueueEvent(&event);
    event_kept = event.is_valid();
  }

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnknown));
  EXPECT_FALSE(event_kept);
  EXPECT_EQ(released_events, 0);
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
