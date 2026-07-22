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

#include "ml_drift/cl/cl_event.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace cl {

CLEvent::CLEvent(cl_event event) : event_(event) {}

CLEvent::CLEvent(const CLEvent& event)
    : event_(event.event_), name_(event.name_) {
  Retain();
}

CLEvent::CLEvent(CLEvent&& event)
    : event_(event.event_), name_(std::move(event.name_)) {
  event.event_ = nullptr;
}

CLEvent& CLEvent::operator=(const CLEvent& event) {
  if (this != &event) {
    Release();
    event_ = event.event_;
    name_ = event.name_;
    Retain();
  }
  return *this;
}

CLEvent& CLEvent::operator=(CLEvent&& event) {
  if (this != &event) {
    Release();
    std::swap(event_, event.event_);
    name_ = std::move(event.name_);
  }
  return *this;
}

uint64_t CLEvent::GetEnqueuedTimeNs() const {
  cl_ulong time_ns;
  clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong),
                          &time_ns, /*param_value_size_ret=*/nullptr);
  return time_ns;
}

uint64_t CLEvent::GetSubmittedTimeNs() const {
  cl_ulong time_ns;
  clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
                          &time_ns, /*param_value_size_ret=*/nullptr);
  return time_ns;
}

uint64_t CLEvent::GetStartedTimeNs() const {
  cl_ulong time_ns;
  clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                          &time_ns, /*param_value_size_ret=*/nullptr);
  return time_ns;
}

uint64_t CLEvent::GetFinishedTimeNs() const {
  cl_ulong time_ns;
  clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &time_ns, /*param_value_size_ret=*/nullptr);
  return time_ns;
}

uint64_t CLEvent::GetCompletedTimeNs() const {
  cl_ulong time_ns;
  clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_COMPLETE,
                          sizeof(cl_ulong), &time_ns,
                          /*param_value_size_ret=*/nullptr);
  return time_ns;
}

double CLEvent::GetEventTimeMs() const {
  const uint64_t start = GetStartedTimeNs();
  const uint64_t end = GetFinishedTimeNs();
  const uint64_t time_ns = (end - start);

  return static_cast<double>(time_ns) * 1e-6;
}

uint64_t CLEvent::GetEventTimeNs() const {
  return GetFinishedTimeNs() - GetStartedTimeNs();
}

void CLEvent::SetName(const std::string& name) { name_ = name; }

absl::StatusOr<cl_int> CLEvent::GetState() const {
  cl_int status;
  cl_int error = clGetEventInfo(event_, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                sizeof(cl_int), &status, nullptr);
  if (error != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to clGetEventInfo - ",
                                           CLErrorCodeToString(error)));
  }
  return status;
}

void CLEvent::Wait() const { clWaitForEvents(1, &event_); }

absl::Status CLEvent::ActiveWait(cl_command_queue queue) const {
  cl_int event_status = CL_QUEUED;
  while (event_status != CL_COMPLETE) {
    ABSL_ASSIGN_OR_RETURN(event_status, GetState());
    if (queue) {
      clFlush(queue);
    }
  }
  return absl::OkStatus();
}

CLEvent::~CLEvent() { Release(); }

void CLEvent::Release() {
  if (event_) {
    clReleaseEvent(event_);
    event_ = nullptr;
  }
}

void CLEvent::Retain() {
  if (event_) {
    if (clRetainEvent(event_) != CL_SUCCESS) {
      event_ = nullptr;
    }
  }
}

}  // namespace cl
}  // namespace ml_drift
