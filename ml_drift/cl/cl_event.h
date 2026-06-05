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

#ifndef ML_DRIFT_CL_CL_EVENT_H_
#define ML_DRIFT_CL_CL_EVENT_H_

#include <cstdint>
#include <string>

#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace cl {

// A RAII wrapper around opencl event
class CLEvent {
 public:
  CLEvent() = default;
  explicit CLEvent(cl_event event);

  // Copy and move constructors/assigments.
  CLEvent(const CLEvent& event);
  CLEvent(CLEvent&& event);
  CLEvent& operator=(const CLEvent& event);
  CLEvent& operator=(CLEvent&& event);

  ~CLEvent();

  uint64_t GetEnqueuedTimeNs() const;
  uint64_t GetSubmittedTimeNs() const;
  uint64_t GetStartedTimeNs() const;
  uint64_t GetFinishedTimeNs() const;
  uint64_t GetCompletedTimeNs() const;

  double GetEventTimeMs() const;
  uint64_t GetEventTimeNs() const;

  absl::StatusOr<cl_int> GetState() const;
  void Wait() const;
  absl::Status ActiveWait(cl_command_queue queue = nullptr) const;

  cl_event event() const { return event_; }
  cl_event* GetEventPtr() { return &event_; }

  bool is_valid() const { return event_ != nullptr; }

  void SetName(const std::string& name);
  std::string GetName() const { return name_; }

 private:
  void Release();
  void Retain();

  cl_event event_ = nullptr;

  std::string name_;  // optional, for profiling mostly
};

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_EVENT_H_
