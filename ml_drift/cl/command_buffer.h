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

#ifndef ML_DRIFT_CL_COMMAND_BUFFER_H_
#define ML_DRIFT_CL_COMMAND_BUFFER_H_

#include "absl/status/status.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_operation.h"

namespace ml_drift {
namespace cl {

class CommandBuffer {
 public:
  CommandBuffer() = default;
  virtual ~CommandBuffer() = default;

  // Move only
  CommandBuffer(CommandBuffer&& storage) = default;
  CommandBuffer& operator=(CommandBuffer&& storage) = default;
  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  virtual absl::Status Finalize() = 0;
  virtual absl::Status AddOp(ClOperation* op) = 0;
  virtual absl::Status Enqueue(CLCommandQueue* queue, CLEvent* event) = 0;
  absl::Status Enqueue(CLCommandQueue* queue) {
    return Enqueue(queue, nullptr);
  }
};

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_COMMAND_BUFFER_H_
