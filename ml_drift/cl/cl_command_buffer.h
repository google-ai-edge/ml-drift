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

#ifndef ML_DRIFT_CL_CL_COMMAND_BUFFER_H_
#define ML_DRIFT_CL_CL_COMMAND_BUFFER_H_

#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/command_buffer.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace cl {

class CLCommandBuffer : public CommandBuffer {
 public:
  CLCommandBuffer() = default;
  // Move only
  CLCommandBuffer(CLCommandBuffer&& cb);
  CLCommandBuffer& operator=(CLCommandBuffer&& cb);
  CLCommandBuffer(const CLCommandBuffer&) = delete;
  CLCommandBuffer& operator=(const CLCommandBuffer&) = delete;

  ~CLCommandBuffer() { Release(); }

  absl::Status Init(CLCommandQueue* queue, bool simultaneous_use = false);
  absl::Status Finalize() override;
  absl::Status AddOp(ClOperation* op) override;
  absl::Status Enqueue(CLCommandQueue* queue, CLEvent* event) override;
  cl_command_buffer_khr GetCommandBuffer() const { return cb_; }

 private:
  void Release();
  cl_command_buffer_khr cb_ = nullptr;
};

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_COMMAND_BUFFER_H_
