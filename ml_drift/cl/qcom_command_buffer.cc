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

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/command_buffer.h"
#include "ml_drift/cl/qcom_wrapper.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util.h"

namespace ml_drift {
namespace cl {

class QcomCommandBuffer : public CommandBuffer {
 public:
  QcomCommandBuffer() = default;
  ~QcomCommandBuffer() override { Release(); }
  void Release() {
    if (recording_) {
      clReleaseRecordingQCOM(recording_);
      recording_ = nullptr;
    }
  }

  // Move only
  QcomCommandBuffer(QcomCommandBuffer&& other)
      : temp_queue_(std::move(other.temp_queue_)),
        recording_(other.recording_) {
    other.recording_ = nullptr;
  }
  QcomCommandBuffer& operator=(QcomCommandBuffer&& other) {
    if (this != &other) {
      Release();
      temp_queue_ = std::move(other.temp_queue_);
      std::swap(recording_, other.recording_);
    }
    return *this;
  }
  QcomCommandBuffer(const QcomCommandBuffer&) = delete;
  QcomCommandBuffer& operator=(const QcomCommandBuffer&) = delete;

  absl::Status Init(const CLDevice& device, const CLContext& context) {
    cl_int error;
    cl_command_queue recordable_queue = clCreateCommandQueue(
        context.context(), device.id(), CL_QUEUE_RECORDABLE_QCOM, &error);
    if (error != CL_SUCCESS) {
      return absl::InternalError(
          "Unable to create queue with CL_QUEUE_RECORDABLE_QCOM property. " +
          CLErrorCodeToString(error));
    }
    temp_queue_ = std::make_unique<CLCommandQueue>(recordable_queue,
                                                   /* has_ownership = */ true);
    recording_ = clNewRecordingQCOM(recordable_queue, &error);
    if (error != CL_SUCCESS) {
      return absl::InternalError("Unable to create Qcom recording. " +
                                 CLErrorCodeToString(error));
    }
    return absl::OkStatus();
  }
  absl::Status Finalize() override {
    cl_int error = clEndRecordingQCOM(recording_);
    if (error != CL_SUCCESS) {
      return absl::InternalError("Failed recording to Qcom recording. " +
                                 CLErrorCodeToString(error));
    }
    temp_queue_.reset();
    return absl::OkStatus();
  }

  absl::Status AddOp(ClOperation* op) override {
    return op->AddToQueue(temp_queue_.get());
  }
  absl::Status Enqueue(CLCommandQueue* queue, CLEvent* event) override {
    cl_event resulting_event;
    cl_int errcode_ret = clEnqueueRecordingQCOM(
        queue->queue(), recording_, 0, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, nullptr, event ? &resulting_event : nullptr);
    if (errcode_ret != CL_SUCCESS) {
      return absl::InternalError("Failed Qcom recording execution. " +
                                 CLErrorCodeToString(errcode_ret));
    }
    if (event) {
      *event = CLEvent(resulting_event);
    }
    return absl::OkStatus();
  }

  std::unique_ptr<CLCommandQueue> temp_queue_;
  cl_recording_qcom recording_ = nullptr;
};

bool IsQcomCommandBufferSupported(const CLDevice& device) {
  if (!device.GetInfo().SupportsExtension("cl_qcom_recordable_queues")) {
    return false;
  }
  if (!clNewRecordingQCOM || !clEndRecordingQCOM || !clEnqueueRecordingQCOM ||
      !clReleaseRecordingQCOM) {
    return false;
  }
  return true;
}

size_t GetQcomCommandBufferMaxSize(const CLDevice& device) {
  return GetDeviceInfo<cl_uint>(device.id(),
                                CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>> CreateQcomCommandBuffer(
    const CLDevice& device, const CLContext& context) {
  QcomCommandBuffer cb;
  ABSL_RETURN_IF_ERROR(cb.Init(device, context));
  return std::make_unique<QcomCommandBuffer>(std::move(cb));
}

}  // namespace cl
}  // namespace ml_drift
