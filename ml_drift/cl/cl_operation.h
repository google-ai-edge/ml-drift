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

#ifndef ML_DRIFT_CL_CL_OPERATION_H_
#define ML_DRIFT_CL_CL_OPERATION_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/time/time.h"
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_arguments.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_kernel.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/program_cache.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {

class ClOperation {
 public:
  ClOperation() = default;
  virtual ~ClOperation() = default;
  // Move only
  ClOperation(ClOperation&& operation) = default;
  ClOperation& operator=(ClOperation&& operation) = default;
  ClOperation(const ClOperation&) = delete;
  ClOperation& operator=(const ClOperation&) = delete;

  void Init(std::unique_ptr<GPUOperation>&& gpu_operation) {
    operation_ = std::move(gpu_operation);
  }

  absl::Status InitArgs(const GpuInfo& gpu_info, CLContext* context);
  absl::Status InitArgsDeserialized(const GpuInfo& gpu_info,
                                    CLContext* context);

  uint64_t GetKernelFingerprint() const { return kernel_fingerprint_; }

  // should be called after changes of inputs/outputs.
  absl::Status UpdateParams();

  absl::Status SetInt(const std::string& name, int value) {
    return cl_args_.SetInt(name, value);
  }
  absl::Status SetUint(const std::string& name, unsigned int value) {
    return cl_args_.SetUint(name, value);
  }
  absl::Status SetFloat(const std::string& name, float value) {
    return cl_args_.SetFloat(name, value);
  }
  absl::Status SetHalf(const std::string& name, float value) {
    return cl_args_.SetHalf(name, half(value));
  }

  absl::Status SetSrcTensor(int index, Tensor* tensor);
  absl::Status SetDstTensor(int index, Tensor* tensor);
  absl::Status SetSrcBuffer(int index, Buffer* buffer);
  absl::Status SetDstBuffer(int index, Buffer* buffer);

  absl::Status AddToQueue(CLCommandQueue* queue, CLEvent* event = nullptr);
  absl::Status AddToCommandBuffer(cl_command_buffer_khr cb);
  absl::Status AddToQueueForProfiling(ProfilingCommandQueue* queue, int n = 1,
                                      int flush_period = 0);

  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);

  absl::Status Compile(const CLDevice* device, CLContext* context,
                       ProgramCache* cache);

  absl::Status RestoreDeserialized(const ProgramCache& program_cache,
                                   uint64_t fingerprint,
                                   const GpuInfo& gpu_info,
                                   const int3& work_group_size,
                                   CLContext* context);

  int3 GetWorkGroupSize() const { return operation_->work_group_size_; }
  void SetWorkGroupSize(const int3& wg_size);

  uint64_t GetConstArgsSize() const { return operation_->const_args_size_; }
  uint64_t GetFlopsCount() const { return operation_->flops_; }
  uint64_t GetReadSize() const { return operation_->read_size_; }
  uint64_t GetWriteSize() const { return operation_->write_size_; }

  bool HasEqualScalarArguments(const ClOperation& op) const {
    return cl_args_.HasEqualScalarArguments(op.cl_args_);
  }

  // Can take long time and use different approaches for different devices.
  absl::StatusOr<absl::Duration> GetOpTime(
      const GpuInfo& gpu_info, CLCommandQueue* queue,
      ProfilingCommandQueue* profiling_queue);

 private:
  std::unique_ptr<GPUOperation> operation_;
  CLKernel kernel_;
  uint64_t kernel_fingerprint_;
  CLArguments cl_args_;
};

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_OPERATION_H_
