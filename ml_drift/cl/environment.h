// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_CL_ENVIRONMENT_H_
#define ML_DRIFT_CL_ENVIRONMENT_H_

#include <vector>

#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/program_cache.h"
#include "ml_drift/cl/util_types.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace cl {

class Environment {
 public:
  Environment() = default;
  explicit Environment(CLDevice&& device, CLContext&& context,
                       CLCommandQueue&& queue,
                       ProfilingCommandQueue&& profiling_queue);
  // Move only
  Environment(Environment&& environment);
  Environment& operator=(Environment&& environment);
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;

  const CLDevice& device() const { return device_; }
  CLDevice* GetDevicePtr() { return &device_; }
  const CLDevice* GetDevicePtr() const { return &device_; }
  CLContext& context() { return context_; }
  const CLContext& context() const { return context_; }
  CLCommandQueue* queue() { return &queue_; }
  ProfilingCommandQueue* profiling_queue() { return &profiling_queue_; }
  ProgramCache* program_cache() { return &program_cache_; }
  const ProgramCache* program_cache() const { return &program_cache_; }

  std::vector<CalculationsPrecision> GetSupportedPrecisions() const;
  bool IsSupported(CalculationsPrecision precision) const;
  std::vector<TensorStorageType> GetSupportedStorages() const;
  // returns storage types that support zero clamping when reading OOB in HW
  // (Height/Width) dimensions.
  std::vector<TensorStorageType> GetSupportedStoragesWithHWZeroClampSupport()
      const;
  bool IsSupported(TensorStorageType storage_type) const;

  absl::Status Init();

 private:
  CLDevice device_;
  CLContext context_;
  CLCommandQueue queue_;
  ProfilingCommandQueue profiling_queue_;
  ProgramCache program_cache_;
};

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);
TensorStorageType GetStorageTypeWithMinimalMemoryConsumption(
    const GpuInfo& gpu_info);

// Checks if image 2D creation from sub-buffer is supported.
bool CanUseSubBufferForImage2d(const GpuInfo& gpu_info);

struct EnvironmentOptions {
  // The priority affects the order work is taken from the queues, with high
  // priority work taken first.
  PriorityHint priority = PriorityHint::kNormal;

  // Desired performance level. Higher performance implies higher frequencies
  // and power usage.
  PerformanceHint performance = PerformanceHint::kHigh;
};

absl::Status CreateEnvironment(Environment* result,
                               const EnvironmentOptions& options = {});

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_ENVIRONMENT_H_
