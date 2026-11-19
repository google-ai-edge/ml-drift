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

#ifndef ML_DRIFT_CL_CL_COMMAND_QUEUE_H_
#define ML_DRIFT_CL_CL_COMMAND_QUEUE_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_kernel.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util_types.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {

// A wrapper around opencl command queue
class CLCommandQueue {
 public:
  CLCommandQueue();
  CLCommandQueue(cl_command_queue queue, bool has_ownership);

  // Move only
  CLCommandQueue(CLCommandQueue&& queue);
  CLCommandQueue& operator=(CLCommandQueue&& queue);
  CLCommandQueue(const CLCommandQueue&) = delete;
  CLCommandQueue& operator=(const CLCommandQueue&) = delete;

  virtual ~CLCommandQueue();

  cl_command_queue queue() const { return queue_; }

  absl::Status Dispatch(const CLKernel& kernel, const int3& work_groups_count,
                        const int3& work_group_size, CLEvent* event = nullptr);

  absl::Status EnqueueEvent(CLEvent* event);

  absl::Status EnqueueWriteImage(cl_mem memory, int3 region, const void* data,
                                 bool async = false);
  absl::Status EnqueueReadImage(cl_mem memory, int3 region, void* data,
                                bool async = false);

  absl::Status EnqueueWriteBuffer(cl_mem memory, size_t size_in_bytes,
                                  const void* data, bool async = false);
  absl::Status EnqueueReadBuffer(cl_mem memory, size_t size_in_bytes,
                                 void* data, bool async = false);

  absl::Status WaitForCompletion();

 protected:
  void Release();

  cl_command_queue queue_ = nullptr;
  bool has_ownership_ = false;
};

class ProfilingCommandQueue : public CLCommandQueue {
 public:
  ProfilingCommandQueue();
  explicit ProfilingCommandQueue(cl_command_queue queue);

  // Move only
  ProfilingCommandQueue(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue& operator=(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue(const ProfilingCommandQueue&) = delete;
  ProfilingCommandQueue& operator=(const ProfilingCommandQueue&) = delete;

  absl::Status ProfilingDispatch(const CLKernel& kernel,
                                 const int3& work_groups_count,
                                 const int3& work_group_size, int n = 1,
                                 int flush_period = 0);

  // will write index for fastest work_group among work_group_sizes
  absl::Status GetBestWorkGroupIndex(const CLKernel& kernel,
                                     const GpuInfo& gpu_info,
                                     const std::vector<int3>& work_groups_count,
                                     const std::vector<int3>& work_group_sizes,
                                     int* index);

  // call ResetMeasurements() to start new series of measurements
  void ResetMeasurements();

  double GetQueueExecutionTimeMs() const;

  // Difference from GetQueueExecutionTimeMs is that this number doesn't include
  // time between kernels(kernels launches or preparing) on GPU. Usually, this
  // time should be 5-10% better than GetQueueExecutionTimeMs, because 5-10%
  // spend on something else(maybe kernels launches or preparing)
  double GetSumOfEventsTimeMs() const;

  // This label will be used for all subsequent dispatches.
  void SetEventsLabel(const std::string& name);

  ProfilingInfo GetProfilingInfo() const;

 private:
  std::vector<CLEvent> events_;
  std::vector<int> number_of_dispatches_;
  std::string current_label_;
};

struct CLCommandQueueOptions {
  // The priority affects the order work is taken from the queues, with high
  // priority work taken first.
  // Defaults to kNormal according to the extension specs.
  PriorityHint priority = PriorityHint::kNormal;
};

absl::Status CreateCLCommandQueue(const CLDevice& device,
                                  const CLContext& context,
                                  CLCommandQueue* result,
                                  CLCommandQueueOptions options = {});

absl::Status CreateProfilingCommandQueue(const CLDevice& device,
                                         const CLContext& context,
                                         ProfilingCommandQueue* result,
                                         CLCommandQueueOptions options = {});

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_COMMAND_QUEUE_H_
