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

#include "ml_drift/cl/cl_operation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/program_cache.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/cl/util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {
namespace {
std::string GetCommonOpenCLDefines() {
  std::string result;
  result += "#define MAIN_FUNCTION __kernel void main_function\n";
  result += "#define bool2 uchar2\n";
  result += "#define bool3 uchar3\n";
  result += "#define bool4 uchar4\n";
  return result;
}
}  // namespace

absl::Status ClOperation::UpdateParams() {
  RETURN_IF_ERROR(operation_->BindArguments(&cl_args_));
  operation_->RecalculateGridSize();
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

absl::Status ClOperation::SetSrcTensor(int index, Tensor* tensor) {
  operation_->SetSrc(tensor, index);
  return cl_args_.SetObjectRef(operation_->GetSrcTensorsNames()[index], tensor);
}

absl::Status ClOperation::SetDstTensor(int index, Tensor* tensor) {
  operation_->SetDst(tensor, index);
  return cl_args_.SetObjectRef(operation_->GetDstTensorsNames()[index], tensor);
}

absl::Status ClOperation::SetSrcBuffer(int index, Buffer* buffer) {
  return cl_args_.SetObjectRef(operation_->GetSrcTensorsNames()[index], buffer);
}

absl::Status ClOperation::SetDstBuffer(int index, Buffer* buffer) {
  return cl_args_.SetObjectRef(operation_->GetDstTensorsNames()[index], buffer);
}

absl::Status ClOperation::AddToQueue(CLCommandQueue* queue, CLEvent* event) {
  RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
  return queue->Dispatch(kernel_, operation_->GetWorkGroupsCount(),
                         operation_->work_group_size_, event);
}

absl::Status ClOperation::AddToCommandBuffer(cl_command_buffer_khr cb) {
  RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
  std::array<size_t, 3> local;
  std::array<size_t, 3> global;
  for (int i = 0; i < 3; ++i) {
    local[i] = operation_->work_group_size_[i];
    global[i] =
        operation_->GetWorkGroupsCount()[i] * operation_->work_group_size_[i];
  }
  const int error_code = clCommandNDRangeKernelKHR(
      cb, /*command_queue=*/nullptr, /*properties=*/nullptr, kernel_.kernel(),
      /*work_dim=*/3, /*global_work_offset=*/nullptr,
      /*global_work_size=*/global.data(),
      /*local_work_size=*/local.data(), /*num_sync_points_in_wait_list=*/0,
      /*sync_point_wait_list=*/nullptr, /*sync_point=*/nullptr,
      /*mutable_handle=*/nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to clCommandNDRangeKernelKHR - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status ClOperation::AddToQueueForProfiling(ProfilingCommandQueue* queue,
                                                 int n, int flush_period) {
  RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
  return queue->ProfilingDispatch(kernel_, operation_->GetWorkGroupsCount(),
                                  operation_->work_group_size_, n,
                                  flush_period);
}

absl::Status ClOperation::InitArgs(const GpuInfo& gpu_info,
                                   CLContext* context) {
  RETURN_IF_ERROR(
      cl_args_.Init(gpu_info, context, &operation_->args_, &operation_->code_));
  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status ClOperation::InitArgsDeserialized(const GpuInfo& gpu_info,
                                               CLContext* context) {
  RETURN_IF_ERROR(cl_args_.Init(gpu_info, &operation_->args_, context));
  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status ClOperation::Compile(const CLDevice* device, CLContext* context,
                                  ProgramCache* cache) {
  const GpuInfo& gpu_info = device->info_;
  RETURN_IF_ERROR(InitArgs(gpu_info, context));
  const std::string defines = GetCommonOpenCLDefines();
  std::string extensions;
  if (gpu_info.SupportsExtension("cl_khr_fp16")) {
    // Make this optional when we have a proper way to detect half usage in
    // specific kernel.
    extensions += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }
  if (cl_args_.HasWriteOnly3dImages()) {
    extensions += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
  }
  if (gpu_info.opencl_info.IsCLVK()) {
    operation_->compiler_options_.push_back(CompilerOptions::kClVkNativeMath);
  }
  if (gpu_info.IsPowerVR()) {
    operation_->compiler_options_.push_back(
        CompilerOptions::kClUniformWorkGroupSize);
  }
  operation_->code_ = defines + extensions + operation_->code_;
  return cache->GetOrCreateCLKernel(operation_->code_, "main_function",
                                    operation_->compiler_options_, *context,
                                    *device, &kernel_, &kernel_fingerprint_);
}

absl::Status ClOperation::RestoreDeserialized(const ProgramCache& program_cache,
                                              uint64_t fingerprint,
                                              const GpuInfo& gpu_info,
                                              const int3& work_group_size,
                                              CLContext* context) {
  kernel_fingerprint_ = fingerprint;
  RETURN_IF_ERROR(
      program_cache.GetKernel(kernel_fingerprint_, "main_function", &kernel_));
  operation_->work_group_size_ = work_group_size;
  operation_->RecalculateWorkGroupsCount();
  RETURN_IF_ERROR(InitArgsDeserialized(gpu_info, context));
  return absl::OkStatus();
}

absl::Status ClOperation::Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                               ProfilingCommandQueue* profiling_queue) {
  std::vector<GPUOperation::DispatchInfo> possible_dispatches;
  operation_->GetPossibleDispatches(tuning_type, gpu_info, kernel_.info_,
                                    &possible_dispatches);
  if (possible_dispatches.empty()) {
    return absl::NotFoundError("No dispatch parameters to launch kernel");
  }
  if (possible_dispatches.size() == 1) {
    operation_->work_group_size_ = possible_dispatches[0].work_group_size;
    operation_->RecalculateWorkGroupsCount();
    return absl::OkStatus();
  } else {
    std::vector<int3> work_group_sizes(possible_dispatches.size());
    std::vector<int3> work_groups_counts(possible_dispatches.size());
    for (int i = 0; i < possible_dispatches.size(); ++i) {
      work_group_sizes[i] = possible_dispatches[i].work_group_size;
      work_groups_counts[i] = possible_dispatches[i].work_groups_count;
    }
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    int best_work_group_index;
    RETURN_IF_ERROR(profiling_queue->GetBestWorkGroupIndex(
        kernel_, gpu_info, work_groups_counts, work_group_sizes,
        &best_work_group_index));
    operation_->work_group_size_ = work_group_sizes[best_work_group_index];
    operation_->RecalculateWorkGroupsCount();
    return absl::OkStatus();
  }
}

void ClOperation::SetWorkGroupSize(const int3& wg_size) {
  operation_->work_group_size_ = wg_size;
  operation_->RecalculateWorkGroupsCount();
}

absl::StatusOr<absl::Duration> ClOperation::GetOpTime(
    const GpuInfo& gpu_info, CLCommandQueue* queue,
    ProfilingCommandQueue* profiling_queue) {
  uint64_t min_time_ns = std::numeric_limits<uint64_t>::max();
  const int kTestRuns = 16;
  std::vector<CLEvent> events(kTestRuns);
  for (int j = 0; j < kTestRuns; ++j) {
    RETURN_IF_ERROR(AddToQueue(profiling_queue, &events[j]));
  }
  RETURN_IF_ERROR(profiling_queue->WaitForCompletion());
  for (int j = 0; j < kTestRuns; ++j) {
    min_time_ns = std::min(min_time_ns, events[j].GetEventTimeNs());
  }

  const bool use_gpu_timer = gpu_info.IsAdreno() || gpu_info.IsAMD();
  if (use_gpu_timer) {
    return absl::Nanoseconds(min_time_ns);
  }
  double min_time_ms = static_cast<double>(min_time_ns) * 1e-6;

  // Approximately how many times to run to be in total ~ 1sek
  const int kRuns = std::max(4, static_cast<int>(1000.0 / min_time_ms));
  // Approximately flush every 10ms
  const int kFlush = std::max(4, static_cast<int>(10.0 / min_time_ms));
  const auto start = absl::Now();
  for (int j = 0; j < kRuns; ++j) {
    RETURN_IF_ERROR(AddToQueue(queue));
    if ((j + 1) % kFlush == 0) {
      clFlush(queue->queue());
    }
  }
  RETURN_IF_ERROR(queue->WaitForCompletion());
  const auto end = absl::Now();
  return (end - start) / kRuns;
}

}  // namespace cl
}  // namespace ml_drift
