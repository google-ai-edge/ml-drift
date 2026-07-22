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

#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/status.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util_types.h"
#include "ml_drift/common/gpu_info.h"

namespace ml_drift {
namespace cl {
namespace {

cl_context_properties PriorityToContextPropertyKhr(PriorityHint hint) {
  switch (hint) {
    case PriorityHint::kHigh:
      return CL_QUEUE_PRIORITY_HIGH_KHR;
    default:
    case PriorityHint::kNormal:
      return CL_QUEUE_PRIORITY_MED_KHR;
    case PriorityHint::kLow:
      return CL_QUEUE_PRIORITY_LOW_KHR;
  }
}

}  // namespace

absl::Status CreateQcomConvolutionFilter(cl_context context, int kernel_x,
                                         int kernel_y, cl_mem* filter,
                                         const void* data) {
  return absl::UnavailableError("CreateQcomConvolutionFilter not available.");
}

std::vector<std::string> GetUnsupportedExtensions() {
  return {"cl_qcom_accelerated_image_ops", "cl_qcom_recordable_queues",
          "cl_qcom_perf_hint", "cl_qcom_priority_hint"};
}

std::vector<std::string> GetPrivateExtensions(const GpuInfo& gpu_info) {
  std::vector<std::string> extensions;
  if (gpu_info.IsIntel()) {
    // On intel we have GPR File that any register can be accessed from any
    // thread. So usual 'private' registers can be used as wave memory.
    if (gpu_info.opencl_info.IsCLVK()) {
      // for CLVK it will work because of specific driver
      extensions.push_back("ucl_wave_memory");
    } else {
      const bool supports_subgroups =
          (gpu_info.SupportsExtension("cl_khr_subgroups") ||
           gpu_info.SupportsExtension("cl_intel_subgroups"));
      if (supports_subgroups &&
          gpu_info.SupportsExtension("cl_intel_required_subgroup_size")) {
        extensions.push_back("ucl_wave_memory");
      }
    }
  }
  if (gpu_info.IsPowerVR()) {
    if (gpu_info.powervr_info.gpu_version >= PowerVRGpu::kCXT &&
        gpu_info.SupportsExtension("cl_khr_subgroups")) {
      extensions.push_back("ucl_wave_memory");
    }
  }
  return extensions;
}

void AppendContextHints(const GpuInfo& gpu_info,
                        PerformanceHint performance_hint,
                        PriorityHint priority_hint,
                        std::vector<cl_context_properties>* props) {
  // There's no extension for priority or performance context hints yet.
}

void AppendCommandQueueHints(const GpuInfo& gpu_info,
                             PriorityHint priority_hint,
                             std::vector<cl_queue_properties>* props) {
  if (gpu_info.SupportsExtension("cl_khr_priority_hints")) {
    props->push_back(CL_QUEUE_PRIORITY_KHR);
    props->push_back(PriorityToContextPropertyKhr(priority_hint));
  }
}

void FixAdrenoBinary(std::vector<uint8_t>* binary) {
  // No-op.
}

}  // namespace cl
}  // namespace ml_drift
