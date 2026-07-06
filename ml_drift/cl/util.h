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

#ifndef ML_DRIFT_CL_UTIL_H_
#define ML_DRIFT_CL_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util_types.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace cl {

std::string CLErrorCodeToString(cl_int error_code);

int ChannelTypeToSizeInBytes(cl_channel_type type);

bool OpenCLSupported();

template <DataType S, typename T>
void CopyLinearFLT4(const ml_drift::Tensor<Linear, S>& src, absl::Span<T> dst) {
  const int dst_depth = dst.size();
  for (int d = 0; d < dst_depth; ++d) {
    T val;
    for (int i = 0; i < 4; ++i) {
      const int dst_ch = d * 4 + i;
      val[i] = dst_ch >= src.shape.v ? 0.0f : src.data[dst_ch];
    }
    dst[d] = val;
  }
}

absl::Status CreateCLBuffer(cl_context context, size_t size_in_bytes,
                            bool read_only, void* data, cl_mem* result);

absl::Status CreateCLSubBuffer(cl_context context, cl_mem parent,
                               size_t origin_in_bytes, size_t size_in_bytes,
                               bool read_only, cl_mem* result);

absl::Status CreateRGBAImage2D(cl_context context, int width, int height,
                               cl_channel_type channel_type, void* data,
                               cl_mem* result);

absl::StatusOr<cl_mem_flags> GetCLMemObjectFlags(cl_mem memobj);

absl::Status CreateQcomConvolutionFilter(cl_context context, int kernel_x,
                                         int kernel_y, cl_mem* filter,
                                         const void* data);

// Extensions that cannot be used in open-source
std::vector<std::string> GetUnsupportedExtensions();
// Extensions that cannot be used in open-source
std::vector<std::string> GetPrivateExtensions(const GpuInfo& gpu_info);

void AppendContextHints(const GpuInfo& gpu_info,
                        PerformanceHint performance_hint,
                        PriorityHint priority_hint,
                        std::vector<cl_context_properties>* props);

void AppendCommandQueueHints(const GpuInfo& gpu_info,
                             PriorityHint priority_hint,
                             std::vector<cl_queue_properties>* props);

AdrenoInfo::OpenClCompilerVersion GetQualcommOpenClCompilerVersion(
    const std::string& cl_driver_version);

void FixAdrenoBinary(std::vector<uint8_t>* binary);

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_UTIL_H_
