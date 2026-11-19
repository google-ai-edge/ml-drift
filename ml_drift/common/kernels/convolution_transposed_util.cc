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

#include "ml_drift/common/kernels/convolution_transposed_util.h"

#include <string>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/wave_memory_util.h"

namespace ml_drift {
WeightsUploadType GetWeightsUploadTypeForFixedSizeConvTransposed(
    const GpuInfo& gpu_info) {
  WeightsUploadType weights_upload_type = WeightsUploadType::kGlobalMemory;
  if (gpu_info.IsAdreno() && gpu_info.SupportsExtension("ucl_wave_memory") &&
      gpu_info.IsApiOpenCl()) {
    weights_upload_type = WeightsUploadType::kWaveMemory;
  } else if (gpu_info.IsIntel()) {
    if (gpu_info.SupportsExtension("ucl_wave_memory") &&
        gpu_info.SupportsSubGroupWithSize(32) && gpu_info.IsApiOpenCl()) {
      weights_upload_type = WeightsUploadType::kWaveMemory;
    } else {
      weights_upload_type = WeightsUploadType::kLocalMemoryByThreads;
    }
  } else if (gpu_info.IsApple()) {
    if (gpu_info.apple_info.IsBionic()) {
      weights_upload_type = WeightsUploadType::kGlobalMemory;
    } else {
      weights_upload_type = WeightsUploadType::kLocalMemoryByThreads;
    }
  } else if (gpu_info.IsPowerVR()) {
    if (gpu_info.SupportsExtension("ucl_wave_memory") &&
        gpu_info.IsApiOpenCl()) {
      weights_upload_type = WeightsUploadType::kWaveMemory;
    } else {
      if (gpu_info.IsApiOpenCl()) {
        weights_upload_type = WeightsUploadType::kLocalMemoryWorkGroupLoad;
      } else {
        weights_upload_type = WeightsUploadType::kLocalMemoryByThreads;
      }
    }
  } else if (gpu_info.IsNvidia()) {
    weights_upload_type = WeightsUploadType::kLocalMemoryByThreads;
  } else if (gpu_info.IsAMD()) {
    weights_upload_type = WeightsUploadType::kConstantMemory;
  } else {
    weights_upload_type = WeightsUploadType::kGlobalMemory;
  }
  if (!gpu_info.SupportsPointersInKernels()) {
    weights_upload_type = WeightsUploadType::kLocalMemoryByThreads;
  }
  return weights_upload_type;
}

BufferDescriptor GetWeightsBufferDescForFixedSizeConvTransposed(
    const GpuInfo& gpu_info, DataType weights_data_type,
    WeightsUploadType weights_upload_type) {
  BufferDescriptor desc;
  desc.element_type = weights_data_type;
  desc.element_size = 4;
  desc.memory_type = weights_upload_type == WeightsUploadType::kConstantMemory
                         ? MemoryType::kConstant
                         : MemoryType::kGlobal;
  if (weights_upload_type == WeightsUploadType::kWaveMemory) {
    desc = GetBufferDescForWaveMemoryUpload(gpu_info, weights_data_type);
  }
  return desc;
}

}  // namespace ml_drift
