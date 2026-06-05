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

#ifndef ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/buffer_desc.h"

namespace ml_drift {

enum class WeightsUploadType {
  kLocalMemoryWorkGroupLoad,
  kLocalMemoryByThreads,
  kWaveMemory,
  kGlobalMemory,
  kConstantMemory,
};

// Returns the weights upload type for fixed size transposed convolution.
WeightsUploadType GetWeightsUploadTypeForFixedSizeConvTransposed(
    const GpuInfo& gpu_info);

// Returns the buffer descriptor for fixed size transposed convolution weights.
BufferDescriptor GetWeightsBufferDescForFixedSizeConvTransposed(
    const GpuInfo& gpu_info, DataType weights_data_type,
    WeightsUploadType weights_upload_type);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_UTIL_H_
