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

#ifndef ML_DRIFT_COMMON_TASK_WAVE_MEMORY_UTIL_H_
#define ML_DRIFT_COMMON_TASK_WAVE_MEMORY_UTIL_H_

#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/weights_layout.h"

namespace ml_drift {

BufferDescriptor GetBufferDescForWaveMemoryUpload(const GpuInfo& gpu_info,
                                                  DataType data_type);
BufferDescriptor GetBufferDescForWaveMemoryUpload(
    const GpuInfo& gpu_info, const WeightsDescription& weights_desc);
// this version will allocate memory for the buffer using the weights_shape
BufferDescriptor GetBufferDescForWaveMemoryUpload(
    const GpuInfo& gpu_info, const WeightsDescription& weights_desc,
    const OHWI& weights_shape);

absl::Status ResolveWaveMemory(const GpuInfo& gpu_info, std::string* code,
                               Arguments* args,
                               std::vector<CompilerOptions>* compiler_options);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_WAVE_MEMORY_UTIL_H_
