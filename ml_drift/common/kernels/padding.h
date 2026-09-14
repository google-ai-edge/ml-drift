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

#ifndef ML_DRIFT_COMMON_KERNELS_PADDING_H_
#define ML_DRIFT_COMMON_KERNELS_PADDING_H_

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

// Creates a GPU operation for the padding operation.
//
// `src_channels` is the channel count of the source tensor. It has to be passed
// in because the tensor descriptors in `definition` are not required to carry a
// shape at task creation time.
GPUOperation CreatePadding(const GpuInfo& gpu_info,
                           const OperationDef& definition,
                           const PadAttributes& attr, int src_channels);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_PADDING_H_
