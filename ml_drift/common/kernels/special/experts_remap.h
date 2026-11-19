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

#ifndef ML_DRIFT_COMMON_KERNELS_SPECIAL_EXPERTS_REMAP_H_
#define ML_DRIFT_COMMON_KERNELS_SPECIAL_EXPERTS_REMAP_H_

#include <memory>

#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

// input: active expert ids: seq_size(W) x num_active_experts(C)
// outputs:
//   remap: num_experts(H) x seq_size(W) x 2(C)
//   count: num_experts
std::unique_ptr<GPUOperation> CreateExpertsRemapOp(const TensorDescriptor& src,
                                                   const TensorDescriptor& dst);

std::unique_ptr<GPUOperation> CreateOffsetsOp();

std::unique_ptr<GPUOperation> CreateLinearizeMapOp(
    const TensorDescriptor& src_map, const TensorDescriptor& dst_map);

std::unique_ptr<GPUOperation> CreateExpertsRemapToOp(
    const TensorDescriptor& src, const TensorDescriptor& experts_packed_map,
    const TensorDescriptor& dst);

std::unique_ptr<GPUOperation> CreateExpertsRemapFromOp(
    const TensorDescriptor& src, const TensorDescriptor& experts_packed_map,
    const TensorDescriptor& dst);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPECIAL_EXPERTS_REMAP_H_
