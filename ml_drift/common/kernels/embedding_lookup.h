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

#ifndef ML_DRIFT_COMMON_KERNELS_EMBEDDING_LOOKUP_H_
#define ML_DRIFT_COMMON_KERNELS_EMBEDDING_LOOKUP_H_

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"

namespace ml_drift {

// Creates an embedding lookup operation.
GPUOperation CreateEmbeddingLookup(const OperationDef& op_def,
                                   const GpuInfo& gpu_info,
                                   const EmbeddingLookupAttributes& attr);

// Creates an embedding lookup operation with external weights.
GPUOperation CreateEmbeddingLookupExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const TensorDescriptor& weights, const WeightsDescription& weights_desc,
    const OHWI& weights_shape, const TensorDescriptor* weights_scale,
    const TensorDescriptor* weights_zero_point, Axis lookup_axis);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_EMBEDDING_LOOKUP_H_
