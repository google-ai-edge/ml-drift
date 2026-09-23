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

#ifndef ML_DRIFT_COMMON_KERNELS_SCATTER_ND_H_
#define ML_DRIFT_COMMON_KERNELS_SCATTER_ND_H_

#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

// Creates a GPU operation for the ScatterND operation.
//
// Implements the TFLite `ScatterNd` builtin: the output is zero-initialized and
// updates are scattered into it according to `indices`; duplicate indices are
// summed. Inputs are expected in the order {indices, updates}; the output shape
// is taken from the destination tensor.
//
// Only an index depth (last dimension of `indices`) of 3 is currently
// supported: each index vector addresses the leading three output dimensions
// (mapped to BHWC batch/height/width) and the matching `updates` row supplies
// the channel vector.
GPUOperation CreateScatterNd(const OperationDef& op_def);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SCATTER_ND_H_
