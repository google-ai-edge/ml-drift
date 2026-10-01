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

#ifndef ML_DRIFT_COMMON_GPU_MODEL_BUILDER_MOE_UTIL_H_
#define ML_DRIFT_COMMON_GPU_MODEL_BUILDER_MOE_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"
#include "ml_drift/common/gpu_model_builder.h"

namespace ml_drift {

std::vector<GpuModelBuilder::TensorHandle> CreateExpertsRemap(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& indices,
    int num_experts);

GpuModelBuilder::TensorHandle ExpertsRemapTo(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& experts_remap, int num_active_experts);

GpuModelBuilder::TensorHandle ExpertsRemapFrom(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& experts_remap, int num_active_experts);

absl::StatusOr<GpuModelBuilder::TensorHandle> MakeConvWithPackedGroups(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& params,
    const GpuModelBuilder::Weights& weights, int num_active_experts);

absl::StatusOr<GpuModelBuilder::TensorHandle> MakeConvWithBatchIds(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& ids,
    const GpuModelBuilder::Weights& weights);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_MODEL_BUILDER_MOE_UTIL_H_
