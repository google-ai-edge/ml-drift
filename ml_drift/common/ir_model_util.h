#include <vector>

#include "absl/container/flat_hash_map.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
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

#ifndef ML_DRIFT_COMMON_IR_MODEL_UTIL_H_
#define ML_DRIFT_COMMON_IR_MODEL_UTIL_H_

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/status.h"

namespace ml_drift::ir {

absl::Status IrModelToGpuModel(const ir::IrModel& ir_model,
                               const CreateGpuModelInfo& create_info,
                               const GpuInfo& gpu_info, GpuModel* gpu_model);

absl::Status IrModelToGpuModelWithWeightsConversion(
    const ir::IrModel& ir_model, const CreateGpuModelInfo& create_info,
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    GpuModel* gpu_weights_conversion_model,
    absl::flat_hash_map<ValueId, ValueId>* weights_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_info);

}  // namespace ml_drift::ir

#endif  // ML_DRIFT_COMMON_IR_MODEL_UTIL_H_
