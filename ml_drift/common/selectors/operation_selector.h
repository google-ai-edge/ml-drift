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

#ifndef ML_DRIFT_COMMON_SELECTORS_OPERATION_SELECTOR_H_
#define ML_DRIFT_COMMON_SELECTORS_OPERATION_SELECTOR_H_

#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

absl::Status GPUOperationFromNode(const GpuInfo& gpu_info,
                                  const OperationDef& op_def,
                                  const CreateGpuModelInfo& create_info,
                                  const std::vector<Value*>& inputs,
                                  const std::vector<Value*>& outputs,
                                  const Node& node,
                                  GpuModelBuilder* model_builder);

// Converts an IR operation to a GPU operation and adds it to the model builder.
absl::Status GPUOperationFromNode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    const CreateGpuModelInfo& create_info,
    const std::vector<const ir::IrTensor*>& inputs,
    const std::vector<const ir::IrTensor*>& outputs, const ir::IrOp& node,
    GpuModelBuilder* model_builder);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_SELECTORS_OPERATION_SELECTOR_H_
