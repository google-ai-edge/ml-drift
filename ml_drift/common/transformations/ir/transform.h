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

#ifndef ML_DRIFT_COMMON_TRANSFORMATIONS_IR_TRANSFORM_H_
#define ML_DRIFT_COMMON_TRANSFORMATIONS_IR_TRANSFORM_H_

#include "absl/status/status.h"
#include "ml_drift/common/ir_model.h"

namespace ml_drift::ir {

// Applies all standard graph transformations (RemoveNoop, etc.)
// in a single unified pass until convergence.
absl::Status TransformIrModel(::ml_drift::ir::IrModel* ir_model);

}  // namespace ml_drift::ir

#endif  // ML_DRIFT_COMMON_TRANSFORMATIONS_IR_TRANSFORM_H_
