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

#ifndef ML_DRIFT_COMMON_SELECTORS_SPECIAL_SELECTOR_H_
#define ML_DRIFT_COMMON_SELECTORS_SPECIAL_SELECTOR_H_

#include <set>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"

namespace ml_drift {

absl::Status GPUSubgraphFromGraph(
    const ModelHints& hints, const GpuInfo& gpu_info, const GraphFloat32& graph,
    NodeId first_node_id, const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder);

absl::Status GPUSubgraphFromIrModel(
    const ModelHints& hints, const GpuInfo& gpu_info,
    const ir::IrModel& ir_model, ir::IrOpId first_op_id,
    const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_SELECTORS_SPECIAL_SELECTOR_H_
