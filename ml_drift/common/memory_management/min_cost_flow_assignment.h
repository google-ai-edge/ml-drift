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

#ifndef ML_DRIFT_COMMON_MEMORY_MANAGEMENT_MIN_COST_FLOW_ASSIGNMENT_H_
#define ML_DRIFT_COMMON_MEMORY_MANAGEMENT_MIN_COST_FLOW_ASSIGNMENT_H_

#include <stddef.h>

#include <vector>

#include "absl/status/status.h"
#include "ml_drift/common/memory_management/types.h"

namespace ml_drift {

// Implements memory management with a Minimum-cost flow matching algorithm.
//
// The problem of memory management is NP-complete. This function creates
// auxiliary flow graph, find minimum-cost flow in it and calculates the
// assignment of shared objects to tensors, using the result of the flow
// algorithm.
absl::Status MinCostFlowAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_MEMORY_MANAGEMENT_MIN_COST_FLOW_ASSIGNMENT_H_
