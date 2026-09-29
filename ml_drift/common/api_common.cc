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

#include "ml_drift/common/api_common.h"

namespace ml_drift {

int GetPosition(const InferenceOptions& options, InferencePriority p) {
  if (options.priority1 == p) return 1;
  if (options.priority2 == p) return 2;
  if (options.priority3 == p) return 3;
  return 4;  // least important
}

PriorityImportance GetRelativeImportance(const InferenceOptions& options,
                                         InferencePriority p1,
                                         InferencePriority p2) {
  int p1_position = GetPosition(options, p1);
  int p2_position = GetPosition(options, p2);
  if (p1_position == p2_position) return PriorityImportance::kUnknown;
  return p1_position < p2_position ? PriorityImportance::kHigher
                                   : PriorityImportance::kLower;
}

bool IsValid(const InferenceOptions& options) {
  if (options.usage == InferenceUsage::kUnknown) {
    return false;
  }
  if (options.priority1 == InferencePriority::kUnknown ||
      options.priority2 == InferencePriority::kUnknown ||
      options.priority3 == InferencePriority::kUnknown) {
    return false;
  }
  if (options.priority1 == InferencePriority::kAuto) {
    return false;
  }
  if (options.priority2 == InferencePriority::kAuto &&
      options.priority3 != InferencePriority::kAuto) {
    return false;
  }
  if (options.priority1 == options.priority2 ||
      options.priority1 == options.priority3) {
    return false;
  }
  if (options.priority2 == options.priority3 &&
      options.priority2 != InferencePriority::kAuto) {
    return false;
  }
  return true;
}

// Implementation note: this resolution logic is shared between GL and CL
// backends, but they might have own logic. Thus, the function is defined
// here just for code re-use purposes.
void ResolveAutoPriority(InferenceOptions* options) {
  // priority1 can not be AUTO as it would make options invalid.
  if (options->priority2 == InferencePriority::kAuto) {
    switch (options->priority1) {
      case InferencePriority::kMinLatency:
        options->priority2 = InferencePriority::kMinMemoryUsage;
        options->priority3 = InferencePriority::kMaxPrecision;
        return;
      case InferencePriority::kMinMemoryUsage:
        options->priority2 = InferencePriority::kMaxPrecision;
        options->priority3 = InferencePriority::kMinLatency;
        return;
      case InferencePriority::kMaxPrecision:
        options->priority2 = InferencePriority::kMinLatency;
        options->priority3 = InferencePriority::kMinMemoryUsage;
        return;
      case InferencePriority::kUnknown:
      case InferencePriority::kAuto:
        // Invalid and unreachable option.
        return;
    }
  }

  if (options->priority3 == InferencePriority::kAuto) {
    // Simply add missing priority
    if (GetPosition(*options, InferencePriority::kMinLatency) == 4) {
      options->priority3 = InferencePriority::kMinLatency;
    } else if (GetPosition(*options, InferencePriority::kMaxPrecision) == 4) {
      options->priority3 = InferencePriority::kMaxPrecision;
    } else if (GetPosition(*options, InferencePriority::kMinMemoryUsage) == 4) {
      options->priority3 = InferencePriority::kMinMemoryUsage;
    }
  }
}

}  // namespace ml_drift
