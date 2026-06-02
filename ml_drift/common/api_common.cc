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
  if (p1_position == p2_position) return PriorityImportance::UNKNOWN;
  return p1_position < p2_position ? PriorityImportance::HIGHER
                                   : PriorityImportance::LOWER;
}

bool IsValid(const InferenceOptions& options) {
  if (options.usage == InferenceUsage::UNKNOWN) {
    return false;
  }
  if (options.priority1 == InferencePriority::UNKNOWN ||
      options.priority2 == InferencePriority::UNKNOWN ||
      options.priority3 == InferencePriority::UNKNOWN) {
    return false;
  }
  if (options.priority1 == InferencePriority::AUTO) {
    return false;
  }
  if (options.priority2 == InferencePriority::AUTO &&
      options.priority3 != InferencePriority::AUTO) {
    return false;
  }
  if (options.priority1 == options.priority2 ||
      options.priority1 == options.priority3) {
    return false;
  }
  if (options.priority2 == options.priority3 &&
      options.priority2 != InferencePriority::AUTO) {
    return false;
  }
  return true;
}

// Implementation note: this resolution logic is shared between GL and CL
// backends, but they might have own logic. Thus, the function is defined
// here just for code re-use purposes.
void ResolveAutoPriority(InferenceOptions* options) {
  // priority1 can not be AUTO as it would make options invalid.
  if (options->priority2 == InferencePriority::AUTO) {
    switch (options->priority1) {
      case InferencePriority::MIN_LATENCY:
        options->priority2 = InferencePriority::MIN_MEMORY_USAGE;
        options->priority3 = InferencePriority::MAX_PRECISION;
        return;
      case InferencePriority::MIN_MEMORY_USAGE:
        options->priority2 = InferencePriority::MAX_PRECISION;
        options->priority3 = InferencePriority::MIN_LATENCY;
        return;
      case InferencePriority::MAX_PRECISION:
        options->priority2 = InferencePriority::MIN_LATENCY;
        options->priority3 = InferencePriority::MIN_MEMORY_USAGE;
        return;
      case InferencePriority::UNKNOWN:
      case InferencePriority::AUTO:
        // Invalid and unreachable option.
        return;
    }
  }

  if (options->priority3 == InferencePriority::AUTO) {
    // Simply add missing priority
    if (GetPosition(*options, InferencePriority::MIN_LATENCY) == 4) {
      options->priority3 = InferencePriority::MIN_LATENCY;
    } else if (GetPosition(*options, InferencePriority::MAX_PRECISION) == 4) {
      options->priority3 = InferencePriority::MAX_PRECISION;
    } else if (GetPosition(*options, InferencePriority::MIN_MEMORY_USAGE) ==
               4) {
      options->priority3 = InferencePriority::MIN_MEMORY_USAGE;
    }
  }
}

}  // namespace ml_drift
