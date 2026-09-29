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

#ifndef ML_DRIFT_COMMON_API_COMMON_H_
#define ML_DRIFT_COMMON_API_COMMON_H_

#include <cstdint>
#include <cstdlib>

#include "absl/types/span.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

// Common abbreviations:
//   B  - batch
//   H  - height
//   W  - width
//   C  - channels
//   D  - depth := DivideRoundUp(C, 4)
//   C4 - is the constant = 4.
enum class DataLayout {
  kUnknown,
  kBHWC,
  kDHWC4,
  kHWDC4,
  kHDWC4,

  // Deprecated aliases:
  UNKNOWN = kUnknown,
  BHWC = kBHWC,
  DHWC4 = kDHWC4,
  HWDC4 = kHWDC4,
  HDWC4 = kHDWC4,
};

struct CpuMemory {
  CpuMemory() = default;
  CpuMemory(void* new_data, size_t new_size_bytes)
      : data(new_data), size_bytes(new_size_bytes) {}

  void* data = nullptr;
  size_t size_bytes = 0;
};

template <typename T>
inline CpuMemory MakeCpuMemory(absl::Span<T> t) {
  CpuMemory m;
  m.data = t.data();
  m.size_bytes = t.size() * sizeof(T);
  return m;
}

template <typename T>
inline CpuMemory MakeReadableCpuMemory(absl::Span<const T> t) {
  CpuMemory m;
  m.data = const_cast<T*>(t.data());
  m.size_bytes = t.size() * sizeof(T);
  return m;
}

struct Dimensions {
  Dimensions() : b(1), h(1), w(1), c(1) {}

  Dimensions(int32_t batch, int32_t height, int32_t width, int32_t channels)
      : b(batch), h(height), w(width), c(channels) {}

  int32_t d() const { return DivideRoundUp(c, 4); }

  int32_t product() const { return b * h * w * c; }

  bool operator==(const Dimensions& other) const {
    return b == other.b && h == other.h && w == other.w && c == other.c;
  }

  int32_t b;
  int32_t h;
  int32_t w;
  int32_t c;
};

// Encapsulated compilation/runtime tradeoffs.
enum class InferenceUsage {
  kUnknown,

  // InferenceRunner will be used only once. Therefore, it is important to
  // minimize bootstrap time as well.
  kFastSingleAnswer,

  // Prefer maximizing the throughput. Same inference runner will be used
  // repeatedly on different inputs.
  kSustainedSpeed,

  // Balance init latency and throughput. This option will result in slightly
  // higher init latency than FAST_SINGLE_ANSWER but should have inference
  // latency closer to SUSTAINED_SPEED.
  kBalanced,

  // Deprecated aliases:
  UNKNOWN = kUnknown,
  FAST_SINGLE_ANSWER = kFastSingleAnswer,
  SUSTAINED_SPEED = kSustainedSpeed,
  BALANCED = kBalanced,
};

// Defines aspects to control while instantiating a runner.
enum class InferencePriority {
  kUnknown,
  kAuto,
  kMinLatency,
  kMaxPrecision,
  kMinMemoryUsage,

  // Deprecated aliases:
  UNKNOWN = kUnknown,
  AUTO = kAuto,
  MIN_LATENCY = kMinLatency,
  MAX_PRECISION = kMaxPrecision,
  MIN_MEMORY_USAGE = kMinMemoryUsage,
};

struct InferenceOptions {
  InferenceUsage usage = InferenceUsage::kSustainedSpeed;

  // Ordered priorities provide better understanding of desired semantics,
  // where priority(n) is more important than priority(n+1).
  // AUTO priority is needed when a single priority is the most important
  // factor. For example, priority1 = InferencePriority::MIN_LATENCY and leaving
  // everything else to AUTO would result in configuration that achieves maximum
  // performance.
  //
  // AUTO priority can only be used when higher priorities are fully specified.
  // For example:
  //   VALID:   priority1 = MIN_LATENCY, priority2 = AUTO, priority3 = AUTO
  //   VALID:   priority1 = MIN_LATENCY, priority2 = MAX_PRECISION,
  //            priority3 = AUTO
  //   INVALID: priority1 = AUTO, priority2 = MIN_LATENCY, priority3 = AUTO
  //   INVALID: priority1 = MIN_LATENCY, priority2 = AUTO,
  //            priority3 = MAX_PRECISION
  // Invalid priorities will result in error.
  InferencePriority priority1 = InferencePriority::kMaxPrecision;
  InferencePriority priority2 = InferencePriority::kAuto;
  InferencePriority priority3 = InferencePriority::kAuto;
};

// Returns a position number for the priority. If priority is missing,
// then it would return 'max num priorities + 1'.
int GetPosition(const InferenceOptions& options, InferencePriority p);

// Return true if options are valid.
bool IsValid(const InferenceOptions& options);

// Resolves AUTO priorities and specifies them explicitly.
// Note, no-one should assume that these mappings will not change.
// Technically this function is declared here for code re-use purposes and
// by no means it should be treated as canonical way to resolve AUTO.
void ResolveAutoPriority(InferenceOptions* options);

enum class PriorityImportance {
  kUnknown,
  kHigher,
  kLower,

  // Deprecated aliases:
  UNKNOWN = kUnknown,
  HIGHER = kHigher,
  LOWER = kLower,
};

// If both p1 and p2 are not present in options, return UNKNOWN
// If p1 is present, but p2 is not, return HIGHER
// If p2 is present, but p1 is not, return LOWER
// If both are present, and p1 is more important, return HIGHER, otherwise,
// LOWER.
PriorityImportance GetRelativeImportance(const InferenceOptions& options,
                                         InferencePriority p1,
                                         InferencePriority p2);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_API_COMMON_H_
