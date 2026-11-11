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

#ifndef ML_DRIFT_COMMON_TASK_TUNING_TYPE_H_
#define ML_DRIFT_COMMON_TASK_TUNING_TYPE_H_

namespace ml_drift {

enum class TuningType {
  // Benchmarks several work group sizes per kernel and keeps the fastest.
  kExhaustive,
  // Picks one work group size per kernel from the grid size.
  kFast,
  // kFast with grid-independent work group sizes, so identical kernels at
  // different tensor shapes share one (kernel, work group size) pipeline.
  kFastStable,
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_TUNING_TYPE_H_
