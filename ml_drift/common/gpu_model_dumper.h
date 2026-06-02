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

#ifndef ML_DRIFT_COMMON_GPU_MODEL_DUMPER_H_
#define ML_DRIFT_COMMON_GPU_MODEL_DUMPER_H_

#include <string>

#include "ml_drift/common/gpu_model.h"

namespace ml_drift {

// Interface for dumping GpuModel contents.
class GpuModelDumper {
 public:
  virtual ~GpuModelDumper() = default;
  virtual void Dump(const GpuModel& model) const = 0;
};

// Dumps the contents of the GpuModel using the provided dumper.
void DumpGpuModel(const GpuModel& model, const GpuModelDumper& dumper);

// Dumps the contents of the GpuModel using the DefaultGpuModelDumper.
void DumpGpuModel(const GpuModel& model);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_MODEL_DUMPER_H_
