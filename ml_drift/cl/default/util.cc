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

#include <cstdint>
#include <vector>

#include "ml_drift/common/gpu_info.h"

namespace ml_drift {
namespace cl {

bool AdrenoSupportsWaveMemory(const GpuInfo& gpu_info) {
  // The Adreno wave-memory path is not available in this build.
  return false;
}

void FixAdrenoBinary(std::vector<uint8_t>* binary) {
  // No-op.
}

}  // namespace cl
}  // namespace ml_drift
