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

#ifndef ML_DRIFT_COMMON_KERNELS_RESHAPEX4_H_
#define ML_DRIFT_COMMON_KERNELS_RESHAPEX4_H_

#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Reshapex4 : public GPUOperation {
 public:
  Reshapex4() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  Reshapex4(Reshapex4&& operation) = default;
  Reshapex4& operator=(Reshapex4&& operation) = default;
  Reshapex4(const Reshapex4&) = delete;
  Reshapex4& operator=(const Reshapex4&) = delete;
};

// Creates a Reshapex4 operation.
// More optimized, but require src_channels % 4 == 0 and dst_channels % 4 == 0
Reshapex4 CreateReshapex4(const OperationDef& definition);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_RESHAPEX4_H_
