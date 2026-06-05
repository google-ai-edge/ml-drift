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

#ifndef ML_DRIFT_COMMON_KERNELS_SOFTMAX_H_
#define ML_DRIFT_COMMON_KERNELS_SOFTMAX_H_

#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Softmax : public GPUOperation {
 public:
  Softmax() = default;
  Softmax(const OperationDef& definition, const GpuInfo& gpu_info,
          const BHWC& shape, bool reduce_only,
          const SoftmaxRuntimeCheckDesc& runtime_check = {});

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  Softmax(Softmax&& kernel) = default;
  Softmax& operator=(Softmax&& kernel) = default;
  Softmax(const Softmax&) = delete;
  Softmax& operator=(const Softmax&) = delete;

 private:
  bool use_wg_reduction_ = false;
};

// Creates a Softmax operation.
Softmax CreateSoftmax(const OperationDef& definition, const GpuInfo& gpu_info,
                      const BHWC& shape,
                      const SoftmaxRuntimeCheckDesc& runtime_check = {});

// Creates a Softmax reduce operation.
// Softmax can be done in 2 steps
// 1) Reduction operation that calculates exp/max as separate tensor.
//    This tensor needs only 2 channels.
// 2) Elementwise operation that uses src tensor and exp/max tensor from 1)
Softmax CreateSoftmaxReduce(const OperationDef& definition,
                            const GpuInfo& gpu_info, const BHWC& shape,
                            const SoftmaxRuntimeCheckDesc& runtime_check = {});

// Creates a Softmax final operation.
// Elementwise operation that uses src tensor and exp/max tensor from
// SoftmaxReduce
GPUOperation CreateSoftmaxFinal(const OperationDef& definition);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SOFTMAX_H_
