// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_COMMON_KERNELS_SOFTMAX1X1_H_
#define ML_DRIFT_COMMON_KERNELS_SOFTMAX1X1_H_

#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Softmax1x1 : public GPUOperation {
 public:
  Softmax1x1() = default;
  Softmax1x1(const OperationDef& definition, const GpuInfo& gpu_info,
             const BHWC& shape, bool reduce_only,
             const SoftmaxRuntimeCheckDesc& runtime_check = {});
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  int3 GetGridSize() const override;

  // Move only
  Softmax1x1(Softmax1x1&& kernel) = default;
  Softmax1x1& operator=(Softmax1x1&& kernel) = default;
  Softmax1x1(const Softmax1x1&) = delete;
  Softmax1x1& operator=(const Softmax1x1&) = delete;

  friend Softmax1x1 CreateSoftmax1x1();

 private:
  std::string GetSoftmaxKernelCode(const OperationDef& op_def,
                                   const GpuInfo& gpu_info);
  bool reduce_only_ = false;
  SoftmaxRuntimeCheckDesc runtime_check_ = {};
};

// Creates a Softmax1x1 operation.
Softmax1x1 CreateSoftmax1x1(const OperationDef& definition,
                            const GpuInfo& gpu_info, const BHWC& shape,
                            const SoftmaxRuntimeCheckDesc& runtime_check = {});

// Creates a Softmax1x1 reduce operation.
Softmax1x1 CreateSoftmax1x1Reduce(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    const SoftmaxRuntimeCheckDesc& runtime_check = {});

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SOFTMAX1X1_H_
