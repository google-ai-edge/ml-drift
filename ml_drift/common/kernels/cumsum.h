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

#ifndef ML_DRIFT_COMMON_KERNELS_CUMSUM_H_
#define ML_DRIFT_COMMON_KERNELS_CUMSUM_H_

#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Cumsum : public GPUOperation {
 public:
  Cumsum() = default;
  explicit Cumsum(Axis axis) : axis_(axis) {}
  int3 GetGridSize() const override;

  // Move only
  Cumsum(Cumsum&& operation);
  Cumsum& operator=(Cumsum&& operation);
  Cumsum(const Cumsum&) = delete;
  Cumsum& operator=(const Cumsum&) = delete;
  void GetCumsumCode(const OperationDef& op_def);

 private:
  Axis axis_;
};

// Creates a cumulative sum operation.
Cumsum CreateCumsum(const OperationDef& definition,
                    const CumsumAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CUMSUM_H_
