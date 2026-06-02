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

#ifndef ML_DRIFT_COMMON_KERNELS_SPLIT_H_
#define ML_DRIFT_COMMON_KERNELS_SPLIT_H_

#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Split : public GPUOperation {
 public:
  Split(const GpuInfo& gpu_info, const OperationDef& definition,
        const SplitAttributes& attr, const std::vector<int>& channels);
  int3 GetGridSize() const override;

  // Move only
  Split(Split&& operation) = default;
  Split& operator=(Split&& operation) = default;
  Split(const Split&) = delete;
  Split& operator=(const Split&) = delete;

 private:
  std::string GetSplitCode(const OperationDef& definition);
  std::string GetSplitChannelsCode(const GpuInfo& gpu_info,
                                   const OperationDef& definition,
                                   const std::vector<int>& channels);

  SplitAttributes attr_;
};

// Creates a Split operation.
Split CreateSplit(const GpuInfo& gpu_info, const OperationDef& definition,
                  const SplitAttributes& attr,
                  const std::vector<int>& channels);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPLIT_H_
