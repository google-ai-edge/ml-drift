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

#ifndef ML_DRIFT_COMMON_KERNELS_TOP_K_H_
#define ML_DRIFT_COMMON_KERNELS_TOP_K_H_

#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

class TopKOp : public GPUOperation {
 public:
  explicit TopKOp(const int3& wg_size, bool wg_reduction) {
    wg_reduction_ = wg_reduction;
    work_group_size_ = wg_size;
  }
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override {
    if (wg_reduction_) {
      return int3(work_group_size_.x * dst_[0]->Width(), dst_[0]->Height(), 1);
    } else {
      return int3(dst_[0]->Width(), dst_[0]->Height(), 1);
    }
  }
  absl::Status BindArguments(ArgumentsBinder* args) override {
    const int wg_count =
        wg_reduction_ ? dst_[0]->Width()
                      : DivideRoundUp(dst_[0]->Width(), work_group_size_.x);
    const int reduction_per_group = DivideRoundUp(src_[0]->Width(), wg_count);
    RETURN_IF_ERROR(args->SetInt("reduction_per_group", reduction_per_group));
    return absl::OkStatus();
  }

  // Move only
  TopKOp(TopKOp&& operation) = default;
  TopKOp& operator=(TopKOp&& operation) = default;
  TopKOp(const TopKOp&) = delete;
  TopKOp& operator=(const TopKOp&) = delete;

  bool wg_reduction_ = true;
};

// Creates a TopK operation.
TopKOp CreateTopK(const GpuInfo& gpu_info, const OperationDef& op_def,
                  bool use_wg_reduction = true, int k_offset = 0);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TOP_K_H_
