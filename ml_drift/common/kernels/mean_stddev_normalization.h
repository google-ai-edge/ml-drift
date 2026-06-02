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

#ifndef ML_DRIFT_COMMON_KERNELS_MEAN_STDDEV_NORMALIZATION_H_
#define ML_DRIFT_COMMON_KERNELS_MEAN_STDDEV_NORMALIZATION_H_

#include <memory>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

// Implements tensor_utils::MeanStddevNormalization
class MeanStdDevNormalization : public GPUOperation {
 public:
  MeanStdDevNormalization() = default;
  explicit MeanStdDevNormalization(
      const OperationDef& definition, const GpuInfo& gpu_info,
      const BHWC& shape, float variance_bias, bool two_step,
      const Tensor<Linear, DataType::FLOAT32>* gamma,
      const Tensor<Linear, DataType::FLOAT32>* beta);

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    if (!work_group_reduction_) {
      return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                   grid_size_);
    }
    return {work_group_size_};
  }
  int3 GetGridSize() const override;

  // Move only
  MeanStdDevNormalization(MeanStdDevNormalization&& kernel) = default;
  MeanStdDevNormalization& operator=(MeanStdDevNormalization&& kernel) =
      default;
  MeanStdDevNormalization(const MeanStdDevNormalization&) = delete;
  MeanStdDevNormalization& operator=(const MeanStdDevNormalization&) = delete;

 private:
  friend MeanStdDevNormalization CreateRMSNormalization(
      const OperationDef& definition, const GpuInfo& gpu_info,
      const BHWC& shape, float variance_bias);
  friend std::unique_ptr<GPUOperation> CreateStatisticalTopK(
      const OperationDef& definition, const GpuInfo& gpu_info,
      const BHWC& shape, float stddev_multiplier);
  std::string GetNormalizationCode(const GpuInfo& gpu_info, bool has_batch,
                                   bool channels_x4, bool two_step, bool gamma,
                                   bool beta, bool has_depth);
  bool work_group_reduction_ = true;
  bool has_depth_ = false;
};

// Creates a mean and standard deviation normalization operation.
// std dev can be calculated in single step, but two step algorithm can
// provide more stable and robust results
MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float variance_bias = 1.0e-8f, bool two_step = true);

// Creates a mean and standard deviation normalization operation with gamma and
// beta tensors.
MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float variance_bias, const Tensor<Linear, DataType::FLOAT32>& gamma,
    const Tensor<Linear, DataType::FLOAT32>& beta, bool two_step);

// Creates a root mean square normalization operation.
MeanStdDevNormalization CreateRMSNormalization(const OperationDef& definition,
                                               const GpuInfo& gpu_info,
                                               const BHWC& shape,
                                               float variance_bias = 1.0e-8f);

// Creates a statistical TopK operation.
std::unique_ptr<GPUOperation> CreateStatisticalTopK(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float stddev_multiplier);

class HWCGroupNormalization : public GPUOperation {
 public:
  explicit HWCGroupNormalization(const OperationDef& definition,
                                 const GpuInfo& gpu_info, const BHWC& shape,
                                 int groups, float variance_bias,
                                 const Tensor<Linear, DataType::FLOAT32>& gamma,
                                 const Tensor<Linear, DataType::FLOAT32>& beta);

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override;

  // Move only
  HWCGroupNormalization(HWCGroupNormalization&& kernel) = default;
  HWCGroupNormalization& operator=(HWCGroupNormalization&& kernel) = default;
  HWCGroupNormalization(const HWCGroupNormalization&) = delete;
  HWCGroupNormalization& operator=(const HWCGroupNormalization&) = delete;

 private:
  int groups_ = 1;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_MEAN_STDDEV_NORMALIZATION_H_
