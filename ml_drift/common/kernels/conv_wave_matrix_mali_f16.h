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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_MALI_F16_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_MALI_F16_H_

#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvWaveMatrixMaliF16 : public GPUOperation {
 public:
  ConvWaveMatrixMaliF16() = default;
  ConvWaveMatrixMaliF16(
      const TensorDescriptor& src_desc, const TensorDescriptor& dst_desc,
      const ml_drift::Tensor<OHWI, DataType::kFloat32>& weights);

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  int3 GetGridSize() const override;

  // Move only
  ConvWaveMatrixMaliF16(ConvWaveMatrixMaliF16&& kernel) = default;
  ConvWaveMatrixMaliF16& operator=(ConvWaveMatrixMaliF16&& kernel) = default;
  ConvWaveMatrixMaliF16(const ConvWaveMatrixMaliF16&) = delete;
  ConvWaveMatrixMaliF16& operator=(const ConvWaveMatrixMaliF16&) = delete;

 private:
  int x_block_size_ = 1;
  int s_block_size_ = 1;
  bool slices_first_ = false;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_MALI_F16_H_
