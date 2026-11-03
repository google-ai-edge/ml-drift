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

#include <string_view>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvWaveMatrixMaliF16 : public GPUOperation {
 public:
  struct ConvParams {
    bool x_kernel_is_1 = true;
    bool y_kernel_is_1 = true;
    bool has_bias = true;
    OHWI weights_shape;

    void InitKernelXY(const Convolution2DAttributes& attr) {
      const OHWI& weights_shape = attr.GetWeightsShape();
      x_kernel_is_1 = weights_shape.w == 1 && attr.strides.w == 1 &&
                      attr.dilations.w == 1 && attr.padding.prepended.w == 0 &&
                      attr.padding.appended.w == 0;
      y_kernel_is_1 = weights_shape.h == 1 && attr.strides.h == 1 &&
                      attr.dilations.h == 1 && attr.padding.prepended.h == 0 &&
                      attr.padding.appended.h == 0;
    }
  };

  struct KernelParams {
    int x_block_size = 1;
    int s_block_size = 1;
    bool slices_first = false;
  };

  ConvWaveMatrixMaliF16() = default;
  ConvWaveMatrixMaliF16(const GpuInfo& gpu_info, const ConvParams& conv_params,
                        const TensorDescriptor& src_desc,
                        const TensorDescriptor& dst_desc);

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription weights_desc;
    weights_desc.type = DataType::kFloat16;
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI2I2gr0O8I2gr1O2;
    weights_desc.output_group_size = kernel_params_.s_block_size;
    return weights_desc;
  }

  std::string_view GetDebugName() const override {
    return "conv_wave_matrix_mali_f16";
  }

  // Move only
  ConvWaveMatrixMaliF16(ConvWaveMatrixMaliF16&& kernel) = default;
  ConvWaveMatrixMaliF16& operator=(ConvWaveMatrixMaliF16&& kernel) = default;
  ConvWaveMatrixMaliF16(const ConvWaveMatrixMaliF16&) = delete;
  ConvWaveMatrixMaliF16& operator=(const ConvWaveMatrixMaliF16&) = delete;

  void AddWeights(const Tensor<OHWI, DataType::kFloat32>& weights);

 private:
  ConvParams conv_params_;
  KernelParams kernel_params_;
};

bool SupportsConvWaveMatrixMaliF16(const GpuInfo& gpu_info);

ConvWaveMatrixMaliF16 CreateConvWaveMatrixMaliF16(
    const GpuInfo& gpu_info, const TensorDescriptor& src_desc,
    const TensorDescriptor& dst_desc, const Convolution2DAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_MALI_F16_H_
