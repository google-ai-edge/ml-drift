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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_MALI_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_MALI_H_

#include <string_view>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvWaveMatrixMali : public GPUOperation {
 public:
  struct ConvParams {
    DataType weights_type;
  };
  struct KernelParams {
    int src_slices_loop_unroll = 1;
    int dst_s4_block_size = 1;  // block in quantities of 4 slices
    int dst_w4_block_size = 1;  // block in quantities of 4 width, non 1 value
                                // can be used only if batch size is 1.
    bool slices_first = false;
  };
  ConvWaveMatrixMali() = default;
  ConvWaveMatrixMali(const GpuInfo& gpu_info, const OperationDef& definition,
                     const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8);
  ConvWaveMatrixMali(const GpuInfo& gpu_info, const OperationDef& definition,
                     const OHWI& weights_shape);
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription weights_desc;
    weights_desc.type = conv_params_.weights_type;
    weights_desc.layout = WeightsLayout::kCustomGroups;
    weights_desc.group_sizes = {
        {Axis::INPUT_CHANNELS, 4},
        {Axis::OUTPUT_CHANNELS, 16},
        {Axis::INPUT_CHANNELS, 4},
        {Axis::OUTPUT_CHANNELS, kernel_params_.dst_s4_block_size},
        {Axis::INPUT_CHANNELS, 0},
        {Axis::WIDTH, 0},
        {Axis::HEIGHT, 0},
        {Axis::OUTPUT_CHANNELS, 0}};
    return weights_desc;
  }

  std::string_view GetDebugName() const override {
    return "conv_wave_matrix_mali";
  }

  // Move only
  ConvWaveMatrixMali(ConvWaveMatrixMali&& kernel) = default;
  ConvWaveMatrixMali& operator=(ConvWaveMatrixMali&& kernel) = default;
  ConvWaveMatrixMali(const ConvWaveMatrixMali&) = delete;
  ConvWaveMatrixMali& operator=(const ConvWaveMatrixMali&) = delete;

 private:
  void InitBase(const GpuInfo& gpu_info, const OperationDef& definition,
                const OHWI& weights_shape);
  ConvParams conv_params_;
  KernelParams kernel_params_;
};

// Checks if the convolution with wave matrix is supported on the given Mali
// GPU.
bool SupportsConvWaveMatrixMaliInt8(const GpuInfo& gpu_info,
                                    const BHWC& src_shape);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_MALI_H_
