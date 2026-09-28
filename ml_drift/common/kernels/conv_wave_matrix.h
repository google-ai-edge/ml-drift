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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_H_

#include <string_view>
#include <variant>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvWaveMatrix : public GPUOperation {
 public:
  struct KernelParams {
    int3 work_group_size;
    int3 work_group_launch_order;
    bool linear_spatial;  // spatial dimensions are Width/Height/Depth
    int dst_slices_per_thread;
    int wave_matrix_m;
    int wave_matrix_n;
    int wave_matrix_k;
    int wave_size;
    bool load_right_transposed = false;
    bool linearized_wg = false;  // hack for WebGPU

    int GetX4SlicesCount() const {
      if (linear_spatial) {
        return work_group_size.y;
      } else {
        return work_group_size.z;
      }
    }
  };
  struct ConvParams {
    WeightsDescription weights_desc;         // only used for ExternalWeights
    OHWI scale_zp_shape = OHWI(1, 1, 1, 1);  // only used for ExternalWeights
    bool has_zero_point = false;             // only used for ExternalWeights
    int src_group_slices = 0;                // only used for ExternalWeights
    CalculationsPrecision precision = CalculationsPrecision::kF32;
    DataType weights_data_type;
    bool x_kernel_is_1 = true;
    bool y_kernel_is_1 = true;
    bool z_kernel_is_1 = true;
    bool softmax_input_activation = false;
    bool different_weights_for_height = false;
    bool has_bias = true;
    ConvRuntimeCheckDesc runtime_check;
    bool Is8Bit() const { return SizeInBitsOf(weights_data_type) == 8; }

    void Init(const Convolution2DAttributes& attr,
              bool different_weight_for_height) {
      different_weights_for_height = different_weight_for_height;
      if (different_weights_for_height) {
        return;
      }
      const auto& weights_shape =
          std::visit([](const auto& w) { return w.shape; }, attr.weights);
      x_kernel_is_1 = weights_shape.w == 1 && attr.strides.w == 1 &&
                      attr.dilations.w == 1 && attr.padding.prepended.w == 0 &&
                      attr.padding.appended.w == 0;
      y_kernel_is_1 = weights_shape.h == 1 && attr.strides.h == 1 &&
                      attr.dilations.h == 1 && attr.padding.prepended.h == 0 &&
                      attr.padding.appended.h == 0;
    }
  };

  ConvWaveMatrix(const OperationDef& definition, const GpuInfo& gpu_info,
                 const ConvParams& conv_params,
                 const KernelParams& kernel_params);
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override;

  std::string_view GetDebugName() const override { return "conv_wave_matrix"; }

  // Move only
  ConvWaveMatrix(ConvWaveMatrix&& kernel) = default;
  ConvWaveMatrix& operator=(ConvWaveMatrix&& kernel) = default;
  ConvWaveMatrix(const ConvWaveMatrix&) = delete;
  ConvWaveMatrix& operator=(const ConvWaveMatrix&) = delete;

  WeightsDescription GetWeightsDescription() const {
    if (params_.weights_desc.layout != WeightsLayout::kUnknown) {
      return params_.weights_desc;
    }
    WeightsDescription desc;
    desc.type = params_.weights_data_type;
    desc.layout = WeightsLayout::kCustomGroups;
    auto& groups = desc.group_sizes;
    if (kernel_params_.load_right_transposed) {
      groups.push_back({Axis::kInputChannels, kernel_params_.wave_matrix_k});
      groups.push_back({Axis::kOutputChannels, kernel_params_.wave_matrix_n});
    } else {
      groups.push_back({Axis::kOutputChannels, kernel_params_.wave_matrix_n});
      groups.push_back({Axis::kInputChannels, kernel_params_.wave_matrix_k});
    }
    groups.push_back(
        {Axis::kOutputChannels, kernel_params_.dst_slices_per_thread /
                                    (kernel_params_.wave_matrix_n / 4)});
    groups.push_back({Axis::kInputChannels, 0});
    groups.push_back({Axis::kWidth, 0});
    groups.push_back({Axis::kHeight, 0});
    groups.push_back({Axis::kOutputChannels, 0});
    return desc;
  }

  ConvParams params_;
  KernelParams kernel_params_;

 private:
  void UploadWeights(const Tensor<OHWI, DataType::kFloat32>& weights);
  void UploadWeights(const Tensor<OHWI, DataType::kInt8>& weights);
  // adding conv params for non 1x1 kernels
  void AddConvParams(const Convolution2DAttributes& attr);

  friend ConvWaveMatrix CreateConvWaveMatrix(
      const OperationDef& definition, CalculationsPrecision precision,
      const BHWC& dst_shape, const Convolution2DAttributes& attr,
      const GpuInfo& gpu_info);
  friend ConvWaveMatrix CreateConvWaveMatrixExternalWeights(
      const OperationDef& definition, CalculationsPrecision precision,
      const BHWC& dst_shape, const Convolution2DAttributes& attr,
      const GpuInfo& gpu_info, const TensorDescriptor* bias,
      const TensorDescriptor* src_exp, bool different_weights_for_height,
      const ConvRuntimeCheckDesc& runtime_check);
  friend ConvWaveMatrix CreateConvWaveMatrixExternalWeights(
      const OperationDef& definition, CalculationsPrecision precision,
      const BHWC& dst_shape, const ExternalWeights& weights,
      const GpuInfo& gpu_info, const TensorDescriptor* bias,
      const TensorDescriptor* src_exp, bool different_weights_for_height,
      const ConvRuntimeCheckDesc& runtime_check);
  friend ConvWaveMatrix CreateConvWaveMatrixInt8(
      const OperationDef& definition, const BHWC& dst_shape,
      const Tensor<OHWI, DataType::kInt8>& weights, const GpuInfo& gpu_info);
  friend ConvWaveMatrix CreateConvWaveMatrixInt8ExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const OHWI& weights_shape, const BHWC& dst_shape);
};

// Checks if the convolution with wave matrix is supported on the given GPU.
bool SupportsConvWaveMatrix(const GpuInfo& gpu_info,
                            CalculationsPrecision precision,
                            const Convolution2DAttributes& attr);

bool SupportsConvWaveMatrix(const GpuInfo& gpu_info,
                            CalculationsPrecision precision,
                            const ExternalWeights& weights);

// Checks if the INT8 convolution with wave matrix is supported on the given
// GPU.
bool SupportsConvWaveMatrixInt8(const GpuInfo& gpu_info,
                                const OHWI& weights_shape);
// Returns the source packed type for the INT8 convolution with wave matrix.
PackedType GetConvWaveMatrixInt8SrcType();

// Checks if the task size is good for Apple convolution SIMD.
bool IsGoodTaskSizeForAppleConvSimd(const BHWC& dst_shape,
                                    const OHWI& weights_shape,
                                    CalculationsPrecision precision,
                                    const GpuInfo& gpu_info);

// Creates a convolution operation with wave matrix.
ConvWaveMatrix CreateConvWaveMatrix(const OperationDef& definition,
                                    CalculationsPrecision precision,
                                    const BHWC& dst_shape,
                                    const Convolution2DAttributes& attr,
                                    const GpuInfo& gpu_info);

// Creates a convolution operation with wave matrix and external weights.
ConvWaveMatrix CreateConvWaveMatrixExternalWeights(
    const OperationDef& definition, CalculationsPrecision precision,
    const BHWC& dst_shape, const Convolution2DAttributes& attr,
    const GpuInfo& gpu_info, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check = {});

// Experimental
ConvWaveMatrix CreateConvWaveMatrixExternalWeights(
    const OperationDef& definition, CalculationsPrecision precision,
    const BHWC& dst_shape, const ExternalWeights& weights,
    const GpuInfo& gpu_info, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check = {});

// Creates an INT8 convolution operation with wave matrix.
ConvWaveMatrix CreateConvWaveMatrixInt8(
    const OperationDef& definition, const BHWC& dst_shape,
    const Tensor<OHWI, DataType::kInt8>& weights, const GpuInfo& gpu_info);

// Creates an INT8 convolution operation with wave matrix and external weights.
ConvWaveMatrix CreateConvWaveMatrixInt8ExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OHWI& weights_shape, const BHWC& dst_shape);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MATRIX_H_
