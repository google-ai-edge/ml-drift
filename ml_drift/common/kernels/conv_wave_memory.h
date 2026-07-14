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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MEMORY_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MEMORY_H_

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/wave_memory_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvWaveMemory : public GPUOperation {
 public:
  struct KernelParams {
    bool precise_spatial =
        false;  // can not be used with different_weights_for_height
    bool slices_first = false;
    int slices_out = 1;
    int slices_in = 1;
    int wave_size = 32;
    bool slices_loop_first =
        true;  // if false, spatial loops will be first(most internal), (false)
               // supported only in specific combinations.
    bool unroll_x_loop = false;  // applicable with slices_loop_first = false;
    bool img_wave_dot = false;
  };
  struct ConvParams {
    TensorDescriptor src_desc;
    CalculationsPrecision precision = CalculationsPrecision::F32;
    DataType weights_data_type;  // used for weights and biases
    int groups_count = 1;
    bool x_kernel_is_1 = false;
    bool y_kernel_is_1 = false;
    // By default, we have the same weights for WH dims in 2d convolution,
    // but there are cases when we need separate weights for the H dimension
    // and the convolution kernel requires small modifications to support it.
    bool different_weights_for_height = false;
    bool has_bias = true;
    bool softmax_input_activation = false;
    bool blocked_quantization = false;  // for using with int8 convs
    int2 kernel_size = int2(1, 1);
    ConvRuntimeCheckDesc runtime_check;

    bool Is8Bit() const { return weights_data_type == DataType::INT8; }
  };

  ConvWaveMemory() = default;
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = conv_params_.weights_data_type;
    if (kernel_params_.img_wave_dot) {
      if (conv_params_.precision == CalculationsPrecision::F32) {
        if (kernel_params_.slices_loop_first) {
          desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
        } else {
          desc.layout = WeightsLayout::kOISpatialOGroupO4I4;
        }
        desc.output_group_size = kernel_params_.slices_out;
      } else if (conv_params_.precision == CalculationsPrecision::F16) {
        desc.layout = WeightsLayout::kCustomGroups;
        desc.group_sizes = {{Axis::OUTPUT_CHANNELS, 2},
                            {Axis::INPUT_CHANNELS, 4},
                            {Axis::OUTPUT_CHANNELS, 2},
                            {Axis::OUTPUT_CHANNELS, kernel_params_.slices_out}};
        if (kernel_params_.slices_loop_first) {
          desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 0});
          desc.group_sizes.push_back({Axis::WIDTH, 0});
          desc.group_sizes.push_back({Axis::HEIGHT, 0});
        } else {
          desc.group_sizes.push_back({Axis::WIDTH, 0});
          desc.group_sizes.push_back({Axis::HEIGHT, 0});
          desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 0});
        }
        desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 0});
      }
      return desc;
    }
    if (conv_params_.weights_data_type == DataType::INT8) {
      desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
    } else {
      if (kernel_params_.slices_loop_first) {
        desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
      } else {
        desc.layout = WeightsLayout::kOISpatialOGroupI4O4;
      }
    }
    desc.output_group_size = kernel_params_.slices_out;
    return desc;
  }

  std::string_view GetDebugName() const override { return "conv_wave_memory"; }

  // Move only
  ConvWaveMemory(ConvWaveMemory&& kernel) = default;
  ConvWaveMemory& operator=(ConvWaveMemory&& kernel) = default;
  ConvWaveMemory(const ConvWaveMemory&) = delete;
  ConvWaveMemory& operator=(const ConvWaveMemory&) = delete;

 private:
  friend ConvWaveMemory CreateConvWaveMemory(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision, const Convolution2DAttributes& attr,
      const BHWC* dst_shape);
  friend ConvWaveMemory CreateConvWaveMemory(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision, const FullyConnectedAttributes& attr,
      const BHWC* dst_shape);
  friend ConvWaveMemory CreateConvWaveMemoryExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision, const Convolution2DAttributes& attr,
      const TensorDescriptor* bias, const BHWC* dst_shape,
      const TensorDescriptor* src_exp, bool different_weights_for_height,
      const ConvRuntimeCheckDesc& runtime_check);
  friend ConvWaveMemory CreateConvWaveMemoryBatchedMatMul(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const OHWI& weights_shape, const BHWC* dst_shape,
      const TensorDescriptor* src_exp);
  friend ConvWaveMemory CreateConvWaveMemoryInt8(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Tensor<OHWI, DataType::INT8>& weights, const BHWC* dst_shape);
  friend ConvWaveMemory CreateConvWaveMemoryInt8ExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const OHWI& weights_shape, const BHWC* dst_shape);
  friend ConvWaveMemory CreateConvWaveMemoryInt8Grouped(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Tensor<OHWI, DataType::INT8>& weights, int group_size,
      const TensorDescriptor& src_params, const BHWC* dst_shape);

  ConvWaveMemory(const ConvParams& conv_params, const GpuInfo& gpu_info,
                 const OHWI& weights_shape, const BHWC* dst_shape);

  template <DataType T>
  void UploadWeights(const GpuInfo& gpu_info, const Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadBias(const GpuInfo& gpu_info, const Tensor<Linear, T>& bias);

  void GenerateCode(const GpuInfo& gpu_info, const OperationDef& definition,
                    const int2& stride = int2(1, 1),
                    const int2& padding = int2(0, 0),
                    const int2& dilation = int2(1, 1));

  ConvParams conv_params_;
  KernelParams kernel_params_;
};

template <DataType T>
void ConvWaveMemory::UploadWeights(const GpuInfo& gpu_info,
                                   const Tensor<OHWI, T>& weights) {
  const WeightsDescription weights_desc = GetWeightsDescription();
  BufferDescriptor buffer_desc =
      GetBufferDescForWaveMemoryUpload(gpu_info, weights_desc, weights.shape);
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(buffer_desc.data));
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

template <DataType T>
void ConvWaveMemory::UploadBias(const GpuInfo& gpu_info,
                                const Tensor<Linear, T>& bias) {
  if (!conv_params_.has_bias) return;
  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, conv_params_.weights_data_type, bias);
  args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                std::move(bias_tensor_desc)));
}

// Checks if the convolution with wave memory is supported on the given GPU.
bool IsConvWaveMemorySupported(const GpuInfo& gpu_info);

// Creates a convolution operation with wave memory.
ConvWaveMemory CreateConvWaveMemory(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    CalculationsPrecision precision,
                                    const Convolution2DAttributes& attr,
                                    const BHWC* dst_shape = nullptr);

// Creates a convolution operation with wave memory for fully connected layers.
ConvWaveMemory CreateConvWaveMemory(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    CalculationsPrecision precision,
                                    const FullyConnectedAttributes& attr,
                                    const BHWC* dst_shape = nullptr);

// Creates a convolution operation with wave memory and external weights.
ConvWaveMemory CreateConvWaveMemoryExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const Convolution2DAttributes& attr,
    const TensorDescriptor* bias, const BHWC* dst_shape,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check = {});

// Checks if the INT8 convolution with wave memory is supported on the given
// GPU.
bool SupportsConvWaveMemoryInt8(const GpuInfo& gpu_info);
// Returns the source packed type for the INT8 convolution with wave memory.
PackedType GetConvWaveMemoryInt8SrcType(const GpuInfo& gpu_info,
                                        const BHWC& src_shape);

// Creates an INT8 convolution operation with wave memory.
ConvWaveMemory CreateConvWaveMemoryInt8(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Tensor<OHWI, DataType::INT8>& weights,
    const BHWC* dst_shape = nullptr);

// Creates an INT8 convolution operation with wave memory and external weights.
ConvWaveMemory CreateConvWaveMemoryInt8ExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OHWI& weights_shape, const BHWC* dst_shape = nullptr);

// experimental, do not use
// Creates an INT8 grouped convolution operation with wave memory.
ConvWaveMemory CreateConvWaveMemoryInt8Grouped(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Tensor<OHWI, DataType::INT8>& weights, int group_size,
    const TensorDescriptor& src_params, const BHWC* dst_shape = nullptr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_WAVE_MEMORY_H_
