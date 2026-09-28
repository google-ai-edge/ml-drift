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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_APPLE_MPP_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_APPLE_MPP_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

// Convolution implementation for Apple GPUs with MetalPerformancePrimitives.
class ConvAppleMPP : public GPUOperation {
 public:
  struct ConvParams {
    WeightsDescription weights_desc;         // only used for ExternalWeights
    OHWI scale_zp_shape = OHWI(1, 1, 1, 1);  // only used for ExternalWeights
    bool has_zero_point = false;             // only used for ExternalWeights
    int src_group_slices = 0;                // only used for ExternalWeights
    DataType weights_data_type;
    bool x_kernel_is_1 = true;
    bool y_kernel_is_1 = true;
    OHWI weights_shape;
    bool softmax_input_activation = false;
    bool batched_weights = false;
    bool has_bias = true;
    ConvRuntimeCheckDesc runtime_check;

    void InitKernelXY(const Convolution2DAttributes& attr) {
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

  ConvAppleMPP(const TensorDescriptor& src, const ConvParams& params,
               int m_tile = 64, int n_tile = 128);
  // Move only
  ConvAppleMPP(ConvAppleMPP&& kernel) = default;
  ConvAppleMPP& operator=(ConvAppleMPP&& kernel) = default;
  ConvAppleMPP(const ConvAppleMPP&) = delete;
  ConvAppleMPP& operator=(const ConvAppleMPP&) = delete;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
    if (params_.weights_desc.layout != WeightsLayout::kUnknown) {
      return params_.weights_desc;
    }
    WeightsDescription weights_desc;
    weights_desc.type = params_.weights_data_type;
    weights_desc.layout = WeightsLayout::kCustomGroups;
    weights_desc.group_sizes = {
        {Axis::kOutputChannels, AlignByN(params_.weights_shape.o, 4)},
        {Axis::kInputChannels, AlignByN(params_.weights_shape.i, 4)}};
    if (params_.batched_weights) {
      weights_desc.group_sizes.push_back({Axis::kHeight, 0});
    }
    if (!params_.x_kernel_is_1) {
      weights_desc.group_sizes.push_back({Axis::kWidth, 0});
    }
    if (!params_.y_kernel_is_1) {
      weights_desc.group_sizes.push_back({Axis::kHeight, 0});
    }
    return weights_desc;
  }

  std::string_view GetDebugName() const override { return "conv_apple_mpp"; }

  template <DataType T>
  void UploadWeights(const Tensor<OHWI, T>& weights);
  template <DataType T>
  void UploadBias(const Tensor<Linear, T>& bias);

 private:
  ConvParams params_;
  int m_tile_ = 64;
  int n_tile_ = 64;
  int simdgroups_ = 4;

  std::string GetKernelCode(const TensorDescriptor& src) const;
};

template <DataType T>
void ConvAppleMPP::UploadWeights(const Tensor<OHWI, T>& weights) {
  const WeightsDescription weights_desc = GetWeightsDescription();

  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor buffer_desc;
  buffer_desc.element_type = params_.weights_data_type;
  buffer_desc.element_size = 1;
  buffer_desc.size = elements_count * SizeOf(weights_desc.type);
  buffer_desc.data.resize(buffer_desc.size);
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(buffer_desc.data));
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

template <DataType T>
void ConvAppleMPP::UploadBias(const Tensor<Linear, T>& bias) {
  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      params_.weights_data_type, TensorStorageType::kBuffer, bias);
  args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                std::move(bias_tensor_desc)));
}

// Checks if the Apple MPP convolution is supported on the given GPU.
bool SupportsConvAppleMPP(const GpuInfo& gpu_info);
bool SupportsConvAppleMPP(const GpuInfo& gpu_info,
                          const ExternalWeights& weights);

ConvAppleMPP CreateConvAppleMPP(const TensorDescriptor& src,
                                const TensorDescriptor& dst,
                                const Convolution2DAttributes& attr);

// Creates an Apple MPP convolution operation with the given attributes.
ConvAppleMPP CreateConvAppleMPP(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const Tensor<OHWI, DataType::kFloat32>& weights,
    const Tensor<Linear, DataType::kFloat32>& bias = {});

// Creates an Apple MPP convolution operation with external weights.
ConvAppleMPP CreateConvAppleMPPExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const OHWI& weights_shape, const TensorDescriptor* bias = nullptr,
    const TensorDescriptor* src_exp = nullptr,
    bool different_weights_for_height = false,
    const ConvRuntimeCheckDesc& runtime_check = {});

// Experimental
ConvAppleMPP CreateConvAppleMPPExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias = nullptr,
    const TensorDescriptor* src_exp = nullptr,
    bool different_weights_for_height = false,
    const ConvRuntimeCheckDesc& runtime_check = {},
    const BHWC* dst_shape = nullptr);

// Creates an Apple MPP convolution operation with INT8 weights.
ConvAppleMPP CreateConvAppleMPPInt8(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const Tensor<OHWI, DataType::kInt8>& weights);

// Creates an Apple MPP convolution operation with INT8 external weights.
ConvAppleMPP CreateConvAppleMPPInt8(const TensorDescriptor& src,
                                    const TensorDescriptor& dst,
                                    const OHWI& weights_shape);

// Creates an Apple MPP convolution operation with INT8 external weights.
ConvAppleMPP CreateConvAppleMPPInt8(const TensorDescriptor& src,
                                    const TensorDescriptor& dst,
                                    const ExternalWeights& weights);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_APPLE_MPP_H_
