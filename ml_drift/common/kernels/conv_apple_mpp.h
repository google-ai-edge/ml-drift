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
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
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
  ConvAppleMPP(const TensorDescriptor& src, const OHWI& weights_shape,
               DataType weights_data_type,
               bool different_weights_for_height = false,
               bool softmax_input_activation = false,
               const ConvRuntimeCheckDesc& runtime_check = {});
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
    if (external_weights_params_.weights_desc.layout !=
        WeightsLayout::kUnknown) {
      return external_weights_params_.weights_desc;
    }
    WeightsDescription weights_desc;
    weights_desc.type = weights_data_type_;
    weights_desc.layout = WeightsLayout::kCustomGroups;
    weights_desc.group_sizes = {
        {Axis::OUTPUT_CHANNELS, AlignByN(weights_shape_.o, 4)},
        {Axis::INPUT_CHANNELS, AlignByN(weights_shape_.i, 4)}};
    if (batched_weights_) {
      weights_desc.group_sizes.push_back({Axis::HEIGHT, 0});
    }
    return weights_desc;
  }

  std::string_view GetDebugName() const override { return "conv_apple_mpp"; }

  template <DataType T>
  void UploadWeights(const Tensor<OHWI, T>& weights);
  template <DataType T>
  void UploadBias(const Tensor<Linear, T>& bias);

  struct ExternalWeightsParams {
    WeightsDescription weights_desc;
    OHWI scale_zp_shape = OHWI(1, 1, 1, 1);
    bool has_zero_point = false;
    int src_group_slices = 0;
  };
  void SetExternalWeightsParams(const ExternalWeightsParams& weights_params) {
    external_weights_params_ = weights_params;
  }
  void SetNTile(int n_tile) { n_tile_ = n_tile; }
  std::string GetKernelCode(bool has_batch, bool has_bias = false) const;

 private:
  TensorDescriptor src_desc_;
  int m_tile_ = 64;
  int n_tile_ = 64;
  int simdgroups_ = 4;
  OHWI weights_shape_;
  DataType weights_data_type_;
  bool batched_weights_ = false;
  bool softmax_input_activation_ = false;
  ConvRuntimeCheckDesc runtime_check_;
  ExternalWeightsParams external_weights_params_;
};

template <DataType T>
void ConvAppleMPP::UploadWeights(const Tensor<OHWI, T>& weights) {
  const WeightsDescription weights_desc = GetWeightsDescription();

  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor buffer_desc;
  buffer_desc.element_type = weights_data_type_;
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
      weights_data_type_, TensorStorageType::BUFFER, bias);
  args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                std::move(bias_tensor_desc)));
}

// Checks if the Apple MPP convolution is supported on the given GPU.
bool SupportsConvAppleMPP(const GpuInfo& gpu_info);
bool SupportsConvAppleMPP(const GpuInfo& gpu_info,
                          const ExternalWeights& weights);

// Creates an Apple MPP convolution operation with the given attributes.
ConvAppleMPP CreateConvAppleMPP(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const Tensor<OHWI, DataType::FLOAT32>& weights,
    const Tensor<Linear, DataType::FLOAT32>& bias = {});

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
    const ConvRuntimeCheckDesc& runtime_check = {});

// Creates an Apple MPP convolution operation with INT8 weights.
ConvAppleMPP CreateConvAppleMPPInt8(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const Tensor<OHWI, DataType::INT8>& weights);

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
