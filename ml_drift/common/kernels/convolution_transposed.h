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

#ifndef ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_H_
#define ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
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
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvolutionTransposed : public GPUOperation {
 public:
  ConvolutionTransposed() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed(ConvolutionTransposed&& operation) = default;
  ConvolutionTransposed& operator=(ConvolutionTransposed&& operation) = default;
  ConvolutionTransposed(const ConvolutionTransposed&) = delete;
  ConvolutionTransposed& operator=(const ConvolutionTransposed&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = weights_data_type_;
    desc.layout = weights_layout_;
    desc.output_group_size = block_size_.w;
    return desc;
  }

 private:
  friend ConvolutionTransposed CreateConvolutionTransposed(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed CreateConvolutionTransposed3D(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposed3DAttributes& attr);
  friend ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);

  ConvolutionTransposed(const OperationDef& definition,
                        CalculationsPrecision precision,
                        const ConvolutionTransposedAttributes& attr,
                        const GpuInfo& gpu_info);
  ConvolutionTransposed(const OperationDef& definition,
                        CalculationsPrecision precision,
                        const ConvolutionTransposed3DAttributes& attr,
                        const GpuInfo& gpu_info);

  template <DataType T>
  void UploadWeights(const Tensor<OHWI, T>& weights, bool weights_are_buffer);

  template <DataType T>
  void UploadWeights(const Tensor<OHWDI, T>& weights, bool weights_are_buffer);

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                CalculationsPrecision precision,
                                                const GpuInfo& gpu_info,
                                                const int4& block_size,
                                                bool has_bias);
  int4 stride_;
  int4 block_size_ = int4(1, 1, 1, 1);  // WHDS
  WeightsLayout weights_layout_;
  DataType weights_data_type_;
};

template <DataType T>
void ConvolutionTransposed::UploadWeights(const Tensor<OHWI, T>& weights,
                                          bool weights_are_buffer) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 16;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          weights_desc.type, TensorStorageType::kTexture2D, tex_size.x,
          tex_size.y, weights_data.data() + sub_size * i);
      args_.AddObject("weights" + std::to_string(i),
                      std::make_unique<TensorDescriptor>(std::move(desc)));
    }
  }
}

template <DataType T>
void ConvolutionTransposed::UploadWeights(const Tensor<OHWDI, T>& weights,
                                          bool weights_are_buffer) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 16;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          weights_desc.type, TensorStorageType::kTexture2D, tex_size.x,
          tex_size.y, weights_data.data() + sub_size * i);
      args_.AddObject("weights" + std::to_string(i),
                      std::make_unique<TensorDescriptor>(std::move(desc)));
    }
  }
}

// Creates a transposed convolution operation.
ConvolutionTransposed CreateConvolutionTransposed(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

// Creates a 3D transposed convolution operation.
ConvolutionTransposed CreateConvolutionTransposed3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposed3DAttributes& attr);

// Creates a transposed convolution operation with dynamic weights.
ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_H_
