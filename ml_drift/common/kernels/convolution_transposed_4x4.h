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

#ifndef ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_
#define ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_

#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/convolution_transposed_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvolutionTransposed4x4 : public GPUOperation {
 public:
  ConvolutionTransposed4x4() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed4x4(ConvolutionTransposed4x4&& operation) = default;
  ConvolutionTransposed4x4& operator=(ConvolutionTransposed4x4&& operation) =
      default;
  ConvolutionTransposed4x4(const ConvolutionTransposed4x4&) = delete;
  ConvolutionTransposed4x4& operator=(const ConvolutionTransposed4x4&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = weights_data_type_;
    desc.layout = weights_layout_;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

 private:
  ConvolutionTransposed4x4(const OperationDef& definition,
                           CalculationsPrecision precision,
                           const GpuInfo& gpu_info, bool has_bias);

  friend ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);

  void UploadWeights(const GpuInfo& gpu_info,
                     const Tensor<OHWI, DataType::FLOAT32>& weights);

  std::vector<int> GetSpatialWeightsRemap() const;

  std::string GenerateConvolutionTransposedCode(
      const GpuInfo& gpu_info, const OperationDef& op_def,
      CalculationsPrecision precision, WeightsUploadType weights_upload_type,
      bool has_bias);

  WeightsUploadType weights_upload_type_;
  WeightsLayout weights_layout_;
  DataType weights_data_type_;
  int wave_size_ = 32;  // for wave memory upload
};

// Checks if the 4x4 transposed convolution is supported.
bool IsConvolutionTransposed4x4Supported(
    const ConvolutionTransposedAttributes& attr);

// Creates a 4x4 transposed convolution operation.
ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

// Creates a 4x4 transposed convolution operation with dynamic weights.
ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_
