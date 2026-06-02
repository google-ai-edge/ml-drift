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

#ifndef ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
#define ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_

#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvolutionTransposed3x3Thin : public GPUOperation {
 public:
  ConvolutionTransposed3x3Thin() = default;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed3x3Thin(ConvolutionTransposed3x3Thin&& operation) =
      default;
  ConvolutionTransposed3x3Thin& operator=(
      ConvolutionTransposed3x3Thin&& operation) = default;
  ConvolutionTransposed3x3Thin(const ConvolutionTransposed3x3Thin&) = delete;
  ConvolutionTransposed3x3Thin& operator=(const ConvolutionTransposed3x3Thin&) =
      delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = weights_data_type_;
    desc.layout = weights_layout_;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

 private:
  ConvolutionTransposed3x3Thin(const GpuInfo& gpu_info,
                               const OperationDef& definition,
                               CalculationsPrecision precision,
                               const ConvolutionTransposedAttributes& attr);

  friend ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed3x3Thin
  CreateConvolutionTransposed3x3ThinDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);

  void UploadWeights(const GpuInfo& gpu_info,
                     const Tensor<OHWI, DataType::FLOAT32>& weights);

  std::vector<int> GetSpatialWeightsRemap() const;

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                CalculationsPrecision precision,
                                                const GpuInfo& gpu_info,
                                                int src_depth, int dst_depth,
                                                bool has_bias);

  WeightsLayout weights_layout_;
  DataType weights_data_type_;
};

// Checks if the thin 3x3 transposed convolution is supported.
bool IsConvolutionTransposed3x3ThinSupported(
    const ConvolutionTransposedAttributes& attr);

// Creates a thin 3x3 transposed convolution operation.
ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

// Creates a thin 3x3 transposed convolution operation with dynamic weights.
ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3ThinDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
