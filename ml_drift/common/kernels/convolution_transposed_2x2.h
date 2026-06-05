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

#ifndef ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_2X2_
#define ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_2X2_

#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/convolution_transposed_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvolutionTransposed2x2 : public GPUOperation {
 public:
  ConvolutionTransposed2x2() = default;
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  ConvolutionTransposed2x2(ConvolutionTransposed2x2&& operation) = default;
  ConvolutionTransposed2x2& operator=(ConvolutionTransposed2x2&& operation) =
      default;
  ConvolutionTransposed2x2(const ConvolutionTransposed2x2&) = delete;
  ConvolutionTransposed2x2& operator=(const ConvolutionTransposed2x2&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = weights_data_type_;
    desc.layout = WeightsLayout::kOICustomSpatialI4O4;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

 private:
  ConvolutionTransposed2x2(const OperationDef& definition,
                           CalculationsPrecision precision,
                           const GpuInfo& gpu_info, bool has_bias);

  friend ConvolutionTransposed2x2 CreateConvolutionTransposed2x2(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed2x2 CreateConvolutionTransposed2x2DynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);

  std::vector<int> GetSpatialWeightsRemap() const;

  DataType weights_data_type_;
  int wave_size_ = 32;  // for WeightsUploadType::kWaveMemory
  WeightsUploadType weights_upload_type_;
};

// Checks if the 2x2 transposed convolution is supported on the given GPU.
bool IsConvolutionTransposed2x2Supported(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

// Creates a 2x2 transposed convolution operation.
ConvolutionTransposed2x2 CreateConvolutionTransposed2x2(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

// Creates a 2x2 transposed convolution operation with dynamic weights.
ConvolutionTransposed2x2 CreateConvolutionTransposed2x2DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_2X2_
