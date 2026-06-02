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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_CONSTANTS_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_CONSTANTS_H_

#include <string_view>
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
#include "ml_drift/common/util.h"

namespace ml_drift {
class ConvConstants : public GPUOperation {
 public:
  ConvConstants() = default;
  ConvConstants(const GpuInfo& gpu_info, const OperationDef& definition,
                CalculationsPrecision precision,
                const Convolution2DAttributes& attr, bool has_bias);
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = weights_data_type_;
    if (weights_shape_.i % 4 == 0 && weights_shape_.o % 4 == 0) {
      desc.layout = WeightsLayout::kOISpatialOGroupI4O4;
      desc.output_group_size = DivideRoundUp(weights_shape_.o, 4);
    } else {
      desc.layout = WeightsLayout::kISpatialOI4O4UnalignedIO;
    }
    return desc;
  }

  std::string_view GetDebugName() const override { return "conv_constants"; }

  // Move only
  ConvConstants(ConvConstants&& kernel) = default;
  ConvConstants& operator=(ConvConstants&& kernel) = default;
  ConvConstants(const ConvConstants&) = delete;
  ConvConstants& operator=(const ConvConstants&) = delete;

 private:
  friend ConvConstants CreateConvConstants(const GpuInfo& gpu_info,
                                           const OperationDef& definition,
                                           CalculationsPrecision precision,
                                           const Convolution2DAttributes& attr);

  friend ConvConstants CreateConvConstantsExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision, const Convolution2DAttributes& attr,
      const TensorDescriptor* bias);

  void UploadWeights(const GpuInfo& gpu_info,
                     const Tensor<OHWI, DataType::FLOAT32>& weights);

  bool src_local_memory_caching_ = false;
  DataType weights_data_type_;
  OHWI weights_shape_;
};

// Checks if the convolution with constants is supported on the given GPU.
bool IsConvConstantsSupported(const GpuInfo& gpu_info,
                              CalculationsPrecision precision,
                              const Convolution2DAttributes& attr);

// Creates a convolution operation with constants.
ConvConstants CreateConvConstants(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  CalculationsPrecision precision,
                                  const Convolution2DAttributes& attr);

// Creates a convolution operation with constants and external weights.
ConvConstants CreateConvConstantsExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const Convolution2DAttributes& attr,
    const TensorDescriptor* bias);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_CONSTANTS_H_
