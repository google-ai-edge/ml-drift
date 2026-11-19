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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_WEIGHTS_CONVERTER_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_WEIGHTS_CONVERTER_H_

#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConverterToConvWeights : public GPUOperation {
 public:
  ConverterToConvWeights(const GpuInfo& gpu_info,
                         const OperationDef& definition,
                         const OHWI& weights_shape,
                         const WeightsDescription& weights_desc,
                         Layout input_layout,
                         const TensorDescriptor* weights_scale = nullptr,
                         const TensorDescriptor* weights_zero_point = nullptr);
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  ConverterToConvWeights(ConverterToConvWeights&& operation) = default;
  ConverterToConvWeights& operator=(ConverterToConvWeights&& operation) =
      default;
  ConverterToConvWeights(const ConverterToConvWeights&) = delete;
  ConverterToConvWeights& operator=(const ConverterToConvWeights&) = delete;

 private:
  std::string GetConverterToConvWeightsCode(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const TensorDescriptor* weights_scale,
      const TensorDescriptor* weights_zero_point);
  std::string GetCodeUnalignedIO(const OperationDef& definition);

  OHWI weights_shape_;
  WeightsDescription weights_desc_;
  Layout input_layout_;  // Can be only OHWI or HWIO
  int groups_per_x_ = 0;  // used only with weights_desc_.IsLinearLayout()
  // if input_layout_ is OHWI: reinterpreting weights as OHWI-BHWC tensor
  // if input_layout_ is HWIO: reinterpreting weights as HWIO-BHWC tensor
};

// Can dequantize weights with int8/int4 as uint8 to float.
class WeightsConverter : public GPUOperation {
 public:
  WeightsConverter(const GpuInfo& gpu_info, const OperationDef& definition,
                   const ExternalWeights& src_weights,
                   const WeightsDescription& dst_weights_desc,
                   const ConvRuntimeCheckDesc& runtime_check = {});
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  WeightsConverter(WeightsConverter&& operation) = default;
  WeightsConverter& operator=(WeightsConverter&& operation) = default;
  WeightsConverter(const WeightsConverter&) = delete;
  WeightsConverter& operator=(const WeightsConverter&) = delete;

 private:
  OHWI weights_shape_;
  WeightsDescription dst_weights_desc_;
  bool use_2d_grid_x_is_i_ogroup_y_is_o_ = false;
  int groups_per_x_ = 0;  // used only with dst_weights_desc_.IsLinearLayout()
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_WEIGHTS_CONVERTER_H_
