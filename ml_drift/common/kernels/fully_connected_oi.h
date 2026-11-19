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

#ifndef ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_OI_H_
#define ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_OI_H_

#include <stdint.h>

#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
class FullyConnectedOI : public GPUOperation {
 public:
  struct ConvParams {
    DataType weights_type;
    OHWI scale_zp_shape = OHWI(1, 1, 1, 1);  // applicable with int8/4/2 weights
    bool softmax_input_activation = false;
    bool batched_weights = false;
    int runtime_batch_ids = 0;
    bool has_bias = true;
    bool has_zero_point = true;  // applicable with int8/4/2 weights
    bool sparse_2x4 = false;
    int src_n = 1;
    ConvRuntimeCheckDesc runtime_check;
    BHWC block_size = BHWC(1, 1, 1, 1);
  };

  FullyConnectedOI() = default;
  FullyConnectedOI(const TensorDescriptor& src, const TensorDescriptor& dst,
                   CalculationsPrecision precision, const GpuInfo& gpu_info,
                   const OHWI& weights_shape,
                   const WeightsDescription& weights_desc,
                   const ConvParams& conv_params);
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override {
    int w_batch_size = conv_params_.batched_weights ? dst_[0]->Height() : 1;
    int w_groups = 1;
    if (conv_params_.runtime_check.packed_groups.has_value()) {
      w_groups = DivideRoundUp(
          conv_params_.runtime_check.packed_groups->max_group_size,
          conv_params_.block_size.w);
      w_batch_size = conv_params_.runtime_check.packed_groups->num_groups;
    }
    return int3(work_group_size_.x * w_groups, dst_[0]->Slices(), w_batch_size);
  }

  // Move only
  FullyConnectedOI(FullyConnectedOI&& kernel) = default;
  FullyConnectedOI& operator=(FullyConnectedOI&& kernel) = default;
  FullyConnectedOI(const FullyConnectedOI&) = delete;
  FullyConnectedOI& operator=(const FullyConnectedOI&) = delete;

  ConvParams conv_params_;
};

FullyConnectedOI CreateFullyConnectedOI(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias = nullptr,
    const BHWC* dst_shape_ptr = nullptr,
    const TensorDescriptor* src_exp = nullptr,
    const ConvRuntimeCheckDesc& runtime_check = {});

FullyConnectedOI CreateFullyConnectedOIWeightsBatchIds(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& batch_ids,
    const TensorDescriptor& dst, const ExternalWeights& weights,
    const TensorDescriptor* bias = nullptr,
    const BHWC* dst_shape_ptr = nullptr);

FullyConnectedOI CreateFullyConnectedOIInt4Sparse2x4(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias = nullptr,
    const BHWC* dst_shape_ptr = nullptr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_OI_H_
