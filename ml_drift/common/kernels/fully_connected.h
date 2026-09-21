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

#ifndef ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_H_
#define ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/status/statusor.h"
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

namespace ml_drift {
class FullyConnected : public GPUOperation {
 public:
  FullyConnected() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override;

  struct ConvParams {
    DataType weights_type;
    OHWI scale_zp_shape = OHWI(1, 1, 1, 1);  // applicable with int8/4/2 weights
    bool softmax_input_activation = false;
    bool batched_weights = false;
    bool has_bias = true;
    bool has_zero_point = true;  // applicable with int8/4/2 weights
    bool sparse_2x4 = false;
    int runtime_batch_ids = 0;
    ConvRuntimeCheckDesc runtime_check;
    BHWC block_size = BHWC(1, 1, 1, 1);
    int3 wg_size = int3(0, 0, 0);
  };

  // Move only
  FullyConnected(FullyConnected&& kernel) = default;
  FullyConnected& operator=(FullyConnected&& kernel) = default;
  FullyConnected(const FullyConnected&) = delete;
  FullyConnected& operator=(const FullyConnected&) = delete;

 private:
  FullyConnected(const TensorDescriptor& src, const TensorDescriptor& dst,
                 CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const OHWI& weights_shape,
                 const WeightsDescription& weights_desc,
                 const ConvParams& conv_params);
  friend absl::StatusOr<FullyConnected>
  CreateFullyConnectedWeightsAreSpatialTensor(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision, const OHWI& weights_shape,
      const TensorDescriptor* bias, const BHWC* dst_shape_ptr,
      const int3* wg_size, const ConvRuntimeCheckDesc& runtime_check);
  friend absl::StatusOr<FullyConnected> CreateFullyConnectedExternalWeights(
      const GpuInfo& gpu_info, CalculationsPrecision precision,
      const TensorDescriptor& src, const TensorDescriptor& dst,
      const ExternalWeights& weights, const TensorDescriptor* bias,
      const BHWC* dst_shape_ptr, const TensorDescriptor* src_exp,
      const ConvRuntimeCheckDesc& runtime_check, const int3* wg_size);
  friend absl::StatusOr<FullyConnected> CreateFullyConnectedWeightsBatchIds(
      const GpuInfo& gpu_info, CalculationsPrecision precision,
      const TensorDescriptor& src, const TensorDescriptor& batch_ids,
      const TensorDescriptor& dst, const ExternalWeights& weights,
      const TensorDescriptor* bias, const BHWC* dst_shape_ptr);

  friend FullyConnected CreateFullyConnectedInt4Sparse2x4(
      const GpuInfo& gpu_info, CalculationsPrecision precision,
      const TensorDescriptor& src, const TensorDescriptor& dst,
      const ExternalWeights& weights, const TensorDescriptor* bias,
      const BHWC* dst_shape_ptr, const int3* wg_size);

  std::string GetFullyConnectedKernelCode(
      const TensorDescriptor& src, const TensorDescriptor& dst,
      CalculationsPrecision precision, const GpuInfo& gpu_info,
      const WeightsDescription& weights_desc, int scale_zp_group_size);

  bool wg_reduction_;
  ConvParams conv_params_;
};

// Returns the recommended maximum total spatial size for the given GPU and
// precision.
int GetRecommendedMaxTotalSpatialSize(const GpuInfo& gpu_info,
                                      CalculationsPrecision precision);

// Checks if the fully connected weights are spatial tensor is supported.
inline bool IsFullyConnectedWeightsAreSpatialTensorSupported(
    const OHWI& weight_shape) {
  return weight_shape.o % 4 == 0 && weight_shape.i % 4 == 0 &&
         weight_shape.w == 1;
}

// Creates a fully connected operation with weights as a spatial tensor.
// FullyConnected with runtime(spatial) second tensor(BHWC as OHWI)
// if H != 1 in OHWI then it will be used as 'batch' for height dimension and H
// in BHWC must be equal to H in OHWI.
absl::StatusOr<FullyConnected> CreateFullyConnectedWeightsAreSpatialTensor(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const OHWI& weights_shape,
    const TensorDescriptor* bias = nullptr, const BHWC* dst_shape_ptr = nullptr,
    const int3* wg_size = nullptr,
    const ConvRuntimeCheckDesc& runtime_check = {});

// Returns the buffer based weights description for a fully connected operation.
WeightsDescription GetFullyConnectedWeightsDesc(DataType weights_type,
                                                const OHWI& weights_shape);
// Returns the weights description(buffer or texture) for a fully connected
// operation.
WeightsDescription GetFullyConnectedWeightsDesc(const GpuInfo& gpu_info,
                                                DataType weights_type,
                                                const OHWI& weights_shape);

// Creates a fully connected operation with external weights.
// Supported layouts for fp32/fp16 weights:
//   kOSpatialIOGroupI4O4 with OGroup = output_slices (SpatialIOI4O4)
//   kOSpatialIOGroupO4I4 with OGroup = output_slices (SpatialIOO4I4)
//   k2DX4I4YIsSpatialIAndXIsOOGroupO4 with OGroup = output_slices
// For INT8/INT4/INT2 weights must be used with corresponding description
// functions (GetFullyConnectedInt[8/4/2]WeightsDesc).
absl::StatusOr<FullyConnected> CreateFullyConnectedExternalWeights(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias,
    const BHWC* dst_shape_ptr, const TensorDescriptor* src_exp = nullptr,
    const ConvRuntimeCheckDesc& runtime_check = {},
    const int3* wg_size = nullptr);

// Creates a fully connected operation with batched weights accessed via
// batch_ids. The batch dimension of src and dst is at the H dimension of their
// BHWC shape. Expected input/output shapes:
//
// - If src has shape [1, 1, 1, input_channels], since src.shape.h is 1, the
// kernel will broadcast against the batch_ids to get dst with shape [1,
// batch_ids.shape.c, 1, output_channels].
//
// - If src has shape [1, batch_size, 1, input_channels], then the batch_ids is
// expected to have shape [1, 1, 1, batch_size] and the dst will have the same
// shape as src.
absl::StatusOr<FullyConnected> CreateFullyConnectedWeightsBatchIds(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& batch_ids,
    const TensorDescriptor& dst, const ExternalWeights& weights,
    const TensorDescriptor* bias, const BHWC* dst_shape_ptr);

// Checks if UINT8 math is supported for fully connected operations on the given
// GPU.
bool SupportsFullyConnectedUint8Math(const GpuInfo& gpu_info);

// Returns the weights description for an INT8 fully connected operation.
WeightsDescription GetFullyConnectedInt8WeightsDesc(
    const GpuInfo& gpu_info, const OHWI& weights_shape,
    bool prefer_textures = false);

// Returns the weights description for an INT4 fully connected operation.
WeightsDescription GetFullyConnectedInt4WeightsDesc(
    const GpuInfo& gpu_info, const OHWI& weights_shape,
    bool prefer_textures = false);

// Returns the weights description for an INT2 fully connected operation.
WeightsDescription GetFullyConnectedInt2WeightsDesc(
    const GpuInfo& gpu_info, const OHWI& weights_shape,
    bool prefer_textures = false);

// Creates a sparse 2x4 fully connected operation with INT4 weights.
// prototype, no correctness check, dummy values, not works for all cases
FullyConnected CreateFullyConnectedInt4Sparse2x4(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias = nullptr,
    const BHWC* dst_shape_ptr = nullptr, const int3* wg_size = nullptr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_H_
