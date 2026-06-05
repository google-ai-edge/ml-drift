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

#ifndef ML_DRIFT_COMMON_TASK_WEIGHTS_CONVERSION_H_
#define ML_DRIFT_COMMON_TASK_WEIGHTS_CONVERSION_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

std::unique_ptr<half[]> ConvertF32F16(const std::vector<float>& src);

unsigned int GetTotalElementsCountForLayout(
    const WeightsDescription& weight_desc, const OHWI& shape);
unsigned int GetTotalElementsCountForLayout(
    const WeightsDescription& weight_desc, const OHWDI& shape);

// Applicable to:
//   k2DX4I4YIsSpatialIAndXIsOOGroupO4
//   k2DX4O4YIsSpatialIAndXIsOOGroupI4
//   k2DYIsSpatialIOAndXIsOGroupI4O4
uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWI& shape);
// Applicable to:
//   k2DX4I4YIsSpatialIAndXIsOOGroupO4
//   k2DX4O4YIsSpatialIAndXIsOOGroupI4
//   k2DYIsSpatialIOAndXIsOGroupI4O4
uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWDI& shape);

void RearrangeWeights(const Tensor<OHWI, DataType::FLOAT32>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst);

void RearrangeWeights(const Tensor<OHWDI, DataType::FLOAT32>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst);

void RearrangeWeights(const Tensor<OHWI, DataType::INT8>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst);

void RearrangeWeights(const Tensor<OHWI, DataType::UINT8>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst, uint8_t pad_value = 0);

void RearrangeWeightsInt8AsUint8(const Tensor<OHWI, DataType::INT8>& weights,
                                 const WeightsDescription& dst_weight_desc,
                                 absl::Span<uint8_t> dst, int shift_value,
                                 unsigned int pad_value);

void RearrangeWeightsInt8AsUint4(const Tensor<OHWI, DataType::INT8>& weights,
                                 const WeightsDescription& dst_weight_desc,
                                 absl::Span<uint8_t> dst, int shift_value,
                                 unsigned int pad_value);

void RearrangeWeightsInt8AsUint2(const Tensor<OHWI, DataType::INT8>& weights,
                                 const WeightsDescription& dst_weight_desc,
                                 absl::Span<uint8_t> dst, int shift_value,
                                 unsigned int pad_value);

void RearrangeWeightsInt4(const Tensor<OHWI, DataType::INT8>& weights_i4,
                          const WeightsDescription& dst_weight_desc,
                          absl::Span<uint8_t> dst);

void RearrangeWeightsUint2(const Tensor<OHWI, DataType::UINT8>& weights_i2,
                           const WeightsDescription& dst_weight_desc,
                           absl::Span<uint8_t> dst);

absl::Status RearrangeWeightsUInt4Packed(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims);

absl::Status RearrangeWeightsUInt2Packed(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims);

Tensor<Linear, DataType::INT32> GetWeightsAccumulatedInputChannels(
    const Tensor<OHWI, DataType::INT8>& weights);

// Returns tensor descriptor(s) with a proper shape, layout, and rearranged
// weights from the weights_desc.
std::vector<TensorDescriptor> GetTensorDescriptorsForWeightsLayout(
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& weights_desc);
TensorDescriptor GetTensorDescriptorForWeightsLayout(
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights,
    const WeightsDescription& weights_desc);

// Returns descriptors without data.
std::vector<TensorDescriptor> GetTensorDescriptorsForWeightsLayout(
    const OHWI& weights_shape, const WeightsDescription& weights_desc);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_WEIGHTS_CONVERSION_H_
