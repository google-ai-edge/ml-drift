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

#ifndef ML_DRIFT_COMMON_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_
#define ML_DRIFT_COMMON_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_

#include <memory>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

// Creates a GPU operation for the QuantizeAndDequantize operation.
// Performs the operation: {Quantize, Dequantize} on floating-point data.
// We need this operation to emulate the error introduced by quantization
// on the GPU, which cannot represent int8 tensors.
//
// Implemented as:
// qvalue = round((min(qmax, max(qmin, src_val)) - qmin) * (1/qscale))
// dq_value = qvalue * qscale + qmin
// Here, qmin, qmax & qscale refer to the quantization values as implemented in
// TensorFlow Lite's 'FakeQuant' kernel.
//
// NOTE: We do not need to nudge min/max values in this op, since they would
// already be adjusted while generating the quantized model.
GPUOperation CreateQuantizeAndDequantize(
    const OperationDef& definition,
    const QuantizeAndDequantizeAttributes& attr);

// Creates a GPU operation for the static range quantization operation.
// The Quantization part of CreateQuantizeAndDequantize
//
// Implemented as:
// qvalue = round((min(qmax, max(qmin, src_val)) - qmin) * (1/qscale))
//
// Refer to CreateQuantizeAndDequantize for more details.
GPUOperation CreateStaticRangeQuantization(
    const OperationDef& definition,
    const QuantizeAndDequantizeAttributes& attr);

// Creates a GPU operation for the quantization operation.
// Supports quantization to int8/uint8/int4/uint4 with packing
// definition.dst_tensors[0].type must be ToSpatialTensorType(dst_type);
//
// definition.dst_tensors[1] receives per-(b, h, w) (inv_scale, offset, [sum])
// such that x ~= q * inv_scale + offset for quantized values q. sum is the sum
// of q across C, written when `calculate_sum` is true.
//
// If `symmetric` is true, scales by max(|x|) instead of [min, max] and maps 0
// to the midpoint of a narrow range (e.g. int8 in [-127, 127] with offset = 0,
// or uint8 in [1, 255] around 128 with offset = -128 * inv_scale).
std::unique_ptr<GPUOperation> CreateQuantization(const OperationDef& definition,
                                                 PackedType dst_type,
                                                 const GpuInfo& gpu_info,
                                                 const BHWC& shape,
                                                 bool calculate_sum = false,
                                                 bool symmetric = false);

// Creates a GPU operation for the dequantization operation.
GPUOperation CreateDequantization(
    const OHWI& weights_shape, const GpuInfo& gpu_info,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const TensorDescriptor& src_scale_zp_sum,
    const TensorDescriptor& weights_sum, const TensorDescriptor& weights_scale,
    const TensorDescriptor* weights_zero_point = nullptr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_
