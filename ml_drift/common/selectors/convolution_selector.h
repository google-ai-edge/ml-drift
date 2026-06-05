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

#ifndef ML_DRIFT_COMMON_SELECTORS_CONVOLUTION_SELECTOR_H_
#define ML_DRIFT_COMMON_SELECTORS_CONVOLUTION_SELECTOR_H_

#include <memory>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

std::unique_ptr<GPUOperation> SelectConvolution(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints);

std::unique_ptr<GPUOperation> SelectConvolutionWithExternalWeights(
    const Convolution2DAttributes& attr, const TensorDescriptor* bias_desc,
    const BHWC& dst_shape, const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints,
    WeightsDescription* weights_desc, const TensorDescriptor* src_exp,
    bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check = {});

bool SupportsConvolutionInt8(const GpuInfo& gpu_info, const BHWC& src_shape);
PackedType GetConvolutionInt8SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape);

std::unique_ptr<GPUOperation> SelectConvolutionInt8(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    PackedType src_packed_type, const Tensor<OHWI, DataType::INT8>& weights,
    const BHWC& dst_shape);

std::unique_ptr<GPUOperation> SelectConvolutionInt8(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    PackedType src_packed_type, const OHWI& weights_shape,
    const BHWC& dst_shape, WeightsDescription* weights_desc);

bool SupportsConvolutionInt4(const GpuInfo& gpu_info, const BHWC& src_shape);
PackedType GetConvolutionInt4SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape);

std::unique_ptr<GPUOperation> SelectConvolutionInt4(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    const OHWI& weights_shape, const BHWC& dst_shape,
    WeightsDescription* weights_desc);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_SELECTORS_CONVOLUTION_SELECTOR_H_
