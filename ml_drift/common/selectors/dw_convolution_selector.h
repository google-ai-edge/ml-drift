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

#ifndef ML_DRIFT_COMMON_SELECTORS_DW_CONVOLUTION_SELECTOR_H_
#define ML_DRIFT_COMMON_SELECTORS_DW_CONVOLUTION_SELECTOR_H_

#include <memory>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

std::unique_ptr<GPUOperation> SelectDWConvolution(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision);

std::unique_ptr<GPUOperation> SelectDWConvolutionExternalWeights(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_SELECTORS_DW_CONVOLUTION_SELECTOR_H_
