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

#include <memory>
#include <utility>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/convolution_transposed.h"
#include "ml_drift/common/kernels/convolution_transposed_2x2.h"
#include "ml_drift/common/kernels/convolution_transposed_3x3.h"
#include "ml_drift/common/kernels/convolution_transposed_3x3_thin.h"
#include "ml_drift/common/kernels/convolution_transposed_4x4.h"
#include "ml_drift/common/kernels/convolution_transposed_thin.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/weights_layout.h"

namespace ml_drift {
namespace {

std::unique_ptr<GPUOperation> SelectConvolutionTransposedAdreno(
    const ConvolutionTransposedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (IsConvolutionTransposedThinSupported(attr)) {
    ConvolutionTransposedThin conv =
        CreateConvolutionTransposedThin(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposedThin>(std::move(conv));
  } else if (IsConvolutionTransposed3x3ThinSupported(attr)) {
    ConvolutionTransposed3x3Thin conv =
        CreateConvolutionTransposed3x3Thin(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed3x3Thin>(std::move(conv));
  } else if (IsConvolutionTransposed2x2Supported(gpu_info, op_def, attr)) {
    ConvolutionTransposed2x2 conv =
        CreateConvolutionTransposed2x2(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed2x2>(std::move(conv));
  } else if (IsConvolutionTransposed3x3Supported(attr)) {
    ConvolutionTransposed3x3 conv =
        CreateConvolutionTransposed3x3(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed3x3>(std::move(conv));
  } else if (IsConvolutionTransposed4x4Supported(attr)) {
    ConvolutionTransposed4x4 conv =
        CreateConvolutionTransposed4x4(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed4x4>(std::move(conv));
  } else {
    ConvolutionTransposed conv =
        CreateConvolutionTransposed(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionTransposedPowerVR(
    const ConvolutionTransposedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (IsConvolutionTransposedThinSupported(attr)) {
    ConvolutionTransposedThin conv =
        CreateConvolutionTransposedThin(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposedThin>(std::move(conv));
  } else if (IsConvolutionTransposed3x3ThinSupported(attr)) {
    ConvolutionTransposed3x3Thin conv =
        CreateConvolutionTransposed3x3Thin(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed3x3Thin>(std::move(conv));
  } else if (IsConvolutionTransposed3x3Supported(attr)) {
    ConvolutionTransposed3x3 conv =
        CreateConvolutionTransposed3x3(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed3x3>(std::move(conv));
  } else if (IsConvolutionTransposed4x4Supported(attr)) {
    ConvolutionTransposed4x4 conv =
        CreateConvolutionTransposed4x4(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed4x4>(std::move(conv));
  } else {
    ConvolutionTransposed conv =
        CreateConvolutionTransposed(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionTransposedMali(
    const ConvolutionTransposedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  ConvolutionTransposed conv =
      CreateConvolutionTransposed(gpu_info, op_def, precision, attr);
  return std::make_unique<ConvolutionTransposed>(std::move(conv));
}
}  // namespace

std::unique_ptr<GPUOperation> SelectConvolutionTransposed(
    const ConvolutionTransposedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (gpu_info.IsApiOpenGl() || gpu_info.IsApiVulkan() || gpu_info.IsAMD() ||
      gpu_info.IsApiWebGpu()) {
    ConvolutionTransposed conv =
        CreateConvolutionTransposed(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvolutionTransposed>(std::move(conv));
  }
  if (gpu_info.IsAdreno()) {
    return SelectConvolutionTransposedAdreno(attr, gpu_info, op_def, precision);
  } else if (gpu_info.IsPowerVR() || gpu_info.IsAMD() || gpu_info.IsNvidia() ||
             gpu_info.IsIntel() || gpu_info.IsApple()) {
    return SelectConvolutionTransposedPowerVR(attr, gpu_info, op_def,
                                              precision);
  } else if (gpu_info.IsMali()) {
    return SelectConvolutionTransposedMali(attr, gpu_info, op_def, precision);
  } else {
    return SelectConvolutionTransposedAdreno(attr, gpu_info, op_def, precision);
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionTransposedExternalWeights(
    const ConvolutionTransposedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision,
    WeightsDescription* weights_desc) {
  if (gpu_info.IsApiOpenGl() || gpu_info.IsApiVulkan() || gpu_info.IsAMD() ||
      gpu_info.IsApiWebGpu()) {
    ConvolutionTransposed conv = CreateConvolutionTransposedDynamicWeights(
        gpu_info, op_def, precision, attr);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvolutionTransposed>(std::move(conv));
  }
  if (gpu_info.IsAdreno()) {
    if (IsConvolutionTransposed3x3ThinSupported(attr)) {
      ConvolutionTransposed3x3Thin conv =
          CreateConvolutionTransposed3x3ThinDynamicWeights(gpu_info, op_def,
                                                           precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed3x3Thin>(std::move(conv));
    } else if (IsConvolutionTransposed2x2Supported(gpu_info, op_def, attr)) {
      ConvolutionTransposed2x2 conv =
          CreateConvolutionTransposed2x2DynamicWeights(gpu_info, op_def,
                                                       precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed2x2>(std::move(conv));
    } else if (IsConvolutionTransposed3x3Supported(attr)) {
      ConvolutionTransposed3x3 conv =
          CreateConvolutionTransposed3x3DynamicWeights(gpu_info, op_def,
                                                       precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed3x3>(std::move(conv));
    } else if (IsConvolutionTransposed4x4Supported(attr)) {
      ConvolutionTransposed4x4 conv =
          CreateConvolutionTransposed4x4DynamicWeights(gpu_info, op_def,
                                                       precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed4x4>(std::move(conv));
    } else {
      ConvolutionTransposed conv = CreateConvolutionTransposedDynamicWeights(
          gpu_info, op_def, precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed>(std::move(conv));
    }
  } else if (gpu_info.IsPowerVR() || gpu_info.IsAMD() || gpu_info.IsNvidia() ||
             gpu_info.IsIntel()) {
    if (IsConvolutionTransposed4x4Supported(attr)) {
      ConvolutionTransposed4x4 conv =
          CreateConvolutionTransposed4x4DynamicWeights(gpu_info, op_def,
                                                       precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed4x4>(std::move(conv));
    } else if (IsConvolutionTransposed3x3ThinSupported(attr)) {
      ConvolutionTransposed3x3Thin conv =
          CreateConvolutionTransposed3x3ThinDynamicWeights(gpu_info, op_def,
                                                           precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed3x3Thin>(std::move(conv));
    } else if (IsConvolutionTransposed3x3Supported(attr)) {
      ConvolutionTransposed3x3 conv =
          CreateConvolutionTransposed3x3DynamicWeights(gpu_info, op_def,
                                                       precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed3x3>(std::move(conv));
    } else {
      ConvolutionTransposed conv = CreateConvolutionTransposedDynamicWeights(
          gpu_info, op_def, precision, attr);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvolutionTransposed>(std::move(conv));
    }
  } else {
    ConvolutionTransposed conv = CreateConvolutionTransposedDynamicWeights(
        gpu_info, op_def, precision, attr);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvolutionTransposed>(std::move(conv));
  }
}

}  // namespace ml_drift
