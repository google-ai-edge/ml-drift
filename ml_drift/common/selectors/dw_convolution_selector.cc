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

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/depthwise_conv.h"
#include "ml_drift/common/kernels/depthwise_conv_3x3.h"
#include "ml_drift/common/kernels/depthwise_conv_tiled.h"
#include "ml_drift/common/kernels/depthwise_conv_wave_memory.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
namespace {

std::unique_ptr<GPUOperation> SelectDWConvolutionDefault(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (IsDepthwiseConvTiledSupported(attr)) {
    return CreateDepthwiseConvTiled(gpu_info, op_def, precision, attr);
  } else {
    return std::make_unique<DepthwiseConv>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, precision, attr));
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionAdreno(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (gpu_info.IsApiOpenCl() &&
      IsDepthwiseConvWaveMemorySupported(gpu_info, attr)) {
    return std::make_unique<DepthwiseConvWaveMemory>(
        CreateDepthwiseConvWaveMemory(gpu_info, op_def, precision, attr));
  } else if (IsDepthwiseConv3x3Supported(gpu_info, attr)) {
    return std::make_unique<DepthwiseConv3x3>(
        CreateDepthwiseConv3x3(gpu_info, op_def, precision, attr));
  } else {
    return std::make_unique<DepthwiseConv>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, precision, attr));
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionPowerVR(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (gpu_info.IsApiOpenCl() &&
      IsDepthwiseConvWaveMemorySupported(gpu_info, attr)) {
    return std::make_unique<DepthwiseConvWaveMemory>(
        CreateDepthwiseConvWaveMemory(gpu_info, op_def, precision, attr));
  } else if (IsDepthwiseConvTiledSupported(attr)) {
    return CreateDepthwiseConvTiled(gpu_info, op_def, precision, attr);
  } else {
    return std::make_unique<DepthwiseConv>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, precision, attr));
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionApple(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (IsDepthwiseConv3x3Supported(gpu_info, attr)) {
    return std::make_unique<DepthwiseConv3x3>(
        CreateDepthwiseConv3x3(gpu_info, op_def, precision, attr));
  } else if (IsDepthwiseConvTiledSupported(attr)) {
    return CreateDepthwiseConvTiled(gpu_info, op_def, precision, attr);
  } else {
    return std::make_unique<DepthwiseConv>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, precision, attr));
  }
}
}  // namespace

std::unique_ptr<GPUOperation> SelectDWConvolution(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  if (gpu_info.IsApiWebGpu() || gpu_info.IsApiOpenGl() ||
      gpu_info.IsApiVulkan()) {
    return SelectDWConvolutionDefault(attr, gpu_info, op_def, precision);
  }
  if (gpu_info.IsAdreno()) {
    return SelectDWConvolutionAdreno(attr, gpu_info, op_def, precision);
  } else if (gpu_info.IsApple()) {
    return SelectDWConvolutionApple(attr, gpu_info, op_def, precision);
  } else if (gpu_info.IsPowerVR()) {
    return SelectDWConvolutionPowerVR(attr, gpu_info, op_def, precision);
  } else {
    return SelectDWConvolutionDefault(attr, gpu_info, op_def, precision);
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionExternalWeights(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, CalculationsPrecision precision) {
  return std::make_unique<DepthwiseConv>(
      CreateDepthwiseConvolution2DExternalWeights(gpu_info, op_def, precision,
                                                  attr));
}

}  // namespace ml_drift
