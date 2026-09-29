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

#include <stddef.h>

#include <memory>
#include <utility>
#include <variant>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_apple_mpp.h"
#include "ml_drift/common/kernels/conv_constants.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_wave_matrix.h"
#include "ml_drift/common/kernels/conv_wave_matrix_mali.h"
#include "ml_drift/common/kernels/conv_wave_memory.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {
bool CanUseConvConstants(bool different_weights_for_height,
                         const TensorDescriptor* src_exp,
                         const ConvRuntimeCheckDesc& runtime_check) {
  return !different_weights_for_height && !src_exp &&
         !runtime_check.HasValues();
}

std::unique_ptr<GPUOperation> SelectConvolutionAdreno(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints) {
  if (hints.Check(ModelHints::kReduceKernelsCount)) {
    if (IsConvWaveMemorySupported(gpu_info)) {
      ConvWaveMemory conv =
          CreateConvWaveMemory(gpu_info, op_def, precision, attr);
      return std::make_unique<ConvWaveMemory>(std::move(conv));
    }
  }
  if (IsConvConstantsSupported(gpu_info, precision, attr)) {
    ConvConstants conv = CreateConvConstants(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvConstants>(std::move(conv));
  } else if (IsConvWaveMemorySupported(gpu_info)) {
    ConvWaveMemory conv =
        CreateConvWaveMemory(gpu_info, op_def, precision, attr, &dst_shape);
    return std::make_unique<ConvWaveMemory>(std::move(conv));
  } else {
    ConvGeneric conv =
        CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionExternalWeightsAdreno(
    const Convolution2DAttributes& attr, const TensorDescriptor* bias_desc,
    const BHWC& dst_shape, const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints,
    WeightsDescription* weights_desc, const TensorDescriptor* src_exp,
    bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  // TODO(sorokin): driver bug? workaround for b/424603203, may need additional
  // clarification
  const bool constant_buffer_sync_bug =
      gpu_info.IsApiOpenCl() && gpu_info.adreno_info.IsAdreno8xx();
  if (!constant_buffer_sync_bug &&
      IsConvConstantsSupported(gpu_info, precision, attr) &&
      CanUseConvConstants(different_weights_for_height, src_exp,
                          runtime_check)) {
    ConvConstants conv = CreateConvConstantsExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvConstants>(std::move(conv));
  } else if (IsConvWaveMemorySupported(gpu_info)) {
    ConvWaveMemory convolution = CreateConvWaveMemoryExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = convolution.GetWeightsDescription();
    return std::make_unique<ConvWaveMemory>(std::move(convolution));
  } else {
    ConvGeneric conv = CreateConvGenericExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionApple(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision) {
  if (SupportsConvWaveMatrix(gpu_info, precision, attr) &&
      IsGoodTaskSizeForAppleConvSimd(
          dst_shape,
          std::visit([](const auto& w) { return w.shape; }, attr.weights),
          precision, gpu_info)) {
    ConvWaveMatrix conv =
        CreateConvWaveMatrix(op_def, precision, dst_shape, attr, gpu_info);
    return std::make_unique<ConvWaveMatrix>(std::move(conv));
  } else {
    ConvGeneric conv =
        CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionExternalWeightsApple(
    const Convolution2DAttributes& attr, const TensorDescriptor* bias_desc,
    const BHWC& dst_shape, const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints,
    WeightsDescription* weights_desc, const TensorDescriptor* src_exp,
    bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  if (SupportsConvWaveMatrix(gpu_info, precision, attr) &&
      IsGoodTaskSizeForAppleConvSimd(
          dst_shape,
          std::visit([](const auto& w) { return w.shape; }, attr.weights),
          precision, gpu_info)) {
    ConvWaveMatrix conv = CreateConvWaveMatrixExternalWeights(
        op_def, precision, dst_shape, attr, gpu_info, bias_desc, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvWaveMatrix>(std::move(conv));
  } else {
    ConvGeneric conv = CreateConvGenericExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionMali(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision) {
  const size_t threads_count = dst_shape.w * dst_shape.h * dst_shape.b;
  if (threads_count >= 1024 * 16 && dst_shape.c <= 4 &&
      IsConvConstantsSupported(gpu_info, precision, attr)) {
    ConvConstants conv = CreateConvConstants(gpu_info, op_def, precision, attr);
    return std::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvGeneric conv =
        CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionExternalWeightsMali(
    const Convolution2DAttributes& attr, const TensorDescriptor* bias_desc,
    const BHWC& dst_shape, const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints,
    WeightsDescription* weights_desc, const TensorDescriptor* src_exp,
    bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  const size_t threads_count = dst_shape.w * dst_shape.h * dst_shape.b;
  if (threads_count >= 1024 * 16 && dst_shape.c <= 4 &&
      IsConvConstantsSupported(gpu_info, precision, attr) &&
      CanUseConvConstants(different_weights_for_height, src_exp,
                          runtime_check)) {
    ConvConstants conv = CreateConvConstantsExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvGeneric conv = CreateConvGenericExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

}  // namespace

std::unique_ptr<GPUOperation> SelectConvolution(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints) {
  if (gpu_info.IsApple()) {
    return SelectConvolutionApple(attr, dst_shape, gpu_info, op_def, precision);
  }
  if (gpu_info.IsApiOpenGl() || gpu_info.IsApiVulkan() ||
      gpu_info.IsApiWebGpu()) {
    if (gpu_info.IsAMD() &&
        IsConvConstantsSupported(gpu_info, precision, attr)) {
      const size_t threads_count = dst_shape.w * dst_shape.h * dst_shape.b;
      float threads_per_cu = threads_count / gpu_info.GetComputeUnitsCount();
      float waves_per_cu = threads_per_cu / 64;
      if (waves_per_cu >= 8 || dst_shape.c <= 4) {
        ConvConstants conv =
            CreateConvConstants(gpu_info, op_def, precision, attr);
        return std::make_unique<ConvConstants>(std::move(conv));
      }
    }
    if (SupportsConvWaveMatrix(gpu_info, precision, attr)) {
      return std::make_unique<ConvWaveMatrix>(
          CreateConvWaveMatrix(op_def, precision, dst_shape, attr, gpu_info));
    }
    ConvGeneric conv =
        CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
  if (gpu_info.IsAdreno()) {
    return SelectConvolutionAdreno(attr, dst_shape, gpu_info, op_def, precision,
                                   hints);
  } else if (gpu_info.IsMali()) {
    return SelectConvolutionMali(attr, dst_shape, gpu_info, op_def, precision);
  } else if (gpu_info.IsPowerVR() &&
             gpu_info.SupportsExtension("cl_img_pixel_subgroup_dot") &&
             IsConvWaveMemorySupported(gpu_info)) {
    ConvWaveMemory conv =
        CreateConvWaveMemory(gpu_info, op_def, precision, attr, &dst_shape);
    return std::make_unique<ConvWaveMemory>(std::move(conv));
  } else if (gpu_info.IsNvidia() || gpu_info.IsPowerVR() || gpu_info.IsAMD()) {
    if (IsConvConstantsSupported(gpu_info, precision, attr) &&
        !hints.Check(ModelHints::kReduceKernelsCount)) {
      ConvConstants conv =
          CreateConvConstants(gpu_info, op_def, precision, attr);
      return std::make_unique<ConvConstants>(std::move(conv));
    } else {
      ConvGeneric conv =
          CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape);
      return std::make_unique<ConvGeneric>(std::move(conv));
    }
  } else {
    ConvGeneric conv =
        CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionWithExternalWeights(
    const Convolution2DAttributes& attr, const TensorDescriptor* bias_desc,
    const BHWC& dst_shape, const GpuInfo& gpu_info, const OperationDef& op_def,
    CalculationsPrecision precision, ModelHints hints,
    WeightsDescription* weights_desc, const TensorDescriptor* src_exp,
    bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  if (gpu_info.IsApple()) {
    return SelectConvolutionExternalWeightsApple(
        attr, bias_desc, dst_shape, gpu_info, op_def, precision, hints,
        weights_desc, src_exp, different_weights_for_height, runtime_check);
  }
  if (gpu_info.IsApiOpenGl() || gpu_info.IsApiVulkan() ||
      gpu_info.IsApiWebGpu()) {
    if (gpu_info.IsAMD() &&
        IsConvConstantsSupported(gpu_info, precision, attr) &&
        CanUseConvConstants(different_weights_for_height, src_exp,
                            runtime_check)) {
      const size_t threads_count = dst_shape.w * dst_shape.h * dst_shape.b;
      float threads_per_cu = threads_count / gpu_info.GetComputeUnitsCount();
      float waves_per_cu = threads_per_cu / 64;
      if (waves_per_cu >= 8 || dst_shape.c <= 4) {
        ConvConstants conv = CreateConvConstantsExternalWeights(
            gpu_info, op_def, precision, attr, bias_desc);
        *weights_desc = conv.GetWeightsDescription();
        return std::make_unique<ConvConstants>(std::move(conv));
      }
    }
    if (SupportsConvWaveMatrix(gpu_info, precision, attr)) {
      ConvWaveMatrix conv = CreateConvWaveMatrixExternalWeights(
          op_def, precision, dst_shape, attr, gpu_info, bias_desc, src_exp,
          different_weights_for_height, runtime_check);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvWaveMatrix>(std::move(conv));
    }
    ConvGeneric conv = CreateConvGenericExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
  if (gpu_info.IsAdreno()) {
    return SelectConvolutionExternalWeightsAdreno(
        attr, bias_desc, dst_shape, gpu_info, op_def, precision, hints,
        weights_desc, src_exp, different_weights_for_height, runtime_check);
  } else if (gpu_info.IsMali()) {
    return SelectConvolutionExternalWeightsMali(
        attr, bias_desc, dst_shape, gpu_info, op_def, precision, hints,
        weights_desc, src_exp, different_weights_for_height, runtime_check);
  } else if (gpu_info.IsPowerVR() &&
             gpu_info.SupportsExtension("cl_img_pixel_subgroup_dot") &&
             IsConvWaveMemorySupported(gpu_info)) {
    ConvWaveMemory convolution = CreateConvWaveMemoryExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = convolution.GetWeightsDescription();
    return std::make_unique<ConvWaveMemory>(std::move(convolution));
  } else if (gpu_info.IsNvidia() || gpu_info.IsPowerVR() || gpu_info.IsAMD()) {
    if (IsConvConstantsSupported(gpu_info, precision, attr) &&
        !hints.Check(ModelHints::kReduceKernelsCount) &&
        CanUseConvConstants(different_weights_for_height, src_exp,
                            runtime_check)) {
      ConvConstants conv = CreateConvConstantsExternalWeights(
          gpu_info, op_def, precision, attr, bias_desc);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvConstants>(std::move(conv));
    } else {
      ConvGeneric conv = CreateConvGenericExternalWeights(
          gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
          different_weights_for_height, runtime_check);
      *weights_desc = conv.GetWeightsDescription();
      return std::make_unique<ConvGeneric>(std::move(conv));
    }
  } else {
    ConvGeneric conv = CreateConvGenericExternalWeights(
        gpu_info, op_def, precision, attr, bias_desc, &dst_shape, src_exp,
        different_weights_for_height, runtime_check);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
  }
}

bool SupportsConvolutionInt8(const GpuInfo& gpu_info, const BHWC& src_shape) {
  return SupportsConvAppleMPP(gpu_info) ||
         SupportsConvWaveMatrixMaliInt8(gpu_info, src_shape) ||
         SupportsConvWaveMemoryInt8(gpu_info) ||
         SupportsConvGenericInt8(gpu_info) ||
         SupportsConvWaveMatrixInt8(gpu_info, OHWI(32, 1, 1, src_shape.c));
}

PackedType GetConvolutionInt8SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape) {
  if (SupportsConvAppleMPP(gpu_info)) {
    return PackedType::kInt8C4;
  } else if (SupportsConvWaveMatrixMaliInt8(gpu_info, src_shape)) {
    return PackedType::kInt8W4C4;
  } else if (SupportsConvWaveMatrixInt8(gpu_info,
                                        OHWI(32, 1, 1, src_shape.c))) {
    return GetConvWaveMatrixInt8SrcType();
  } else if (SupportsConvWaveMemoryInt8(gpu_info)) {
    return GetConvWaveMemoryInt8SrcType(gpu_info, src_shape);
  } else if (SupportsConvGenericInt8(gpu_info)) {
    return GetConvGenericInt8SrcType(gpu_info, src_shape);
  } else {
    return PackedType::kUnknown;
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionInt8(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    PackedType src_packed_type, const Tensor<OHWI, DataType::kInt8>& weights,
    const BHWC& dst_shape) {
  BHWC src_shape = dst_shape;
  src_shape.c = weights.shape.i;
  if (SupportsConvAppleMPP(gpu_info)) {
    auto conv = CreateConvAppleMPPInt8(op_def.src_tensors[0],
                                       op_def.dst_tensors[0], weights);
    return std::make_unique<ConvAppleMPP>(std::move(conv));
  } else if (SupportsConvWaveMatrixMaliInt8(gpu_info, src_shape)) {
    auto conv = ConvWaveMatrixMali(gpu_info, op_def, weights);
    return std::make_unique<ConvWaveMatrixMali>(std::move(conv));
  } else if (SupportsConvWaveMatrixInt8(gpu_info,
                                        OHWI(32, 1, 1, src_shape.c))) {
    auto conv = CreateConvWaveMatrixInt8(op_def, dst_shape, weights, gpu_info);
    return std::make_unique<ConvWaveMatrix>(std::move(conv));
  } else if (SupportsConvWaveMemoryInt8(gpu_info)) {
    auto conv = CreateConvWaveMemoryInt8(gpu_info, op_def, weights, &dst_shape);
    return std::make_unique<ConvWaveMemory>(std::move(conv));
  } else if (SupportsConvGenericInt8(gpu_info)) {
    auto conv = CreateConvGenericInt8(gpu_info, op_def, src_packed_type,
                                      weights, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    return nullptr;
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionInt8(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    PackedType src_packed_type, const OHWI& weights_shape,
    const BHWC& dst_shape, WeightsDescription* weights_desc) {
  BHWC src_shape = dst_shape;
  src_shape.c = weights_shape.i;
  if (SupportsConvAppleMPP(gpu_info)) {
    auto conv = CreateConvAppleMPPInt8(op_def.src_tensors[0],
                                       op_def.dst_tensors[0], weights_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvAppleMPP>(std::move(conv));
  } else if (SupportsConvWaveMatrixMaliInt8(gpu_info, src_shape)) {
    auto conv = ConvWaveMatrixMali(gpu_info, op_def, weights_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvWaveMatrixMali>(std::move(conv));
  } else if (SupportsConvWaveMatrixInt8(gpu_info,
                                        OHWI(32, 1, 1, src_shape.c))) {
    auto conv = CreateConvWaveMatrixInt8ExternalWeights(
        gpu_info, op_def, weights_shape, dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvWaveMatrix>(std::move(conv));
  } else if (SupportsConvWaveMemoryInt8(gpu_info)) {
    auto conv = CreateConvWaveMemoryInt8ExternalWeights(
        gpu_info, op_def, weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvWaveMemory>(std::move(conv));
  } else if (SupportsConvGenericInt8(gpu_info)) {
    auto conv = CreateConvGenericInt8ExternalWeights(
        gpu_info, op_def, src_packed_type, weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    return nullptr;
  }
}

bool SupportsConvolutionInt4(const GpuInfo& gpu_info, const BHWC& src_shape) {
  return SupportsConvGenericInt4(gpu_info, src_shape);
}

PackedType GetConvolutionInt4SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape) {
  if (SupportsConvGenericInt4(gpu_info, src_shape)) {
    return GetConvGenericInt4SrcType(gpu_info, src_shape);
  } else {
    return PackedType::kUnknown;
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionInt4(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    const OHWI& weights_shape, const BHWC& dst_shape,
    WeightsDescription* weights_desc) {
  BHWC src_shape = dst_shape;
  src_shape.c = weights_shape.i;
  if (SupportsConvGenericInt4(gpu_info, src_shape)) {
    auto conv = CreateConvGenericInt4ExternalWeights(gpu_info, op_def,
                                                     weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    return nullptr;
  }
}

}  // namespace ml_drift
