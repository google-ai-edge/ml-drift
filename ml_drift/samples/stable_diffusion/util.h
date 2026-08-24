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

#ifndef ML_DRIFT_SAMPLES_STABLE_DIFFUSION_UTIL_H_
#define ML_DRIFT_SAMPLES_STABLE_DIFFUSION_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

void generateBitmapImage(unsigned char* image, int height, int width,
                         const std::string& file_name);

void GenerateImage(const TensorFloat32& tensor, const std::string& file_name);
TensorFloat32 ReadBMP(const std::string& file_name);
void GenerateIntermediateImage(const TensorFloat32& tensor,
                               const std::string& file_name,
                               const std::string& file_folder);

std::vector<half> LoadF16(const std::string& path, int count);

// Faster variant of LoadF16 without buffer initialization.
absl::Span<const half> LoadF16Ex(const std::string& path, int count,
                                 half* buffer);

Tensor<Linear, DataType::FLOAT32> CreateLinearTensor(
    absl::Span<const half> data);

GPUOperation CreateTembGenerationOp(const GpuInfo& gpu_info,
                                    const TensorDescriptor& dst,
                                    const std::string& file_folder);

GPUOperation CreateDiffusionStepOp(const TensorDescriptor& x_in,
                                   const TensorDescriptor& eta_uncond_in,
                                   const TensorDescriptor& eta_cond_in,
                                   const TensorDescriptor& dst);

Convolution2DAttributes MakeConvAttributes(absl::Span<const half> weights,
                                           absl::Span<const half> bias,
                                           const OHWI& shape, const HW& stride,
                                           const std::string& name);

std::pair<Convolution2DAttributes, Convolution2DAttributes> SplitAttributes(
    const Convolution2DAttributes& attributes);

DepthwiseConvolution2DAttributes MakeDwConvAttributes(
    absl::Span<const half> weights, absl::Span<const half> bias,
    const OHWI& shape, const HW& stride, const std::string& name);

absl::StatusOr<TensorFloat32> GenerateOpenClipMaskTensor(
    const Tensor<BHWC, DataType::INT32>& prompt_tensor);

// Do not use; for unit testing only.
namespace internal {
std::vector<float> ConvertLinearFp16ToFp32(absl::Span<const half> fp16s);
std::vector<float> ConvertOihwFp16ToOhwiFp32(absl::Span<const half> fp16s,
                                             const OHWI& shape);
}  // namespace internal
}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_STABLE_DIFFUSION_UTIL_H_
