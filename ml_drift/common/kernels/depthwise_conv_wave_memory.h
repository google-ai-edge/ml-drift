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

#ifndef ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_WAVE_MEMORY_H_
#define ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_WAVE_MEMORY_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/wave_memory_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class DepthwiseConvWaveMemory : public GPUOperation {
 public:
  DepthwiseConvWaveMemory() { work_group_size_ = int3(16, 8, 1); }
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  DepthwiseConvWaveMemory(DepthwiseConvWaveMemory&& kernel) = default;
  DepthwiseConvWaveMemory& operator=(DepthwiseConvWaveMemory&& kernel) =
      default;
  DepthwiseConvWaveMemory(const DepthwiseConvWaveMemory&) = delete;
  DepthwiseConvWaveMemory& operator=(const DepthwiseConvWaveMemory&) = delete;

 private:
  DepthwiseConvWaveMemory(const OperationDef& definition,
                          CalculationsPrecision precision,
                          const DepthwiseConvolution2DAttributes& attr,
                          const GpuInfo& gpu_info);
  template <DataType T>
  void UploadWeights(const GpuInfo& gpu_info, const Tensor<OHWI, T>& weights,
                     DataType dst_types);

  friend DepthwiseConvWaveMemory CreateDepthwiseConvWaveMemory(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const DepthwiseConvolution2DAttributes& attr);

  template <DataType S, typename T>
  void RearrangeWeightsData(const Tensor<OHWI, S>& weights, absl::Span<T> dst);

  int weights_cache_size_flt4_;
  int wave_size_ = 32;
};

template <DataType T>
void DepthwiseConvWaveMemory::UploadWeights(const GpuInfo& gpu_info,
                                            const Tensor<OHWI, T>& weights,
                                            DataType dst_type) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int flt4_per_layer =
      AlignByN(weights.shape.w * weights.shape.h, weights_cache_size_flt4_);

  const int flt4_count = flt4_per_layer * src_depth;

  BufferDescriptor desc = GetBufferDescForWaveMemoryUpload(gpu_info, dst_type);
  desc.size = SizeOf(dst_type) * 4 * flt4_count;
  desc.data.resize(desc.size);

  if (dst_type == DataType::kFloat32) {
    float4* ptr = reinterpret_cast<float4*>(desc.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(ptr, flt4_count));
  } else if (dst_type == DataType::kFloat16) {
    half4* ptr = reinterpret_cast<half4*>(desc.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(ptr, flt4_count));
  }

  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType S, typename T>
void DepthwiseConvWaveMemory::RearrangeWeightsData(
    const Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int flt4_per_layer =
      AlignByN(weights.shape.w * weights.shape.h, weights_cache_size_flt4_);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        T filter_val;
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + i;
          if (s_ch < weights.shape.i) {
            const int f_index = weights.shape.LinearIndex({0, y, x, s_ch});
            filter_val[i] = weights.data[f_index];
          } else {
            filter_val[i] = 0.0f;
          }
        }
        dst[counter++] = filter_val;
      }
    }

    T filter_val;
    filter_val[0] = 0.0f;
    filter_val[1] = 0.0f;
    filter_val[2] = 0.0f;
    filter_val[3] = 0.0f;
    for (int i = weights.shape.w * weights.shape.h; i < flt4_per_layer; ++i) {
      dst[counter++] = filter_val;
    }
  }
}

// Checks if the hardware supports depthwise convolution with wave memory.
bool HardwareSupportsDepthwiseConvWaveMemory(const GpuInfo& gpu_info);

// Checks if the depthwise convolution with wave memory is supported.
bool IsDepthwiseConvWaveMemorySupported(
    const GpuInfo& gpu_info, const DepthwiseConvolution2DAttributes& attr);

// Creates a depthwise convolution operation with wave memory.
DepthwiseConvWaveMemory CreateDepthwiseConvWaveMemory(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_WAVE_MEMORY_H_
