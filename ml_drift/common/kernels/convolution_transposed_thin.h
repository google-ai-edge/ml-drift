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

#ifndef ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_THIN_H_
#define ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_THIN_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvolutionTransposedThin : public GPUOperation {
 public:
  ConvolutionTransposedThin() = default;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposedThin(ConvolutionTransposedThin&& operation);
  ConvolutionTransposedThin& operator=(ConvolutionTransposedThin&& operation);
  ConvolutionTransposedThin(const ConvolutionTransposedThin&) = delete;
  ConvolutionTransposedThin& operator=(const ConvolutionTransposedThin&) =
      delete;

 private:
  friend ConvolutionTransposedThin CreateConvolutionTransposedThin(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvolutionTransposedAttributes& attr);
  ConvolutionTransposedThin(const OperationDef& definition,
                            CalculationsPrecision precision,
                            const ConvolutionTransposedAttributes& attr,
                            const GpuInfo& gpu_info);
  template <DataType T>
  void UploadData(const GpuInfo& gpu_info, const Tensor<OHWI, T>& weights,
                  const Tensor<Linear, T>& biases, DataType dst_type);

  template <DataType S, typename T>
  void RearrangeWeightsData(const Tensor<OHWI, S>& weights, absl::Span<T> dst);
  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                CalculationsPrecision precision,
                                                int src_depth, int dst_channels,
                                                const int2& kernel_size);
};

template <DataType T>
void ConvolutionTransposedThin::UploadData(const GpuInfo& gpu_info,
                                           const Tensor<OHWI, T>& weights,
                                           const Tensor<Linear, T>& biases,
                                           DataType dst_type) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int flt4_count =
      weights.shape.w * weights.shape.h * src_depth * weights.shape.o;

  BufferDescriptor desc;
  desc.element_type = dst_type;
  desc.element_size = 4;
  desc.memory_type =
      gpu_info.IsApiWebGpu() ? MemoryType::kGlobal : MemoryType::kConstant;
  desc.size = SizeOf(dst_type) * 4 * (flt4_count + 1);
  desc.data.resize(desc.size);

  if (dst_type == DataType::kFloat32) {
    float4* gpu_data = reinterpret_cast<float4*>(desc.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data, flt4_count));
    float4 bias_value(0.0f);
    for (int i = 0; i < weights.shape.o; ++i) {
      bias_value[i] = biases.data.empty() ? 0.0f : biases.data[i];
    }
    gpu_data[flt4_count] = bias_value;
  } else if (dst_type == DataType::kFloat16) {
    half4* gpu_data = reinterpret_cast<half4*>(desc.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data, flt4_count));
    half4 bias_value(0.0f);
    for (int i = 0; i < weights.shape.o; ++i) {
      bias_value[i] = biases.data.empty() ? 0.0f : biases.data[i];
    }
    gpu_data[flt4_count] = bias_value;
  }

  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType S, typename T>
void ConvolutionTransposedThin::RearrangeWeightsData(
    const Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        std::vector<T> filters(weights.shape.o);
        for (int j = 0; j < weights.shape.o; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_ch = s * 4 + i;
            const int d_ch = j;
            if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
              const int f_index = weights.shape.LinearIndex({d_ch, y, x, s_ch});
              filters[j][i] = weights.data[f_index];
            } else {
              filters[j][i] = 0.0f;
            }
          }
        }
        for (int j = 0; j < weights.shape.o; ++j) {
          dst[counter++] = filters[j];
        }
      }
    }
  }
}

// Checks if the thin transposed convolution is supported.
bool IsConvolutionTransposedThinSupported(
    const ConvolutionTransposedAttributes& attr);

// Creates a thin transposed convolution operation.
ConvolutionTransposedThin CreateConvolutionTransposedThin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONVOLUTION_TRANSPOSED_THIN_H_
