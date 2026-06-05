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

#ifndef ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_3X3_H_
#define ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_3X3_H_

#include <cstdint>
#include <memory>
#include <string>
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
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class DepthwiseConv3x3 : public GPUOperation {
 public:
  DepthwiseConv3x3() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;
  int3 GetGridSize() const override;

  // Move only
  DepthwiseConv3x3(DepthwiseConv3x3&& operation) = default;
  DepthwiseConv3x3& operator=(DepthwiseConv3x3&& operation) = default;
  DepthwiseConv3x3(const DepthwiseConv3x3&) = delete;
  DepthwiseConv3x3& operator=(const DepthwiseConv3x3&) = delete;

 private:
  explicit DepthwiseConv3x3(const OperationDef& definition,
                            CalculationsPrecision precision,
                            bool weights_are_buffer, bool local_mem_uploads,
                            const GpuInfo& gpu_info);
  template <DataType T>
  void UploadWeightsAndBiases(const Tensor<OHWI, T>& weights,
                              const Tensor<Linear, T>& biases,
                              DataType dst_type, bool weights_are_buffer);

  friend DepthwiseConv3x3 CreateDepthwiseConv3x3(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const DepthwiseConvolution2DAttributes& attr);

  template <DataType S, typename T>
  void RearrangeWeightsAndBiasesData(const Tensor<OHWI, S>& weights,
                                     const Tensor<Linear, S>& biases,
                                     absl::Span<T> dst);

  std::string GenerateDepthwiseConvCode(const GpuInfo& gpu_info,
                                        const OperationDef& op_def,
                                        CalculationsPrecision precision,
                                        bool weights_are_buffer,
                                        bool local_mem_uploads);

  bool local_mem_uploads_;
};

template <DataType T>
void DepthwiseConv3x3::UploadWeightsAndBiases(const Tensor<OHWI, T>& weights,
                                              const Tensor<Linear, T>& biases,
                                              DataType dst_type,
                                              bool weights_are_buffer) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  int texture_width = 10;  // 3x3 kernel + 1 bias
  int texture_height = src_depth;
  const int elements_count = texture_width * texture_height;

  std::vector<uint8_t> data(SizeOf(dst_type) * 4 * elements_count);
  if (dst_type == DataType::FLOAT32) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    RearrangeWeightsAndBiasesData(weights, biases,
                                  absl::MakeSpan(ptr, elements_count));
  } else if (dst_type == DataType::FLOAT16) {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    RearrangeWeightsAndBiasesData(weights, biases,
                                  absl::MakeSpan(ptr, elements_count));
  }

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = dst_type;
    desc.element_size = 4;
    desc.size = SizeOf(dst_type) * 4 * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
        dst_type, TensorStorageType::TEXTURE_2D, texture_width, texture_height,
        data.data());
    args_.AddObject("weights", std::make_unique<TensorDescriptor>(desc));
  }
}

template <DataType S, typename T>
void DepthwiseConv3x3::RearrangeWeightsAndBiasesData(
    const Tensor<OHWI, S>& weights, const Tensor<Linear, S>& biases,
    absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
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

    T bias_val;
    for (int i = 0; i < 4; ++i) {
      const int dst_ch = s * 4 + i;
      bias_val[i] = biases.data.empty() || dst_ch >= biases.shape.v
                        ? 0.0f
                        : biases.data[dst_ch];
    }
    dst[counter++] = bias_val;
  }
}

// Checks if the 3x3 depthwise convolution is supported on the given GPU.
bool IsDepthwiseConv3x3Supported(const GpuInfo& gpu_info,
                                 const DepthwiseConvolution2DAttributes& attr);

// Creates a 3x3 depthwise convolution operation.
DepthwiseConv3x3 CreateDepthwiseConv3x3(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_3X3_H_
