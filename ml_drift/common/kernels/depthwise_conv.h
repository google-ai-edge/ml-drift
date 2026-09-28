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

#ifndef ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_H_
#define ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
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
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

class DepthwiseConv : public GPUOperation {
 public:
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  DepthwiseConv(DepthwiseConv&& operation) = default;
  DepthwiseConv& operator=(DepthwiseConv&& operation) = default;
  DepthwiseConv(const DepthwiseConv&) = delete;
  DepthwiseConv& operator=(const DepthwiseConv&) = delete;

  friend DepthwiseConv CreateDepthwiseConvolution2D(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const DepthwiseConvolution2DAttributes& attr);

  friend DepthwiseConv CreateDepthwiseConvolution2DExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const DepthwiseConvolution2DAttributes& attr);

  friend DepthwiseConv CreateDepthwiseConvolution3D(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const DepthwiseConvolution3DAttributes& attr);

 private:
  struct DepthwiseConvParams {
    bool UseLocalMem() const {
      return use_weights_caching || use_spatial_caching;
    }
    int GetKernelsTotalSize() const {
      return x_kernel_size * y_kernel_size * z_kernel_size;
    }
    int GetWorkGroupTotalSize() const {
      return work_group_size.x * work_group_size.y * work_group_size.z;
    }
    int channel_multiplier;
    // Supportd only tensors with Width & Height spatial dimensions
    // optional, if true, spatial dims will be uploaded to local mem
    bool use_spatial_caching = false;
    // optional, if true, weights will be uploaded to local memory
    bool use_weights_caching = false;
    // optional, if UsesLocalMem() return true this field must be initialized
    int3 work_group_size = int3(1, 1, 1);

    // optional, if UsesLocalMem() return true this field must be initialized
    int x_kernel_size = 1;
    // optional, if UsesLocalMem() return true this field must be initialized
    int y_kernel_size = 1;
    // optional, if UsesLocalMem() return true this field must be initialized
    int z_kernel_size = 1;

    // optional, if use_spatial_caching true this field must be initialized
    int x_dilation_size = 1;
    // optional, if use_spatial_caching true this field must be initialized
    int y_dilation_size = 1;
    // optional, if use_spatial_caching true this field must be initialized
    int z_dilation_size = 1;
  };

  explicit DepthwiseConv(const OperationDef& definition,
                         const DepthwiseConvParams& params);

  std::string GenerateSrcUpload(const GpuInfo& gpu_info,
                                const OperationDef& op_def);
  std::string GenerateWeightsUpload(const GpuInfo& gpu_info);
  std::string GenerateCode(const GpuInfo& gpu_info, const OperationDef& op_def,
                           CalculationsPrecision precision, bool has_bias);

  void UploadWeightsForDWConv2D(const Tensor<OHWI, DataType::kFloat32>& weights,
                                DataType dst_type, bool weights_are_buffer);

  template <DataType T>
  void UploadWeightsForDWConv3D(const Tensor<OHWDI, T>& weights,
                                DataType dst_type, bool weights_are_buffer);

  DepthwiseConvParams params_;
};

template <typename T>
void RearrangeWeightsForDWConv2D(
    const Tensor<OHWI, DataType::kFloat32>& weights, absl::Span<T> dst) {
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int elements_count = kernel_x * kernel_y * dst_slices;
  ABSL_CHECK_EQ(elements_count, dst.size());

  std::unique_ptr<half[]> half_weights;
  if (std::is_same<T, half4>::value) {
    half_weights = ConvertF32F16(weights.data);
  }

  int counter = 0;
  for (int d = 0; d < dst_slices; ++d) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        T filter_val;
        for (int i = 0; i < 4; ++i) {
          const int d_ch = d * 4 + i;
          if (d_ch < dst_channels) {
            const int f_index = weights.shape.LinearIndex(
                {d_ch % weights.shape.o, y, x, d_ch / weights.shape.o});
            if (std::is_same<T, half4>::value) {
              filter_val[i] = half_weights[f_index];
            } else {
              filter_val[i] = weights.data[f_index];
            }
          } else {
            filter_val[i] = 0.0f;
          }
        }
        dst[counter++] = filter_val;
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsForDWConv3D(const Tensor<OHWDI, S>& weights,
                                 absl::Span<T> dst) {
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;
  const int kernel_z = weights.shape.d;

  int counter = 0;
  for (int d = 0; d < dst_slices; ++d) {
    for (int z = 0; z < kernel_z; ++z) {
      for (int y = 0; y < kernel_y; ++y) {
        for (int x = 0; x < kernel_x; ++x) {
          T filter_val;
          for (int i = 0; i < 4; ++i) {
            const int d_ch = d * 4 + i;
            if (d_ch < dst_channels) {
              const int f_index = weights.shape.LinearIndex(
                  {d_ch % weights.shape.o, y, x, z, d_ch / weights.shape.o});
              filter_val[i] = weights.data[f_index];
            } else {
              filter_val[i] = 0.0f;
            }
          }
          dst[counter++] = filter_val;
        }
      }
    }
  }
}

template <DataType T>
void DepthwiseConv::UploadWeightsForDWConv3D(const Tensor<OHWDI, T>& weights,
                                             DataType dst_type,
                                             bool weights_are_buffer) {
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;
  const int kernel_z = weights.shape.d;

  const int elements_count = kernel_x * kernel_y * kernel_z * dst_slices;

  std::vector<uint8_t> data(SizeOf(dst_type) * 4 * elements_count);

  if (dst_type == DataType::kFloat32) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    RearrangeWeightsForDWConv3D(weights, absl::MakeSpan(ptr, elements_count));
  } else if (dst_type == DataType::kFloat16) {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    RearrangeWeightsForDWConv3D(weights, absl::MakeSpan(ptr, elements_count));
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
        dst_type, TensorStorageType::kTexture2D, kernel_x * kernel_y * kernel_z,
        dst_slices, data.data());
    args_.AddObject("weights", std::make_unique<TensorDescriptor>(desc));
  }
}

// Creates a depthwise convolution operation.
DepthwiseConv CreateDepthwiseConvolution2D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr);

// Creates a depthwise convolution operation with external weights.
DepthwiseConv CreateDepthwiseConvolution2DExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr);

// Creates a 3D depthwise convolution operation.
DepthwiseConv CreateDepthwiseConvolution3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution3DAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_DEPTHWISE_CONV_H_
