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

#ifndef ML_DRIFT_WEBGPU_CONVERTER_H_
#define ML_DRIFT_WEBGPU_CONVERTER_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/compute_task.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
// Implements conversion from tensor to another tensor.
class TensorToTensorConverter {
 public:
  absl::Status Init(const Environment& env, const TensorDescriptor& src_desc,
                    const TensorDescriptor& dst_desc);

  absl::Status Convert(const wgpu::Device& device,
                       wgpu::ComputePassEncoder compute_encoder,
                       SpatialTensor* src, SpatialTensor* dst);

 private:
  ComputeTask op_;
};

// Implements conversion from tensor to bhwc buffer.
// ! For dst_types less than 4 bytes, the buffer is required to be aligned to 4
// * sizeof(dst_types).
class TensorToBHWCBufferConverter {
 public:
  absl::Status Init(const Environment& env, const TensorDescriptor& src_desc,
                    DataType buffer_type);

  absl::Status Init(const Environment& env, const TensorDescriptor& src_desc,
                    const BufferDescriptor& dst_desc);

  absl::Status Convert(const wgpu::Device& device,
                       wgpu::ComputePassEncoder compute_encoder,
                       SpatialTensor* src, Buffer* dst);

 private:
  ComputeTask op_;
};

// Implements conversion from bhwc buffer to tensor.
// ! For src_types less than 4 bytes, the buffer is required to be aligned to 4
// * sizeof(src_type).
class BHWCBufferToTensorConverter {
 public:
  absl::Status Init(const Environment& env, DataType buffer_type,
                    const TensorDescriptor& dst_desc);
  absl::Status Init(const Environment& env, const BufferDescriptor& src_desc,
                    const TensorDescriptor& dst_desc);

  absl::Status Convert(const wgpu::Device& device,
                       wgpu::ComputePassEncoder compute_encoder, Buffer* src,
                       SpatialTensor* dst);

 private:
  ComputeTask op_;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_CONVERTER_H_
