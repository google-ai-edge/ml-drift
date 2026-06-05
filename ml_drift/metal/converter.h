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

#ifndef ML_DRIFT_METAL_CONVERTER_H_
#define ML_DRIFT_METAL_CONVERTER_H_

#import <Metal/Metal.h>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/metal/buffer.h"
#include "ml_drift/metal/compute_task.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {

class TensorToBHWCBufferConverter {
 public:
  TensorToBHWCBufferConverter() = default;
  TensorToBHWCBufferConverter(TensorToBHWCBufferConverter&& converter) =
      default;
  TensorToBHWCBufferConverter& operator=(
      TensorToBHWCBufferConverter&& converter) = default;
  TensorToBHWCBufferConverter(const TensorToBHWCBufferConverter&) = delete;
  TensorToBHWCBufferConverter& operator=(const TensorToBHWCBufferConverter&) =
      delete;

  absl::Status Init(Environment* env, const TensorDescriptor& src_desc,
                    DataType buffer_type);

  absl::Status Encode(id<MTLComputeCommandEncoder> encoder,
                      MetalSpatialTensor* src, Buffer* dst);
  absl::Status Encode(id<MTLComputeCommandEncoder> encoder,
                      MetalSpatialTensor* src, id<MTLBuffer> dst);

 private:
  ComputeTask task_;
};

class BHWCBufferToTensorConverter {
 public:
  BHWCBufferToTensorConverter() = default;
  BHWCBufferToTensorConverter(BHWCBufferToTensorConverter&& converter) =
      default;
  BHWCBufferToTensorConverter& operator=(
      BHWCBufferToTensorConverter&& converter) = default;
  BHWCBufferToTensorConverter(const TensorToBHWCBufferConverter&) = delete;
  BHWCBufferToTensorConverter& operator=(const BHWCBufferToTensorConverter&) =
      delete;

  absl::Status Init(Environment* env, DataType buffer_type,
                    const TensorDescriptor& dst_desc);

  absl::Status Encode(id<MTLComputeCommandEncoder> encoder, Buffer* src,
                      MetalSpatialTensor* dst);
  absl::Status Encode(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> src,
                      MetalSpatialTensor* dst);

 private:
  ComputeTask task_;
};

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_CONVERTER_H_
