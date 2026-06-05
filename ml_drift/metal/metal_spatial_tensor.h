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

#ifndef ML_DRIFT_METAL_METAL_SPATIAL_TENSOR_H_
#define ML_DRIFT_METAL_METAL_SPATIAL_TENSOR_H_

#import <Metal/Metal.h>

#include <cstdint>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/metal/gpu_object.h"

namespace ml_drift {
namespace metal {

class MetalSpatialTensor : public GPUObject, public GpuSpatialTensor {
 public:
  MetalSpatialTensor()
      : memory_(nullptr),
        texture_mem_(nullptr),
        memory_owner_(true),
        texture_mem_owner_(true) {}
  MetalSpatialTensor(id<MTLBuffer> buffer, id<MTLTexture> texture,
                     bool memory_owner, bool texture_mem_owner,
                     const TensorDescriptor& descriptor);

  // Move only
  MetalSpatialTensor(MetalSpatialTensor&& tensor);
  MetalSpatialTensor& operator=(MetalSpatialTensor&& tensor);
  MetalSpatialTensor(const MetalSpatialTensor&) = delete;
  MetalSpatialTensor& operator=(const MetalSpatialTensor&) = delete;

  ~MetalSpatialTensor() override { Release(); }

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  int Width() const override { return descriptor_.GetBHWDCShape().w; }
  int Height() const override { return descriptor_.GetBHWDCShape().h; }
  int Depth() const override { return descriptor_.GetBHWDCShape().d; }
  int Channels() const override { return descriptor_.GetBHWDCShape().c; }
  int Slices() const override {
    return DivideRoundUp(descriptor_.GetBHWDCShape().c, 4);
  }
  int Batch() const override { return descriptor_.GetBHWDCShape().b; }

  TensorDescriptor GetDescriptor() const override { return descriptor_; }
  DataType GetDataType() const { return descriptor_.GetDataType(); }
  TensorStorageType GetStorageType() const {
    return descriptor_.GetStorageType();
  }
  uint64_t GetMemorySizeInBytes() const {
    return descriptor_.GetMemorySizeInBytes();
  }

  absl::Status CreateFromDescriptor(const TensorDescriptor& desc,
                                    id<MTLDevice> device);
  absl::Status UploadDescriptorData(const TensorDescriptor& desc,
                                    id<MTLDevice> device);
  absl::Status UploadDescriptorData(const TensorDescriptor& desc,
                                    id<MTLCommandQueue> command_queue,
                                    bool wait_for_completion);
  absl::Status ToDescriptor(TensorDescriptor* desc, id<MTLDevice> device) const;

  absl::Status SetBufferHandle(id<MTLBuffer> buffer);
  id<MTLBuffer> GetBufferHandle() const { return memory_; }
  id<MTLTexture> GetTextureHandle() const { return texture_mem_; }
  void* GetBufferPointer() const {
    return static_cast<uint8_t*>([memory_ contents]) + buffer_offset_;
  }

  // This method uses Metal's write buffer API to copy data into the tensor.
  // Typically, the data pointed by `ptr` has been rearranged to be in the data
  // layout of the TensorDescriptor which was used for tensor creation.
  absl::Status WriteData(id<MTLCommandQueue> command_queue, const void* ptr,
                         bool wait_for_completion);

  void Release();

 private:
  friend absl::Status CreateTensorSharedBuffer(
      id<MTLBuffer> buffer, const TensorDescriptor& descriptor,
      MetalSpatialTensor* result, uint64_t buffer_offset);

  friend absl::Status CreateTensorSharedImage2DBuffer(
      id<MTLBuffer> buffer, const TensorDescriptor& descriptor,
      int row_bytes_alignment, MetalSpatialTensor* result,
      uint64_t buffer_offset);

  absl::Status ReadData(id<MTLDevice> device, void* ptr) const;

  id<MTLBuffer> memory_;
  id<MTLTexture> texture_mem_;
  bool memory_owner_;
  bool texture_mem_owner_;
  TensorDescriptor descriptor_;
  // for use with TEXTURE_2D and when texture created from buffer.
  int aligned_texture_width_;
  // used when created from shared buffer
  uint64_t buffer_offset_ = 0;
};

absl::Status CreateTensor(id<MTLDevice> device,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateTensorSharedBuffer(id<MTLBuffer> buffer,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result,
                                      uint64_t buffer_offset = 0);

absl::Status CreateTensorSharedTexture(id<MTLTexture> texture,
                                       const TensorDescriptor& descriptor,
                                       MetalSpatialTensor* result);

absl::Status CreateTensorSharedImage2DBuffer(id<MTLBuffer> buffer,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment,
                                             MetalSpatialTensor* result,
                                             uint64_t buffer_offset = 0);

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_METAL_SPATIAL_TENSOR_H_
