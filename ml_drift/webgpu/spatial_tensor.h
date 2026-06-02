// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_WEBGPU_SPATIAL_TENSOR_H_
#define ML_DRIFT_WEBGPU_SPATIAL_TENSOR_H_

#include <cstddef>
#include <cstdint>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

class SpatialTensor : public GpuObject, public GpuSpatialTensor {
 public:
  bool IsSpatialTensor() const override { return true; }
  struct SpatialTensorCreateInfo {
    wgpu::Buffer buffer;
    wgpu::Texture texture;
    wgpu::TextureView texture_view;
    bool memory_owner;
    size_t offset = 0;
  };
  SpatialTensor() = default;
  SpatialTensor(const SpatialTensorCreateInfo& info,
                const TensorDescriptor& descriptor);

  // Move only
  SpatialTensor(SpatialTensor&& tensor);
  SpatialTensor& operator=(SpatialTensor&& tensor);
  SpatialTensor(const SpatialTensor&) = delete;
  SpatialTensor& operator=(const SpatialTensor&) = delete;

  ~SpatialTensor() override { Release(); }

  // memory can be invalid(nullptr) for example for shared tensors.
  bool IsValidMemory() const;

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GpuResourcesWithValue* resources) const override;

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

  wgpu::Buffer GetBufferHandle() const { return buffer_; }
  wgpu::Texture GetTextureHandle() const { return texture_; }
  wgpu::TextureView GetTextureViewHandle() const { return texture_view_; }

  absl::Status CreateFromDescriptor(const wgpu::Device& device,
                                    const TensorDescriptor& desc);
  absl::Status CreateFromBuffer(const wgpu::Buffer& buffer, size_t offset,
                                const TensorDescriptor& desc);
  absl::Status UploadDescriptorData(const wgpu::Queue& queue,
                                    const TensorDescriptor& desc);
  absl::Status ToDescriptor(const wgpu::Device& device,
                            TensorDescriptor* desc) const;

  size_t offset() const { return offset_; }

  // This method uses WebGPU's direct `Write{Buffer|Texture}`-style transfer.
  // Typically, the data pointed by `ptr` has been rearranged to be in the data
  // layout of the TensorDescriptor which was used for tensor creation.
  absl::Status WriteData(const wgpu::Queue& queue, const void* ptr);
  absl::Status WriteDataViaStaging(const Environment& env, const void* ptr);
  void Release();

 private:
  absl::Status ReadData(const wgpu::Device& device, void* ptr) const;

  wgpu::Buffer buffer_ = nullptr;
  wgpu::Texture texture_ = nullptr;
  wgpu::TextureView texture_view_ = nullptr;
  bool memory_owner_;
  TensorDescriptor descriptor_;
  size_t offset_ = 0;
};

absl::Status CreateTensor(const wgpu::Device& device,
                          const TensorDescriptor& descriptor,
                          SpatialTensor* result);

absl::Status CreateSharedTensor(const wgpu::Buffer& buffer,
                                size_t offset,
                                const TensorDescriptor& descriptor,
                                SpatialTensor* result);

inline absl::Status CreateSharedTensor(const wgpu::Buffer& buffer,
                                       const TensorDescriptor& descriptor,
                                       SpatialTensor* result) {
  return CreateSharedTensor(buffer, 0, descriptor, result);
}

absl::Status CreateSharedTensor(const wgpu::Texture& texture,
                                const wgpu::TextureView& texture_view,
                                const TensorDescriptor& descriptor,
                                SpatialTensor* result);

inline SpatialTensor* AsSpatialTensor(GpuSpatialTensor* desc) {
  return (desc && desc->IsSpatialTensor()) ? static_cast<SpatialTensor*>(desc)
                                           : nullptr;
}

inline const SpatialTensor* AsSpatialTensor(const GpuSpatialTensor* desc) {
  return (desc && desc->IsSpatialTensor())
             ? static_cast<const SpatialTensor*>(desc)
             : nullptr;
}

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_SPATIAL_TENSOR_H_
