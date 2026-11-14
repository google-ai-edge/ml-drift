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

#include "ml_drift/webgpu/buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

Buffer::Buffer(const wgpu::Buffer& buffer, uint64_t size)
    : buffer_(buffer), size_(size) {}
Buffer::Buffer(const wgpu::Buffer& buffer, uint64_t size, uint64_t offset)
    : buffer_(buffer), size_(size), offset_(offset) {}

Buffer::Buffer(Buffer&& buffer)
    : buffer_(std::move(buffer.buffer_)),
      size_(buffer.size_),
      offset_(buffer.offset_) {
  buffer.Release();
}

Buffer& Buffer::operator=(Buffer&& buffer) {
  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(offset_, buffer.offset_);
    std::swap(buffer_, buffer.buffer_);
    GpuObject::operator=(std::move(buffer));
  }
  return *this;
}

void Buffer::Release() {
  buffer_ = nullptr;
  size_ = 0;
  offset_ = 0;
}

absl::Status Buffer::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GpuResourcesWithValue* resources) const {
  resources->buffers.push_back(
      {"buffer", BufferResource{buffer_, size_, offset_}});
  return absl::OkStatus();
}

absl::Status Buffer::CreateFromBufferDescriptor(const wgpu::Device& device,
                                                const BufferDescriptor& desc) {
  uint8_t* data_ptr =
      desc.data.empty() ? nullptr : const_cast<uint8_t*>(desc.data.data());
  if (desc.memory_type == MemoryType::kConstant) {
    wgpu::BufferUsage usage = wgpu::BufferUsage::Uniform;
    *this = CreateBuffer(device, usage, desc.size, data_ptr);
  } else {
    *this = CreateBufferStorage(device, desc.size, data_ptr);
  }
  if (desc.size > 0 && !buffer_) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Failed to allocate WebGPU buffer of ", desc.size, " bytes."));
  }
  return absl::OkStatus();
}

Buffer CreateBuffer(const wgpu::Device& device, wgpu::BufferUsage usage,
                    uint64_t size_in_bytes, const void* data_ptr) {
  wgpu::BufferDescriptor buffer_desc = {
      .usage = usage,
      .size = size_in_bytes,
      .mappedAtCreation = data_ptr != nullptr,
  };
  wgpu::Buffer buffer = device.CreateBuffer(&buffer_desc);
  if (!buffer) {
    return Buffer();
  }
  if (data_ptr != nullptr) {
    const size_t mapped_size = static_cast<size_t>(size_in_bytes);
    void* void_ptr = buffer.GetMappedRange(0, mapped_size);
    if (void_ptr == nullptr) {
      // The initial data cannot be written. Report it like a failed allocation
      // rather than returning a buffer whose contents were never set.
      return Buffer();
    }
    memcpy(void_ptr, data_ptr, mapped_size);
    buffer.Unmap();
  }
  return Buffer(buffer, size_in_bytes);
}

Buffer CreateBufferStorage(const wgpu::Device& device, uint64_t size_in_bytes,
                           const void* data_ptr) {
  wgpu::BufferUsage usage = wgpu::BufferUsage::CopySrc |
                            wgpu::BufferUsage::CopyDst |
                            wgpu::BufferUsage::Storage;
  return CreateBuffer(device, usage, size_in_bytes, data_ptr);
}

Buffer CreateBufferConstantStorage(const wgpu::Device& device,
                                   uint64_t size_in_bytes,
                                   const void* data_ptr) {
  wgpu::BufferUsage usage = wgpu::BufferUsage::Storage;
  return CreateBuffer(device, usage, size_in_bytes, data_ptr);
}

Buffer CreateBufferUniform(const wgpu::Device& device, uint64_t size_in_bytes,
                           const void* data_ptr) {
  wgpu::BufferUsage usage =
      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  return CreateBuffer(device, usage, size_in_bytes, data_ptr);
}

}  // namespace webgpu
}  // namespace ml_drift
