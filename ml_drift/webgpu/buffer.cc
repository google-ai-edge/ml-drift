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

#include <utility>

#include "absl/status/status.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/webgpu_api_util.h"

namespace ml_drift {
namespace webgpu {

Buffer::Buffer(const wgpu::Buffer& buffer, int size)
    : buffer_(buffer), size_(size) {}
Buffer::Buffer(const wgpu::Buffer& buffer, int size, int offset)
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

void Buffer::CreateFromBufferDescriptor(const wgpu::Device& device,
                                        const BufferDescriptor& desc) {
  uint8_t* data_ptr =
      desc.data.empty() ? nullptr : const_cast<uint8_t*>(desc.data.data());
  if (desc.memory_type == MemoryType::CONSTANT) {
    wgpu::BufferUsage usage = wgpu::BufferUsage::Uniform;
    *this = CreateBuffer(device, usage, desc.size, data_ptr);
  } else {
    *this = CreateBufferStorage(device, desc.size, data_ptr);
  }
}

Buffer CreateBuffer(const wgpu::Device& device, wgpu::BufferUsage usage,
                    size_t size_in_bytes, const void* data_ptr) {
  wgpu::BufferDescriptor buffer_desc = {
      .usage = usage,
      .size = size_in_bytes,
      .mappedAtCreation = data_ptr != nullptr,
  };
  wgpu::Buffer buffer = device.CreateBuffer(&buffer_desc);
  if (data_ptr != nullptr) {
    void* void_ptr = buffer.GetMappedRange(0, size_in_bytes);
    memcpy(void_ptr, data_ptr, size_in_bytes);
    buffer.Unmap();
  }
  return Buffer(buffer, size_in_bytes);
}

Buffer CreateBufferStorage(const wgpu::Device& device, size_t size_in_bytes,
                           const void* data_ptr) {
  wgpu::BufferUsage usage = wgpu::BufferUsage::CopySrc |
                            wgpu::BufferUsage::CopyDst |
                            wgpu::BufferUsage::Storage;
  return CreateBuffer(device, usage, size_in_bytes, data_ptr);
}

Buffer CreateBufferConstantStorage(const wgpu::Device& device,
                                   size_t size_in_bytes, const void* data_ptr) {
  wgpu::BufferUsage usage = wgpu::BufferUsage::Storage;
  return CreateBuffer(device, usage, size_in_bytes, data_ptr);
}

Buffer CreateBufferUniform(const wgpu::Device& device, size_t size_in_bytes,
                           const void* data_ptr) {
  wgpu::BufferUsage usage =
      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  return CreateBuffer(device, usage, size_in_bytes, data_ptr);
}

}  // namespace webgpu
}  // namespace ml_drift
