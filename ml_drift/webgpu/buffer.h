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

#ifndef ML_DRIFT_WEBGPU_BUFFER_H_
#define ML_DRIFT_WEBGPU_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

class Buffer : public GpuObject {
 public:
  Buffer() = default;
  Buffer(const wgpu::Buffer& buffer, int size);
  // Creating a buffer with an offset will refer to the chunk of `buffer`
  // starting at `offset` and limited to `size` bytes. This can be used to have
  // a single backing wgpu::Buffer that multiple Buffer's refer to at different
  // offsets.
  Buffer(const wgpu::Buffer& buffer, int size, int offset);
  ~Buffer() override { Release(); }

  // Move only
  Buffer(Buffer&& buffer);
  Buffer& operator=(Buffer&& buffer);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  wgpu::Buffer GetMemoryHandle() const { return buffer_; }
  uint64_t GetMemorySizeInBytes() const { return size_; }
  uint64_t offset() const { return offset_; }

  template <typename T>
  absl::Status WriteData(const wgpu::Queue& queue, absl::Span<T> data);

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GpuResourcesWithValue* resources) const override;

  void CreateFromBufferDescriptor(const wgpu::Device& device,
                                  const BufferDescriptor& desc);

 private:
  void Release();

  wgpu::Buffer buffer_ = nullptr;
  uint64_t size_ = 0;
  uint64_t offset_ = 0;
};

Buffer CreateBuffer(const wgpu::Device& device, wgpu::BufferUsage usage,
                    size_t size_in_bytes, const void* data_ptr = nullptr);

Buffer CreateBufferStorage(const wgpu::Device& device, size_t size_in_bytes,
                           const void* data_ptr = nullptr);

Buffer CreateBufferConstantStorage(const wgpu::Device& device,
                                   size_t size_in_bytes, const void* data_ptr);

Buffer CreateBufferUniform(const wgpu::Device& device, size_t size_in_bytes,
                           const void* data_ptr);

template <typename T>
absl::Status Buffer::WriteData(const wgpu::Queue& queue, absl::Span<T> data) {
  const size_t data_size_in_bytes = sizeof(T) * data.size();
  if (data_size_in_bytes > size_) {
    return absl::InvalidArgumentError(
        "absl::Span<T> data size is greater from buffer allocated size.");
  }
  queue.WriteBuffer(buffer_, offset_, data.data(), data_size_in_bytes);
  return absl::OkStatus();
}

class UniformBufferCreator {
 public:
  Buffer CreateUniformBuffer(const wgpu::Device& device, size_t size) {
    if (base_buffers_.empty() ||
        size + uniform_offset_ > base_buffers_.back().GetMemorySizeInBytes()) {
      // Allocate a larger 1MB buffer instead of individual buffers for perf.
      base_buffers_.push_back(
          CreateBufferUniform(device, 1024 * 1024, nullptr));
      uniform_offset_ = 0;
    }
    Buffer buffer(base_buffers_.back().GetMemoryHandle(), size,
                  uniform_offset_);
    uniform_offset_ += size;
    // WebGpu requires uniform buffer offsets to be aligned to 256.
    uniform_offset_ = AlignByN(uniform_offset_, 256);
    return buffer;
  }

 private:
  std::vector<Buffer> base_buffers_;
  uint64_t uniform_offset_ = 0;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_BUFFER_H_
