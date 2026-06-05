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

#ifndef ML_DRIFT_METAL_BUFFER_H_
#define ML_DRIFT_METAL_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#import <Metal/Metal.h>

#include "absl/types/span.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/metal/common.h"
#include "ml_drift/metal/gpu_object.h"

namespace ml_drift {
namespace metal {

class Buffer : public GPUObject {
 public:
  Buffer() = default;
  Buffer(id<MTLBuffer> buffer, size_t size_in_bytes);
  explicit Buffer(id<MTLBuffer> buffer);

  // Move only
  Buffer(Buffer&& buffer);
  Buffer& operator=(Buffer&& buffer);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer();

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const { return size_; }

  id<MTLBuffer> GetMemoryPtr() const { return buffer_; }

  // Writes data to a buffer. Data should point to a region that
  // has exact size in bytes as size_in_bytes(constructor parameter).
  template <typename T>
  absl::Status WriteData(absl::Span<T> data);
  template <typename T>
  absl::Status WriteData(id<MTLCommandQueue> command_queue, absl::Span<T> data,
                         bool wait_for_completion);

  // Reads data from Buffer into CPU memory.
  template <typename T>
  absl::Status ReadData(std::vector<T>* result) const;

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  absl::Status CreateFromBufferDescriptor(const BufferDescriptor& desc, id<MTLDevice> device);

 private:
  void Release();

  id<MTLBuffer> buffer_ = nullptr;
  size_t size_;
  bool owner_ = true;
};

Buffer CreateBufferShared(id<MTLBuffer> buffer);

absl::Status CreateBuffer(size_t size_in_bytes, const void* data, id<MTLDevice> device,
                          Buffer* result);

template <typename T>
absl::Status Buffer::WriteData(absl::Span<T> data) {
  const size_t data_size_in_bytes = sizeof(T) * data.size();
  if (data_size_in_bytes > size_) {
    return absl::InvalidArgumentError(
        "absl::Span<T> data size is greater from buffer allocated size.");
  }
  std::memcpy([buffer_ contents], data.data(), data_size_in_bytes);
  return absl::OkStatus();
}

template <typename T>
absl::Status Buffer::WriteData(id<MTLCommandQueue> command_queue, absl::Span<T> data,
                               bool wait_for_completion) {
  const size_t data_size_in_bytes = sizeof(T) * data.size();
  if (data_size_in_bytes > size_) {
    return absl::InvalidArgumentError(
        "absl::Span<T> data size is greater from buffer allocated size.");
  }
  WriteDataToBuffer(buffer_, /*buffer_offset=*/0, command_queue, data.data(), data_size_in_bytes,
                    wait_for_completion);
  return absl::OkStatus();
}

template <typename T>
absl::Status Buffer::ReadData(std::vector<T>* result) const {
  if (size_ % sizeof(T) != 0) {
    return absl::UnknownError("Wrong element size(typename T is not correct?");
  }

  const int elements_count = size_ / sizeof(T);
  result->resize(elements_count);
  std::memcpy(result->data(), [buffer_ contents], size_);

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_BUFFER_H_
