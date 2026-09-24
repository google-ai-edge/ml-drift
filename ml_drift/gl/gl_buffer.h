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

#ifndef ML_DRIFT_GL_GL_BUFFER_H_
#define ML_DRIFT_GL_GL_BUFFER_H_

#include "absl/status/status.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/gl/gpu_object.h"

namespace ml_drift {
namespace gl {

class GlBuffer : public GPUObject {
 public:
  GlBuffer() = default;
  GlBuffer(GLuint buffer_id, int size, GLenum buffer_type,
           bool memory_owner = true);

  // Move only
  GlBuffer(GlBuffer&& buffer);
  GlBuffer& operator=(GlBuffer&& buffer);
  GlBuffer(const GlBuffer&) = delete;
  GlBuffer& operator=(const GlBuffer&) = delete;

  ~GlBuffer() override { Release(); }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const { return size_; }

  GLuint GetMemoryHandle() const { return buffer_id_; }

  absl::Status GetGPUResources(const ml_drift::GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  void CreateFromBufferDescriptor(const ml_drift::BufferDescriptor& desc);

  template <typename T>
  absl::Status ReadData(std::vector<T>* result) const;

 private:
  void Release();

  GLuint buffer_id_ = -1;
  size_t size_;
  GLenum buffer_type_;
  bool memory_owner_ = true;
};

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, GlBuffer* result);
void CreateSharedSSBOBuffer(GLuint buffer_id, GlBuffer* result);

template <typename T>
absl::Status GlBuffer::ReadData(std::vector<T>* result) const {
  if (size_ % sizeof(T) != 0) {
    return absl::UnknownError("Wrong element size(typename T is not correct?");
  }

  const int elements_count = size_ / sizeof(T);
  result->resize(elements_count);

  glBindBuffer(buffer_type_, buffer_id_);
  void* mapped_ptr = glMapBufferRange(buffer_type_, 0, size_, GL_MAP_READ_BIT);
  memcpy(result->data(), mapped_ptr, size_);
  glUnmapBuffer(buffer_type_);
  glBindBuffer(buffer_type_, 0);

  return absl::OkStatus();
}

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_BUFFER_H_
