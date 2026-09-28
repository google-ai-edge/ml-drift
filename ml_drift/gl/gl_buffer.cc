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

#include "ml_drift/gl/gl_buffer.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/gl/gpu_object.h"

namespace ml_drift {
namespace gl {
namespace {
absl::Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only,
                          const void* data, GlBuffer* result) {
  GLenum buffer_type =
      gpu_read_only ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER;
  GLuint buffer_id;
  glGenBuffers(1, &buffer_id);
  glBindBuffer(buffer_type, buffer_id);
  glBufferData(buffer_type, size_in_bytes, data, GL_STATIC_DRAW);
  glBindBuffer(buffer_type, 0);
  *result = GlBuffer(buffer_id, size_in_bytes, buffer_type);

  return absl::OkStatus();
}
}  // namespace
GlBuffer::GlBuffer(GLuint buffer_id, int size, GLenum buffer_type,
                   bool memory_owner)
    : buffer_id_(buffer_id),
      size_(size),
      buffer_type_(buffer_type),
      memory_owner_(memory_owner) {}

GlBuffer::GlBuffer(GlBuffer&& buffer) {
  buffer_id_ = buffer.buffer_id_;
  size_ = buffer.size_;
  buffer_type_ = buffer.buffer_type_;
  memory_owner_ = buffer.memory_owner_;
  buffer.buffer_id_ = -1;
  buffer.size_ = 0;
  GPUObject::operator=(std::move(buffer));
}

GlBuffer& GlBuffer::operator=(GlBuffer&& buffer) {
  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(buffer_id_, buffer.buffer_id_);
    std::swap(buffer_type_, buffer.buffer_type_);
    std::swap(memory_owner_, buffer.memory_owner_);
    GPUObject::operator=(std::move(buffer));
  }
  return *this;
}

void GlBuffer::Release() {
  if (memory_owner_ && buffer_id_ != -1) {
    glDeleteBuffers(1, &buffer_id_);
    buffer_id_ = -1;
    size_ = 0;
  }
}

absl::Status GlBuffer::GetGPUResources(
    const ml_drift::GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
  resources->buffers.push_back({"buffer", buffer_id_});
  return absl::OkStatus();
}

void GlBuffer::CreateFromBufferDescriptor(
    const ml_drift::BufferDescriptor& desc) {
  size_ = desc.size;
  uint8_t* data_ptr =
      desc.data.empty() ? nullptr : const_cast<uint8_t*>(desc.data.data());

  buffer_type_ = desc.memory_type == ml_drift::MemoryType::kConstant
                     ? GL_UNIFORM_BUFFER
                     : GL_SHADER_STORAGE_BUFFER;
  glGenBuffers(1, &buffer_id_);
  glBindBuffer(buffer_type_, buffer_id_);
  glBufferData(buffer_type_, desc.size, data_ptr, GL_STATIC_DRAW);
  glBindBuffer(buffer_type_, 0);
}

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, GlBuffer* result) {
  return CreateBuffer(size_in_bytes, false, nullptr, result);
}

void CreateSharedSSBOBuffer(GLuint buffer_id, GlBuffer* result) {
  *result =
      GlBuffer(buffer_id, 1, GL_SHADER_STORAGE_BUFFER, /*memory_owner*/ false);
}

}  // namespace gl
}  // namespace ml_drift
