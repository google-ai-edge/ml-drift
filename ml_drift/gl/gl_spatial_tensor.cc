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

#include "ml_drift/gl/gl_spatial_tensor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace gl {
namespace {
absl::Status AllocateTensorMemory(const ml_drift::TensorDescriptor& descriptor,
                                  const void* data_ptr, GLuint* result) {
  std::vector<uint64_t> storage_dims = descriptor.GetStorageDims();
  switch (descriptor.GetStorageType()) {
    case ml_drift::TensorStorageType::BUFFER:
    case ml_drift::TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size = storage_dims[0] * descriptor.GetElementSize() *
                               SizeOf(descriptor.GetDataType());
      GLuint ssbo;
      glGenBuffers(1, &ssbo);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
      glBufferData(GL_SHADER_STORAGE_BUFFER, data_size, data_ptr,
                   GL_STATIC_DRAW);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
      *result = ssbo;
      return absl::OkStatus();
    }
    case ml_drift::TensorStorageType::TEXTURE_2D: {
      int tex_width = storage_dims[0];
      int tex_height = storage_dims[1];
      GLuint texture;
      glGenTextures(1, &texture);
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexStorage2D(
          GL_TEXTURE_2D, 1,
          ml_drift::gl::ToTextureInternalFormat(descriptor.GetDataType()),
          tex_width, tex_height);
      if (data_ptr) {
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, tex_width, tex_height,
            ml_drift::gl::ToTextureFormat(descriptor.GetDataType()),
            ml_drift::gl::ToTextureDataType(descriptor.GetDataType()),
            data_ptr);
      }
      glBindTexture(GL_TEXTURE_2D, 0);
      *result = texture;
      return absl::OkStatus();
    }
    case ml_drift::TensorStorageType::TEXTURE_3D: {
      int tex_width = storage_dims[0];
      int tex_height = storage_dims[1];
      int tex_depth = storage_dims[2];
      GLuint texture;
      glGenTextures(1, &texture);
      glBindTexture(GL_TEXTURE_3D, texture);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      glTexStorage3D(
          GL_TEXTURE_3D, 1,
          ml_drift::gl::ToTextureInternalFormat(descriptor.GetDataType()),
          tex_width, tex_height, tex_depth);
      if (data_ptr) {
        glTexSubImage3D(
            GL_TEXTURE_3D, 0, 0, 0, 0, tex_width, tex_height, tex_depth,
            ml_drift::gl::ToTextureFormat(descriptor.GetDataType()),
            ml_drift::gl::ToTextureDataType(descriptor.GetDataType()),
            data_ptr);
      }
      glBindTexture(GL_TEXTURE_3D, 0);
      *result = texture;
      return absl::OkStatus();
    }
    case ml_drift::TensorStorageType::TEXTURE_ARRAY: {
      int tex_width = storage_dims[0];
      int tex_height = storage_dims[1];
      int layers = storage_dims[2];
      GLuint texture;
      glGenTextures(1, &texture);
      glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexStorage3D(
          GL_TEXTURE_2D_ARRAY, 1,
          ml_drift::gl::ToTextureInternalFormat(descriptor.GetDataType()),
          tex_width, tex_height, layers);
      if (data_ptr) {
        glTexSubImage3D(
            GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, tex_width, tex_height, layers,
            ml_drift::gl::ToTextureFormat(descriptor.GetDataType()),
            ml_drift::gl::ToTextureDataType(descriptor.GetDataType()),
            data_ptr);
      }
      glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
      *result = texture;
      return absl::OkStatus();
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
}

absl::Status CreateImageBufferFromBuffer(GLuint memory,
                                         ml_drift::DataType data_type,
                                         int width, GLuint* result) {
  GLuint tbo_tex;
  glGenTextures(1, &tbo_tex);
  glBindTexture(GL_TEXTURE_BUFFER, tbo_tex);
  glTexBuffer(GL_TEXTURE_BUFFER,
              ml_drift::gl::ToTextureInternalFormat(data_type), memory);
  glBindTexture(GL_TEXTURE_BUFFER, 0);
  *result = tbo_tex;
  return absl::OkStatus();
}

}  // namespace

GlSpatialTensor::GlSpatialTensor(GLuint memory, bool memory_owner,
                                 const ml_drift::TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(-1),
      memory_owner_(memory_owner),
      descriptor_(descriptor) {}

GlSpatialTensor::GlSpatialTensor(GLuint memory, bool memory_owner,
                                 GLuint image_buffer_memory,
                                 const ml_drift::TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(image_buffer_memory),
      memory_owner_(memory_owner),
      descriptor_(descriptor) {
  if (image_buffer_memory &&
      (descriptor.GetStorageType() == ml_drift::TensorStorageType::TEXTURE_2D ||
       descriptor.GetStorageType() ==
           ml_drift::TensorStorageType::SINGLE_TEXTURE_2D)) {
    buffer_based_ = true;
  }
}

GlSpatialTensor::GlSpatialTensor(GlSpatialTensor&& tensor)
    : memory_(tensor.memory_),
      image_buffer_memory_(tensor.image_buffer_memory_),
      memory_owner_(tensor.memory_owner_),
      buffer_based_(tensor.buffer_based_),
      descriptor_(std::move(tensor.descriptor_)) {
  tensor.memory_ = -1;
  tensor.image_buffer_memory_ = -1;
}

GlSpatialTensor& GlSpatialTensor::operator=(GlSpatialTensor&& tensor) {
  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(image_buffer_memory_, tensor.image_buffer_memory_);
    std::swap(memory_owner_, tensor.memory_owner_);
    std::swap(buffer_based_, tensor.buffer_based_);
    descriptor_ = std::move(tensor.descriptor_);
  }
  return *this;
}

void GlSpatialTensor::Release() {
  // image_buffer_memory_ always owned by object
  if (image_buffer_memory_) {
    glDeleteTextures(1, &image_buffer_memory_);
    image_buffer_memory_ = -1;
  }
  if (memory_owner_ && memory_) {
    if (descriptor_.GetStorageType() == ml_drift::TensorStorageType::BUFFER ||
        descriptor_.GetStorageType() ==
            ml_drift::TensorStorageType::IMAGE_BUFFER) {
      glDeleteBuffers(1, &memory_);
    } else {
      glDeleteTextures(1, &memory_);
    }
    memory_ = -1;
  }
}

absl::Status GlSpatialTensor::GetGPUResources(
    const ml_drift::GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
  const auto* buffer_desc =
      dynamic_cast<const ml_drift::BufferDescriptor*>(obj_ptr);
  if (buffer_desc) {
    if (descriptor_.GetStorageType() != ml_drift::TensorStorageType::BUFFER) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only with "
          "TensorStorageType::BUFFER.");
    }
    resources->buffers.push_back({"buffer", memory_});
    return absl::OkStatus();
  }
  const auto* tensor_desc =
      dynamic_cast<const ml_drift::TensorDescriptor*>(obj_ptr);
  if (!tensor_desc) {
    return absl::InvalidArgumentError("Expected TensorDescriptor on input.");
  }
  tensor_desc->GetGpuResources(descriptor_.GetBHWDCShape(),
                               &resources->generic);

  if (descriptor_.GetStorageType() == ml_drift::TensorStorageType::BUFFER) {
    resources->buffers.push_back({"buffer", memory_});
  } else if (descriptor_.GetStorageType() ==
                 ml_drift::TensorStorageType::TEXTURE_2D ||
             descriptor_.GetStorageType() ==
                 ml_drift::TensorStorageType::SINGLE_TEXTURE_2D) {
    GLuint mem = buffer_based_ ? image_buffer_memory_ : memory_;
    resources->images2d.push_back({"image2d", mem});
  } else if (descriptor_.GetStorageType() ==
             ml_drift::TensorStorageType::TEXTURE_ARRAY) {
    resources->image2d_arrays.push_back({"image2d_array", memory_});
  } else if (descriptor_.GetStorageType() ==
             ml_drift::TensorStorageType::TEXTURE_3D) {
    resources->images3d.push_back({"image3d", memory_});
  } else if (descriptor_.GetStorageType() ==
             ml_drift::TensorStorageType::IMAGE_BUFFER) {
    if (obj_ptr->GetAccess() == ml_drift::AccessType::READ) {
      resources->image_buffers.push_back(
          {"image_buffer", image_buffer_memory_});
    } else {
      resources->buffers.push_back({"buffer", memory_});
    }
  }

  return absl::OkStatus();
}

GLuint GlSpatialTensor::GetMemoryPtr() const {
  if (buffer_based_) {
    return image_buffer_memory_;
  } else {
    return descriptor_.GetStorageType() ==
                   ml_drift::TensorStorageType::IMAGE_BUFFER
               ? image_buffer_memory_
               : memory_;
  }
}

GLuint GlSpatialTensor::GetMemoryPtrForWriting() const {
  if (buffer_based_) {
    return image_buffer_memory_;
  } else {
    return memory_;
  }
}

absl::Status GlSpatialTensor::CreateFromDescriptor(
    const ml_drift::TensorDescriptor& desc) {
  desc.CopyWithoutData(&descriptor_);
  memory_owner_ = true;
  GLuint memory;
  const uint8_t* data_ptr =
      desc.GetData().empty() ? nullptr : desc.GetData().data();
  ABSL_RETURN_IF_ERROR(AllocateTensorMemory(descriptor_, data_ptr, &memory));
  memory_ = memory;
  if (desc.GetStorageType() == ml_drift::TensorStorageType::IMAGE_BUFFER) {
    std::vector<uint64_t> storage_dims = descriptor_.GetStorageDims();
    ABSL_RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        memory_, desc.GetDataType(), storage_dims[0], &image_buffer_memory_));
  }
  return absl::OkStatus();
}

absl::Status GlSpatialTensor::UploadDescriptorData(
    const ml_drift::TensorDescriptor& desc) {
  return WriteData(desc.GetData().data());
}

absl::Status GlSpatialTensor::ToDescriptor(
    ml_drift::TensorDescriptor* desc) const {
  *desc = descriptor_;
  std::vector<uint8_t> data(GetMemorySizeInBytes());
  ABSL_RETURN_IF_ERROR(ReadData(data.data()));
  desc->SetData(std::move(data));
  return absl::OkStatus();
}

absl::Status GlSpatialTensor::WriteData(const uint8_t* ptr) {
  auto region = descriptor_.GetFullTensorRegion();
  switch (descriptor_.GetStorageType()) {
    case ml_drift::TensorStorageType::BUFFER:
    case ml_drift::TensorStorageType::IMAGE_BUFFER:
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory_);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GetMemorySizeInBytes(), ptr);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
      break;
    case ml_drift::TensorStorageType::TEXTURE_ARRAY:
      glBindTexture(GL_TEXTURE_2D_ARRAY, memory_);
      glTexSubImage3D(
          GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, region.x, region.y, region.z,
          ml_drift::gl::ToTextureFormat(descriptor_.GetDataType()),
          ml_drift::gl::ToTextureDataType(descriptor_.GetDataType()), ptr);
      glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
      break;
    case ml_drift::TensorStorageType::TEXTURE_3D:
      glBindTexture(GL_TEXTURE_3D, memory_);
      glTexSubImage3D(
          GL_TEXTURE_3D, 0, 0, 0, 0, region.x, region.y, region.z,
          ml_drift::gl::ToTextureFormat(descriptor_.GetDataType()),
          ml_drift::gl::ToTextureDataType(descriptor_.GetDataType()), ptr);
      glBindTexture(GL_TEXTURE_3D, 0);
      break;
    case ml_drift::TensorStorageType::TEXTURE_2D:
    case ml_drift::TensorStorageType::SINGLE_TEXTURE_2D: {
      glBindTexture(GL_TEXTURE_2D, memory_);
      glTexSubImage2D(
          GL_TEXTURE_2D, 0, 0, 0, region.x, region.y,
          ml_drift::gl::ToTextureFormat(descriptor_.GetDataType()),
          ml_drift::gl::ToTextureDataType(descriptor_.GetDataType()), ptr);
      glBindTexture(GL_TEXTURE_2D, 0);
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }

  return absl::OkStatus();
}

absl::Status GlSpatialTensor::ReadData(uint8_t* ptr) const {
  auto region = descriptor_.GetFullTensorRegion();
  switch (descriptor_.GetStorageType()) {
    case ml_drift::TensorStorageType::BUFFER:
    case ml_drift::TensorStorageType::IMAGE_BUFFER: {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, memory_);
      void* mapped_ptr = glMapBufferRange(
          GL_SHADER_STORAGE_BUFFER, 0, GetMemorySizeInBytes(), GL_MAP_READ_BIT);
      memcpy(ptr, mapped_ptr, GetMemorySizeInBytes());
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
      break;
    }
    case ml_drift::TensorStorageType::SINGLE_TEXTURE_2D:
    case ml_drift::TensorStorageType::TEXTURE_2D: {
      GLint draw_fbo_id = 0, read_fbo_id = 0;
      glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo_id);
      glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo_id);

      GLuint fb;
      glGenFramebuffers(1, &fb);
      glBindFramebuffer(GL_FRAMEBUFFER, fb);

      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, memory_, 0);
      glReadPixels(0, 0, region.x, region.y,
                   ml_drift::gl::ToTextureFormat(descriptor_.GetDataType()),
                   ml_drift::gl::ToTextureDataType(descriptor_.GetDataType()),
                   ptr);

      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &fb);

      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo_id);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo_id);
      break;
    }
    case ml_drift::TensorStorageType::TEXTURE_3D:
    case ml_drift::TensorStorageType::TEXTURE_ARRAY: {
      GLint draw_fbo_id = 0, read_fbo_id = 0;
      glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo_id);
      glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo_id);

      GLuint fb;
      glGenFramebuffers(1, &fb);
      glBindFramebuffer(GL_FRAMEBUFFER, fb);

      for (int i = 0; i < region.z; ++i) {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, memory_,
                                  0, i);
        glReadPixels(0, 0, region.x, region.y,
                     ml_drift::gl::ToTextureFormat(descriptor_.GetDataType()),
                     ml_drift::gl::ToTextureDataType(descriptor_.GetDataType()),
                     ptr + region.x * region.y *
                               ml_drift::SizeOf(descriptor_.GetDataType()) * 4 *
                               i);
      }

      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &fb);

      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo_id);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo_id);
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }

  return absl::OkStatus();
}

absl::Status CreateTensor(const ml_drift::TensorDescriptor& descriptor,
                          GlSpatialTensor* result) {
  GLuint memory;
  ABSL_RETURN_IF_ERROR(AllocateTensorMemory(descriptor, nullptr, &memory));
  if (descriptor.GetStorageType() ==
      ml_drift::TensorStorageType::IMAGE_BUFFER) {
    std::vector<uint64_t> storage_dims = descriptor.GetStorageDims();
    GLuint image_memory;
    ABSL_RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        memory, descriptor.GetDataType(), storage_dims[0], &image_memory));
    *result = GlSpatialTensor(memory, /*memory_owner*/ true, image_memory,
                              descriptor);
  } else {
    *result = GlSpatialTensor(memory, /*memory_owner*/ true, descriptor);
  }
  return absl::OkStatus();
}

absl::Status CreateTensorShared(GLuint memory,
                                const ml_drift::TensorDescriptor& descriptor,
                                GlSpatialTensor* result) {
  if (descriptor.GetStorageType() ==
      ml_drift::TensorStorageType::IMAGE_BUFFER) {
    GLuint image_memory;
    std::vector<uint64_t> storage_dims = descriptor.GetStorageDims();
    ABSL_RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        memory, descriptor.GetDataType(), storage_dims[0], &image_memory));
    *result = GlSpatialTensor(memory, /*memory_owner*/ false, image_memory,
                              descriptor);
  } else {
    *result = GlSpatialTensor(memory, /*memory_owner*/ false, descriptor);
  }
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace ml_drift
