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

#include "ml_drift/webgpu/spatial_tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {
bool HasStorageSupport(DataType data_type, int channels_count) {
  if (channels_count == 4) {
    return true;
  }
  if (SizeOf(data_type) == 4) {
    return true;
  }
  return false;
}

absl::Status AllocateTensorMemory(
    const wgpu::Device& device, const TensorDescriptor& descriptor,
    absl::Span<const uint8_t> data,
    SpatialTensor::SpatialTensorCreateInfo* result) {
  result->memory_owner = true;
  std::vector<uint64_t> storage_dims = descriptor.GetStorageDims();
  switch (descriptor.GetStorageType()) {
    case TensorStorageType::kBuffer:
    case TensorStorageType::kImageBuffer: {
      const size_t data_size = storage_dims[0] * descriptor.GetElementSize() *
                               SizeOf(descriptor.GetDataType());
      wgpu::BufferDescriptor buffer_desc = {
          .usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst |
                   wgpu::BufferUsage::Storage,
          .size = data_size,
      };
      result->buffer = device.CreateBuffer(&buffer_desc);
      if (!data.empty()) {
        if (data.size() < data_size) {
          return absl::InternalError(absl::StrCat(
              "Data size mismatch in buffer: ", data.size(), " ", data_size));
        }
        WriteDataToBuffer(device.GetQueue(), result->buffer, data_size,
                          data.data());
      }
      return absl::OkStatus();
    }
    case TensorStorageType::kTextureArray:
    case TensorStorageType::kTexture3D:
    case TensorStorageType::kTexture2D:
    case TensorStorageType::kSingleTexture2D: {
      wgpu::TextureDimension dimension = wgpu::TextureDimension::e2D;
      wgpu::TextureViewDimension view_dimension =
          wgpu::TextureViewDimension::e2D;
      if (descriptor.GetStorageType() == TensorStorageType::kTexture3D) {
        dimension = wgpu::TextureDimension::e3D;
        view_dimension = wgpu::TextureViewDimension::e3D;
      } else if (descriptor.GetStorageType() ==
                 TensorStorageType::kTextureArray) {
        dimension = wgpu::TextureDimension::e2D;
        view_dimension = wgpu::TextureViewDimension::e2DArray;
      }

      wgpu::TextureDescriptor tex_descriptor;
      tex_descriptor.mipLevelCount = 1;
      tex_descriptor.sampleCount = 1;
      tex_descriptor.dimension = dimension;
      tex_descriptor.usage = wgpu::TextureUsage::CopySrc |
                             wgpu::TextureUsage::CopyDst |
                             wgpu::TextureUsage::TextureBinding |
                             wgpu::TextureUsage::StorageBinding;
      const int channels =
          descriptor.GetStorageType() == TensorStorageType::kSingleTexture2D
              ? descriptor.GetBHWCShape().c
              : 4;
      if (descriptor.GetStorageType() == TensorStorageType::kSingleTexture2D &&
          !HasStorageSupport(descriptor.GetDataType(), channels)) {
        tex_descriptor.usage = wgpu::TextureUsage::CopySrc |
                               wgpu::TextureUsage::CopyDst |
                               wgpu::TextureUsage::TextureBinding;
      }
      tex_descriptor.size.width = storage_dims[0];
      tex_descriptor.size.height = storage_dims[1];
      if (descriptor.GetStorageType() == TensorStorageType::kTextureArray ||
          descriptor.GetStorageType() == TensorStorageType::kTexture3D) {
        tex_descriptor.size.depthOrArrayLayers = storage_dims[2];
      }
      tex_descriptor.format =
          DataTypeToTextureFormat(descriptor.GetDataType(), channels);
      result->texture = device.CreateTexture(&tex_descriptor);
      if (!data.empty()) {
        int pixel_size = SizeOf(descriptor.GetDataType()) * channels;
        int total_size = pixel_size * tex_descriptor.size.width *
                         tex_descriptor.size.height *
                         tex_descriptor.size.depthOrArrayLayers;
        if (data.size() < total_size) {
          return absl::InternalError(absl::StrCat(
              "Data size mismatch in texture: ", data.size(), " ", total_size));
        }
        WriteDataToTexture(device.GetQueue(), result->texture, pixel_size,
                           tex_descriptor.size.width,
                           tex_descriptor.size.height,
                           tex_descriptor.size.depthOrArrayLayers, data.data());
      }
      wgpu::TextureViewDescriptor tex_view_desc;
      tex_view_desc.nextInChain = nullptr;
      tex_view_desc.label = nullptr;
      tex_view_desc.format = tex_descriptor.format;
      tex_view_desc.dimension = view_dimension;
      tex_view_desc.baseMipLevel = 0;
      tex_view_desc.mipLevelCount = 1;
      tex_view_desc.baseArrayLayer = 0;
      tex_view_desc.arrayLayerCount =
          descriptor.GetStorageType() == TensorStorageType::kTextureArray
              ? tex_descriptor.size.depthOrArrayLayers
              : 1;
      tex_view_desc.aspect = wgpu::TextureAspect::All;
      result->texture_view = result->texture.CreateView(&tex_view_desc);
      return absl::OkStatus();
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
}

// The maximum size of a single upload chunk to a WebGPU buffer.
// This number is picked based on the observation of the peak resident set size
// when uploading Gemma3N 4B weights to GPU. The peak resident set size keeps
// dropping until the chunk size is reduced to 16MB.
constexpr size_t kMaxUploadChunkBytes = 16 * 1024 * 1024;

#if defined(__linux__) && !defined(__ANDROID__) && !defined(__EMSCRIPTEN__)
#define PREFER_HOST_MAPPED_POINTER 1
// Alignment required for BufferHostMappedPointer not exposed publicly. See
// https://dawn.googlesource.com/dawn.git/+/refs/heads/main/src/dawn/native/vulkan/UtilsVulkan.h#209.
constexpr size_t kMappedPointerAlignment = 4096u;
#endif  // defined(__linux__) && !defined(__ANDROID__) &&
        // !defined(__EMSCRIPTEN__)

// Forward declaration.
absl::Status WriteDataToBufferViaStagingBuffer(
    const Environment& env, wgpu::Buffer buffer, size_t data_size,
    const void* data_ptr, size_t offset = 0, bool dont_split = false);

absl::Status WriteDataToBufferViaStagingBufferInChunks(
    const Environment& env, const wgpu::Buffer buffer, size_t data_size,
    const void* data_ptr, size_t offset) {
#if PREFER_HOST_MAPPED_POINTER
  if (env.GetInfo().webgpu_info.supports_host_mapped_pointer) {
    // Write first data up to aligned pointer.
    uintptr_t data_intptr = reinterpret_cast<uintptr_t>(data_ptr);
    uintptr_t aligned_intptr = AlignByN(data_intptr, kMappedPointerAlignment);
    if (aligned_intptr > data_intptr) {
      size_t chunk_size = aligned_intptr - data_intptr;
      ABSL_RETURN_IF_ERROR(WriteDataToBufferViaStagingBuffer(
          env, buffer, chunk_size, data_ptr, offset,
          /*dont_split=*/true));
      data_ptr = reinterpret_cast<const void*>(aligned_intptr);
      data_size -= chunk_size;
      offset += chunk_size;
    }
  }
#endif  // PREFER_HOST_MAPPED_POINTER

  int num_chunks = DivideRoundUp(data_size, kMaxUploadChunkBytes);
  for (int i = 0; i < num_chunks; ++i) {
    size_t chunk_size = std::min(data_size, kMaxUploadChunkBytes);
    ABSL_RETURN_IF_ERROR(WriteDataToBufferViaStagingBuffer(
        env, buffer, chunk_size, data_ptr, offset,
        /*dont_split=*/true));
    data_ptr = static_cast<const uint8_t*>(data_ptr) + chunk_size;
    data_size -= chunk_size;
    offset += chunk_size;
  }
  return absl::OkStatus();
}

// This function copies data with size `data_size` from a memory pointer
// `data_ptr` to a WebGPU buffer `buffer` at the buffer's offset `offset` via a
// staging buffer.
absl::Status WriteDataToBufferViaStagingBuffer(const Environment& env,
                                               const wgpu::Buffer buffer,
                                               size_t data_size,
                                               const void* data_ptr,
                                               size_t offset, bool dont_split) {
  if (data_size % 4 != 0) {
    return absl::InvalidArgumentError(
        "Data size is not a multiple of 4 in WriteDataToBuffer. WebGPU "
        "requires 4-byte alignment.");
  }

  // It reduces peak resident set size to split large uploads into chunks.
  if (data_size >= kMaxUploadChunkBytes && !dont_split) {
    return WriteDataToBufferViaStagingBufferInChunks(env, buffer, data_size,
                                                     data_ptr, offset);
  }

  const wgpu::Device& device = env.device();
  // Create the staging buffer.
  wgpu::Buffer staging_buffer;
  // There are two methods to copy data to a staging buffer:
  // 1. Allocate a heap memory, copy data to it, and create a staging buffer
  //    that shares this allocated memory using `wgpu::BufferHostMappedPointer`.
  // 2. Create a staging buffer with `mappedAtCreation`, map its memory, copy
  //    data to the mapped range, and then unmap.
  //
  // Performance observations:
  // *   Method 1 is approximately 1.9x faster on a Linux workstations with
  //     NVIDIA GeForce RTX 4090 GPU.
  // *   Method 2 is about 1.26x faster on a Pixel 9 device with Mali-G715 GPU.
  //
  // The implementation uses Method 1 on Linux and Method 2 on other platforms
  // to leverage these performance differences.
#if PREFER_HOST_MAPPED_POINTER
  std::unique_ptr<void, void (*)(void*)> aligned_data(nullptr, std::free);
  if (env.GetInfo().webgpu_info.supports_host_mapped_pointer) {
    wgpu::BufferHostMappedPointer host_mapped_desc;
    host_mapped_desc.disposeCallback = nullptr;
    host_mapped_desc.userdata = nullptr;
    size_t aligned_data_size = AlignByN(data_size, kMappedPointerAlignment);
    const bool aligned =
        (reinterpret_cast<uintptr_t>(data_ptr) % kMappedPointerAlignment ==
         0) &&
        (data_size == aligned_data_size);
    // Error out if trying to map directly but failed alignment.
    if (!aligned && data_size == kMaxUploadChunkBytes) {
      return absl::InternalError(absl::StrCat("Failed to align upload data to ",
                                              kMappedPointerAlignment,
                                              " bytes."));
    }
    if (aligned) {
      host_mapped_desc.pointer = const_cast<void*>(data_ptr);
    } else {
      aligned_data.reset(
          std::aligned_alloc(kMappedPointerAlignment, aligned_data_size));
      memcpy(aligned_data.get(), data_ptr, data_size);
      host_mapped_desc.pointer = aligned_data.get();
    }
    wgpu::BufferDescriptor buffer_desc = {
        .nextInChain = &host_mapped_desc,
        .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
        .size = aligned_data_size,
    };
    staging_buffer = device.CreateBuffer(&buffer_desc);
  } else {
#else
  {
#endif  // PREFER_HOST_MAPPED_POINTER
    wgpu::BufferDescriptor staging_buffer_desc = {
        .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
        .size = data_size,
        .mappedAtCreation = true,
    };
    staging_buffer = device.CreateBuffer(&staging_buffer_desc);
    staging_buffer.WriteMappedRange(0, data_ptr, data_size);
    staging_buffer.Unmap();
  }

  // Copy the data from the staging buffer to the destination buffer.
  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  encoder.CopyBufferToBuffer(staging_buffer, 0, buffer, offset, data_size);
  wgpu::CommandBuffer cb = encoder.Finish();
  device.GetQueue().Submit(1, &cb);
#if defined(__EMSCRIPTEN__)
  // Can't be blocked on Web. Try best effort to get the work done.
  Instance::Get().ProcessEvents();
#else
  ABSL_RETURN_IF_ERROR(
      WaitUntilCompleted(device.GetQueue(), device, absl::Seconds(10)));
#endif  // !__EMSCRIPTEN__
  staging_buffer.Destroy();
  return absl::OkStatus();
}
}  // namespace

SpatialTensor::SpatialTensor(const SpatialTensorCreateInfo& info,
                             const TensorDescriptor& descriptor)
    : buffer_(info.buffer),
      texture_(info.texture),
      texture_view_(info.texture_view),
      memory_owner_(info.memory_owner),
      descriptor_(descriptor),
      offset_(info.offset) {}

SpatialTensor::SpatialTensor(SpatialTensor&& tensor)
    : buffer_(std::move(tensor.buffer_)),
      texture_(std::move(tensor.texture_)),
      texture_view_(std::move(tensor.texture_view_)),
      memory_owner_(tensor.memory_owner_),
      descriptor_(std::move(tensor.descriptor_)),
      offset_(tensor.offset_) {
  tensor.memory_owner_ = false;
}

SpatialTensor& SpatialTensor::operator=(SpatialTensor&& tensor) {
  if (this != &tensor) {
    Release();
    std::swap(buffer_, tensor.buffer_);
    std::swap(texture_, tensor.texture_);
    std::swap(texture_view_, tensor.texture_view_);
    std::swap(memory_owner_, tensor.memory_owner_);
    descriptor_ = std::move(tensor.descriptor_);
    std::swap(offset_, tensor.offset_);
  }
  return *this;
}

void SpatialTensor::Release() {
  if (memory_owner_) {
    if (buffer_) {
      buffer_.Destroy();
    }
    if (texture_) {
      texture_.Destroy();
    }
  }
  buffer_ = nullptr;
  texture_ = nullptr;
  texture_view_ = nullptr;
  memory_owner_ = false;
  offset_ = 0;
}

// memory can be invalid(nullptr) for example for shared tensors.
bool SpatialTensor::IsValidMemory() const {
  const auto& storage_type = descriptor_.GetStorageType();
  if (storage_type == TensorStorageType::kBuffer ||
      storage_type == TensorStorageType::kImageBuffer) {
    return buffer_ != nullptr;
  } else {
    return texture_view_ != nullptr;
  }
}

absl::Status SpatialTensor::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GpuResourcesWithValue* resources) const {
  const BufferDescriptor* buffer_desc = AsBufferDescriptor(obj_ptr);
  if (buffer_desc) {
    if (descriptor_.GetStorageType() != TensorStorageType::kBuffer) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only with "
          "TensorStorageType::BUFFER.");
    }
    resources->buffers.push_back(
        {"buffer", BufferResource{buffer_, GetMemorySizeInBytes(), offset_}});
    return absl::OkStatus();
  }
  const TensorDescriptor* tensor_desc = AsTensorDescriptor(obj_ptr);
  if (!tensor_desc) {
    return absl::InvalidArgumentError("Expected TensorDescriptor on input.");
  }
  tensor_desc->GetGpuResources(descriptor_.GetBHWDCShape(),
                               &resources->generic);

  if (descriptor_.GetStorageType() == TensorStorageType::kBuffer) {
    resources->buffers.push_back(
        {"buffer", BufferResource{buffer_, GetMemorySizeInBytes(), offset_}});
  } else if (descriptor_.GetStorageType() == TensorStorageType::kTexture2D ||
             descriptor_.GetStorageType() ==
                 TensorStorageType::kSingleTexture2D) {
    resources->images2d.push_back({"image2d", texture_view_});
  } else if (descriptor_.GetStorageType() == TensorStorageType::kTextureArray) {
    resources->image2d_arrays.push_back({"image2d_array", texture_view_});
  } else if (descriptor_.GetStorageType() == TensorStorageType::kTexture3D) {
    resources->images3d.push_back({"image3d", texture_view_});
  } else if (descriptor_.GetStorageType() == TensorStorageType::kImageBuffer) {
    return absl::InternalError("No support of IMAGE_BUFFER in WebGPU");
  }

  return absl::OkStatus();
}

absl::Status SpatialTensor::CreateFromBuffer(const wgpu::Buffer& buffer,
                                             size_t offset,
                                             const TensorDescriptor& desc) {
  desc.CopyWithoutData(&descriptor_);
  memory_owner_ = true;
  buffer_ = buffer;
  offset_ = offset;
  return absl::OkStatus();
}

absl::Status SpatialTensor::CreateFromDescriptor(const wgpu::Device& device,
                                                 const TensorDescriptor& desc) {
  desc.CopyWithoutData(&descriptor_);
  memory_owner_ = true;
  SpatialTensorCreateInfo create_info;
  ABSL_RETURN_IF_ERROR(
      AllocateTensorMemory(device, descriptor_, desc.GetData(), &create_info));
  buffer_ = create_info.buffer;
  texture_ = create_info.texture;
  texture_view_ = create_info.texture_view;
  memory_owner_ = create_info.memory_owner;
  offset_ = create_info.offset;
  return absl::OkStatus();
}

absl::Status SpatialTensor::UploadDescriptorData(const wgpu::Queue& queue,
                                                 const TensorDescriptor& desc) {
  return WriteData(queue, desc.GetData().data());
}

absl::Status SpatialTensor::ToDescriptor(const wgpu::Device& device,
                                         TensorDescriptor* desc) const {
  *desc = descriptor_;
  std::vector<uint8_t> data(GetMemorySizeInBytes());
  ABSL_RETURN_IF_ERROR(ReadData(device, data.data()));
  desc->SetData(std::move(data));
  return absl::OkStatus();
}

absl::Status SpatialTensor::WriteData(const wgpu::Queue& queue,
                                      const void* ptr) {
  switch (descriptor_.GetStorageType()) {
    case TensorStorageType::kBuffer:
    case TensorStorageType::kImageBuffer:
      WriteDataToBuffer(queue, buffer_, GetMemorySizeInBytes(), ptr);
      break;
    case TensorStorageType::kTextureArray:
    case TensorStorageType::kTexture2D:
    case TensorStorageType::kTexture3D:
    case TensorStorageType::kSingleTexture2D: {
      const int channels =
          descriptor_.GetStorageType() == TensorStorageType::kSingleTexture2D
              ? Channels()
              : 4;
      const int pixel_size = SizeOf(descriptor_.GetDataType()) * channels;
      auto region = descriptor_.GetFullTensorRegion();
      WriteDataToTexture(queue, texture_, pixel_size, region.x, region.y,
                         region.z, ptr);
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status SpatialTensor::WriteDataViaStaging(const Environment& env,
                                                const void* ptr) {
  switch (descriptor_.GetStorageType()) {
    case TensorStorageType::kBuffer:
    case TensorStorageType::kImageBuffer:
      return WriteDataToBufferViaStagingBuffer(env, buffer_,
                                               GetMemorySizeInBytes(), ptr);
    case TensorStorageType::kTextureArray:
    case TensorStorageType::kTexture2D:
    case TensorStorageType::kTexture3D:
    case TensorStorageType::kSingleTexture2D:
    default:
      ABSL_LOG(WARNING) << absl::StrCat(
          "Writing data via staging is not implemented yet for ",
          ToString(descriptor_.GetStorageType()),
          ", so falling back to writing data directly.");
      return WriteData(env.queue(), ptr);
  }
}

absl::Status SpatialTensor::ReadData(const wgpu::Device& device,
                                     void* ptr) const {
  switch (descriptor_.GetStorageType()) {
    case TensorStorageType::kBuffer:
    case TensorStorageType::kImageBuffer:
      ABSL_RETURN_IF_ERROR(ReadDataFromBuffer(
          device, device.GetQueue(), buffer_, GetMemorySizeInBytes(), ptr));
      break;
    case TensorStorageType::kTextureArray:
    case TensorStorageType::kTexture2D:
    case TensorStorageType::kTexture3D:
    case TensorStorageType::kSingleTexture2D: {
      const int channels =
          descriptor_.GetStorageType() == TensorStorageType::kSingleTexture2D
              ? Channels()
              : 4;
      const int pixel_size = SizeOf(descriptor_.GetDataType()) * channels;
      auto region = descriptor_.GetFullTensorRegion();
      ABSL_RETURN_IF_ERROR(ReadDataFromTexture(device, device.GetQueue(),
                                               texture_, pixel_size, region.x,
                                               region.y, region.z, ptr));
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status CreateTensor(const wgpu::Device& device,
                          const TensorDescriptor& descriptor,
                          SpatialTensor* result) {
  SpatialTensor::SpatialTensorCreateInfo create_info;
  ABSL_RETURN_IF_ERROR(
      AllocateTensorMemory(device, descriptor, {}, &create_info));
  *result = SpatialTensor(create_info, descriptor);
  return absl::OkStatus();
}

absl::Status CreateSharedTensor(const wgpu::Buffer& buffer,
                                size_t offset,
                                const TensorDescriptor& descriptor,
                                SpatialTensor* result) {
  SpatialTensor::SpatialTensorCreateInfo create_info;
  create_info.buffer = buffer;
  create_info.memory_owner = false;
  create_info.offset = offset;

  *result = SpatialTensor(create_info, descriptor);
  return absl::OkStatus();
}

absl::Status CreateSharedTensor(const wgpu::Texture& texture,
                                const wgpu::TextureView& texture_view,
                                const TensorDescriptor& descriptor,
                                SpatialTensor* result) {
  SpatialTensor::SpatialTensorCreateInfo create_info;
  create_info.texture = texture;
  create_info.texture_view = texture_view;
  create_info.memory_owner = false;

  *result = SpatialTensor(create_info, descriptor);
  return absl::OkStatus();
}
}  // namespace webgpu
}  // namespace ml_drift
