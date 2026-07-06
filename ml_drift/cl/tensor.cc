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

#include "ml_drift/cl/tensor.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_image_format.h"
#include "ml_drift/cl/cl_memory.h"
#include "ml_drift/cl/gpu_object.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/util.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>

namespace ml_drift {
namespace cl {
namespace {
absl::Status AllocateTensorMemoryInternal(const CLContext& context,
                                          const TensorDescriptor& tensor_desc,
                                          CLMemory* result) {
  cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
  const uint8_t* data_ptr = nullptr;
  if (!tensor_desc.GetData().empty()) {
    data_ptr = tensor_desc.GetData().data();
    mem_flags |= CL_MEM_COPY_HOST_PTR;
  }
  const DataType data_type = tensor_desc.GetDataType();
  std::vector<uint64_t> storage_dims = tensor_desc.GetStorageDims();
  switch (tensor_desc.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size =
          storage_dims[0] * tensor_desc.GetElementSize() * SizeOf(data_type);
      cl_int error_code;
      cl_mem memory =
          clCreateBuffer(context.context(), mem_flags, data_size,
                         const_cast<uint8_t*>(data_ptr), &error_code);
      if (!memory) {
        return absl::UnknownError(absl::StrCat(
            "MLDrift OpenCL Tensor ", ToStringWithShape(tensor_desc),
            ": Failed to allocate device memory "
            "(clCreateBuffer): ",
            CLErrorCodeToString(error_code), ", buffer size: ", data_size));
      }
      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_2D: {
      cl_image_desc image_desc;
      image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      image_desc.image_width = storage_dims[0];
      image_desc.image_height = storage_dims[1];
      image_desc.image_depth = 0;
      image_desc.image_row_pitch = 0;
      image_desc.image_slice_pitch = 0;
      image_desc.num_mip_levels = 0;
      image_desc.num_samples = 0;
      image_desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = DataTypeToChannelType(data_type);

      cl_int error_code;
      cl_mem memory = CreateImage2DLegacy(
          context.context(), mem_flags, &format, &image_desc,
          const_cast<uint8_t*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(absl::StrCat(
            "MLDrift OpenCL Tensor ", ToStringWithShape(tensor_desc),
            ": Failed to create 2D texture "
            "(clCreateImage): ",
            CLErrorCodeToString(error_code), ", texture width: ",
            storage_dims[0], ", texture height: ", storage_dims[1]));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_3D: {
      cl_image_desc image_desc;
      image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
      image_desc.image_width = storage_dims[0];
      image_desc.image_height = storage_dims[1];
      image_desc.image_depth = storage_dims[2];
      image_desc.image_row_pitch = 0;
      image_desc.image_slice_pitch = 0;
      image_desc.num_mip_levels = 0;
      image_desc.num_samples = 0;
      image_desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = DataTypeToChannelType(data_type);

      cl_int error_code;
      cl_mem memory = CreateImage3DLegacy(
          context.context(), mem_flags, &format, &image_desc,
          const_cast<uint8_t*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(absl::StrCat(
            "MLDrift OpenCL Tensor ", ToStringWithShape(tensor_desc),
            ": Failed to create 3D texture "
            "(clCreateImage): ",
            CLErrorCodeToString(error_code), ", texture width: ",
            storage_dims[0], ", texture height: ", storage_dims[1],
            ", texture depth: ", storage_dims[2]));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      cl_image_desc image_desc;
      image_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
      image_desc.image_width = storage_dims[0];
      image_desc.image_height = storage_dims[1];
      image_desc.image_depth = 0;
      image_desc.image_array_size = storage_dims[2];
      image_desc.image_row_pitch = 0;
      image_desc.image_slice_pitch = 0;
      image_desc.num_mip_levels = 0;
      image_desc.num_samples = 0;
      image_desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = DataTypeToChannelType(data_type);

      cl_int error_code;
      cl_mem memory =
          clCreateImage(context.context(), mem_flags, &format, &image_desc,
                        const_cast<uint8_t*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(absl::StrCat(
            "MLDrift OpenCL Tensor ", ToStringWithShape(tensor_desc),
            ": Failed to create 2D texture "
            "array (clCreateImage): ",
            CLErrorCodeToString(error_code), ", texture width: ",
            storage_dims[0], ", texture height: ", storage_dims[1],
            ", texture layers: ", storage_dims[2]));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }

    case TensorStorageType::SINGLE_TEXTURE_2D: {
      const int element_size = tensor_desc.GetElementSize();
      if (element_size > 4) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SINGLE_TEXTURE_2D support only channels in range [1-4], but ",
            element_size, "was provided"));
      }
      cl_image_desc image_desc;
      image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      image_desc.image_width = storage_dims[0];
      image_desc.image_height = storage_dims[1];
      image_desc.image_depth = 0;
      image_desc.image_row_pitch = 0;
      image_desc.image_slice_pitch = 0;
      image_desc.num_mip_levels = 0;
      image_desc.num_samples = 0;
      image_desc.buffer = nullptr;

      cl_image_format format;
      if (context.IsFloatTexture2DSupported(element_size, data_type)) {
        format.image_channel_order = ToChannelOrder(element_size);
        format.image_channel_data_type = DataTypeToChannelType(data_type);
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("This device doesn't support ", element_size,
                         "-channel textures."));
      }

      cl_int error_code;
      cl_mem memory = CreateImage2DLegacy(
          context.context(), mem_flags, &format, &image_desc,
          const_cast<uint8_t*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(absl::StrCat(
            "MLDrift OpenCL Tensor ", ToStringWithShape(tensor_desc),
            ": Failed to create single "
            "2D texture (clCreateImage): ",
            CLErrorCodeToString(error_code), ", texture width: ",
            storage_dims[0], ", texture height: ", storage_dims[1]));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }

    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
}

absl::Status CreateImageBufferFromBuffer(const CLContext& context,
                                         cl_mem memory, DataType data_type,
                                         int width, cl_mem* result) {
  cl_image_format format;
  cl_image_desc image_desc;
  std::memset(&image_desc, 0, sizeof(image_desc));
  image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  image_desc.image_width = width;
  image_desc.mem_object = memory;

  format.image_channel_data_type = DataTypeToChannelType(data_type);
  format.image_channel_order = CL_RGBA;

  cl_int error_code;
  *result = clCreateImage(context.context(), CL_MEM_READ_WRITE, &format,
                          &image_desc, nullptr, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create Image from Buffer (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateImage2DFromBuffer(const CLContext& context, cl_mem memory,
                                     DataType data_type, int width, int height,
                                     int channels, int width_pixel_alignment,
                                     cl_mem* result) {
  if (!context.IsFloatTexture2DSupported(channels, data_type)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "This device doesn't support ", channels, "-channel textures."));
  }

  cl_image_desc image_desc;
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc.image_width = width;
  image_desc.image_height = height;
  image_desc.image_depth = 0;
  const size_t width_aligned = AlignByN(width, width_pixel_alignment);
  image_desc.image_row_pitch = width_aligned * channels * SizeOf(data_type);
  image_desc.image_slice_pitch = 0;
  image_desc.num_mip_levels = 0;
  image_desc.num_samples = 0;
  image_desc.mem_object = memory;

  cl_image_format format;
  format.image_channel_order = ToChannelOrder(channels);
  format.image_channel_data_type = DataTypeToChannelType(data_type);

  ASSIGN_OR_RETURN(cl_mem_flags parent_flags, GetCLMemObjectFlags(memory));
  const bool read_only = parent_flags & CL_MEM_READ_ONLY;
  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  cl_int error_code;
  *result = CreateImage2DLegacy(context.context(), flags, &format, &image_desc,
                                nullptr, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create Image2D from Buffer (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status WriteDataViaStagingBuffer(const void* ptr, const size_t data_size,
                                       CLCommandQueue* queue,
                                       CLContext* context, cl_mem* result) {
  cl_int error_code;
  cl_mem stage_mem;
  // Create the staging buffer.
  stage_mem = clCreateBuffer(context->context(),
                             CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                             data_size, nullptr, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to create staging buffer: ",
                                           CLErrorCodeToString(error_code)));
  }
  // Upload data to the staging buffer via map/unmap.
  void* mapped_ptr =
      clEnqueueMapBuffer(queue->queue(), stage_mem, CL_TRUE, CL_MAP_WRITE, 0,
                         data_size, 0, nullptr, nullptr, &error_code);
  if (!mapped_ptr) {
    clReleaseMemObject(stage_mem);
    return absl::UnknownError(absl::StrCat("Failed to map staging buffer: ",
                                           CLErrorCodeToString(error_code)));
  }
  std::memcpy(mapped_ptr, ptr, data_size);
  error_code = clEnqueueUnmapMemObject(queue->queue(), stage_mem, mapped_ptr, 0,
                                       nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    clReleaseMemObject(stage_mem);
    return absl::UnknownError(absl::StrCat("Failed to create final buffer: ",
                                           CLErrorCodeToString(error_code)));
  }
  // Perform GPU-to-GPU copy.
  error_code =
      clEnqueueCopyBuffer(queue->queue(), stage_mem, *result,
                          /*src_offset=*/0,
                          /*dst_offset=*/0, data_size, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    clReleaseMemObject(*result);
    clReleaseMemObject(stage_mem);
    return absl::UnknownError(absl::StrCat("Failed to enqueue copy: ",
                                           CLErrorCodeToString(error_code)));
  }
  clFinish(queue->queue());
  clReleaseMemObject(stage_mem);
  return absl::OkStatus();
}
}  // namespace

Tensor::Tensor(cl_mem memory, bool memory_owner,
               const TensorDescriptor& tensor_desc)
    : memory_(memory),
      image_buffer_memory_(nullptr),
      memory_owner_(memory_owner) {
  tensor_desc.CopyWithoutData(&tensor_desc_);
}

Tensor::Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
               const TensorDescriptor& tensor_desc)
    : memory_(memory),
      image_buffer_memory_(image_buffer_memory),
      memory_owner_(memory_owner) {
  tensor_desc.CopyWithoutData(&tensor_desc_);
  if (image_buffer_memory &&
      (tensor_desc.GetStorageType() == TensorStorageType::TEXTURE_2D ||
       tensor_desc.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D)) {
    buffer_based_ = true;
  }
}

Tensor::Tensor(Tensor&& tensor)
    : memory_(tensor.memory_),
      image_buffer_memory_(tensor.image_buffer_memory_),
      memory_owner_(tensor.memory_owner_),
      buffer_based_(tensor.buffer_based_),
      tensor_desc_(std::move(tensor.tensor_desc_)),
      aligned_texture_width_(tensor.aligned_texture_width_) {
  tensor.memory_ = nullptr;
  tensor.image_buffer_memory_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) {
  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(image_buffer_memory_, tensor.image_buffer_memory_);
    std::swap(memory_owner_, tensor.memory_owner_);
    std::swap(buffer_based_, tensor.buffer_based_);
    tensor_desc_ = std::move(tensor.tensor_desc_);
    std::swap(aligned_texture_width_, tensor.aligned_texture_width_);
  }
  return *this;
}

void Tensor::Release() {
  // image_buffer_memory_ always owned by object
  if (image_buffer_memory_) {
    clReleaseMemObject(image_buffer_memory_);
    image_buffer_memory_ = nullptr;
  }
  if (memory_owner_ && memory_) {
    clReleaseMemObject(memory_);
    memory_ = nullptr;
  }
}

absl::Status Tensor::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GPUResourcesWithValue* resources) const {
  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (buffer_desc) {
    if (tensor_desc_.GetStorageType() != TensorStorageType::BUFFER &&
        tensor_desc_.GetStorageType() != TensorStorageType::IMAGE_BUFFER) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only with "
          "TensorStorageType::BUFFER/TensorStorageType::IMAGE_BUFFER.");
    }
    resources->buffers.push_back({"buffer", memory_});
    return absl::OkStatus();
  }
  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(obj_ptr);
  if (!tensor_desc) {
    return absl::InvalidArgumentError("Expected TensorDescriptor on input.");
  }
  tensor_desc->GetGpuResources(tensor_desc_.GetBHWDCShape(),
                               &resources->generic);

  if (tensor_desc_.GetStorageType() == TensorStorageType::BUFFER) {
    resources->buffers.push_back({"buffer", memory_});
  } else if (tensor_desc_.GetStorageType() == TensorStorageType::TEXTURE_2D ||
             tensor_desc_.GetStorageType() ==
                 TensorStorageType::SINGLE_TEXTURE_2D) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->GetUseBufferForWriteOnlyTexture2d()) {
      resources->AddInt("aligned_texture_width", aligned_texture_width_);
      resources->buffers.push_back({"buffer", memory_});
    } else {
      cl_mem mem = buffer_based_ ? image_buffer_memory_ : memory_;
      resources->images2d.push_back({"image2d", mem});
    }
  } else if (tensor_desc_.GetStorageType() ==
             TensorStorageType::TEXTURE_ARRAY) {
    resources->image2d_arrays.push_back({"image2d_array", memory_});
  } else if (tensor_desc_.GetStorageType() == TensorStorageType::TEXTURE_3D) {
    resources->images3d.push_back({"image3d", memory_});
  } else if (tensor_desc_.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->GetUseBufferForWriteOnlyImageBuffer()) {
      resources->buffers.push_back({"buffer", memory_});
    } else {
      resources->image_buffers.push_back(
          {"image_buffer", image_buffer_memory_});
    }
  }

  return absl::OkStatus();
}

cl_mem Tensor::GetMemoryPtr() const {
  if (buffer_based_) {
    return image_buffer_memory_;
  } else {
    return tensor_desc_.GetStorageType() == TensorStorageType::IMAGE_BUFFER
               ? image_buffer_memory_
               : memory_;
  }
}

cl_mem Tensor::GetMemoryPtrForWriting() const {
  if (buffer_based_) {
    return image_buffer_memory_;
  } else {
    return memory_;
  }
}

absl::Status Tensor::CreateFromDescriptor(const TensorDescriptor& tensor_desc,
                                          const CLContext& context) {
  tensor_desc.CopyWithoutData(&tensor_desc_);
  memory_owner_ = true;
  CLMemory memory;
  RETURN_IF_ERROR(AllocateTensorMemoryInternal(context, tensor_desc, &memory));
  memory_ = memory.Release();
  if (tensor_desc.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    std::vector<uint64_t> storage_dims = tensor_desc_.GetStorageDims();
    RETURN_IF_ERROR(
        CreateImageBufferFromBuffer(context, memory_, tensor_desc.GetDataType(),
                                    storage_dims[0], &image_buffer_memory_));
  }
  return absl::OkStatus();
}

absl::Status Tensor::UploadDescriptorData(const TensorDescriptor& tensor_desc,
                                          CLCommandQueue* queue) {
  return WriteData(tensor_desc.GetData().data(), queue);
}

absl::Status Tensor::ToDescriptor(TensorDescriptor* tensor_desc,
                                  CLCommandQueue* queue) const {
  *tensor_desc = tensor_desc_;
  std::vector<uint8_t> data(GetMemorySizeInBytes());
  RETURN_IF_ERROR(ReadData(data.data(), queue));
  tensor_desc->SetData(std::move(data));
  return absl::OkStatus();
}

absl::Status Tensor::WriteDataViaStaging(const void* ptr, CLCommandQueue* queue,
                                         CLContext* context) {
  const size_t data_size = GetMemorySizeInBytes();
  switch (tensor_desc_.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return WriteDataViaStagingBuffer(ptr, data_size, queue, context,
                                       &memory_);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError(
          absl::StrCat("Writing data via staging is not implemented yet for ",
                       ToString(tensor_desc_.GetStorageType())));
  }
  return absl::OkStatus();
}

absl::Status Tensor::WriteData(const void* ptr, CLCommandQueue* queue,
                               bool async) {
  switch (tensor_desc_.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      RETURN_IF_ERROR(queue->EnqueueWriteBuffer(memory_, GetMemorySizeInBytes(),
                                                ptr, async));
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      cl_mem mem = buffer_based_ ? image_buffer_memory_ : memory_;
      RETURN_IF_ERROR(queue->EnqueueWriteImage(
          mem, tensor_desc_.GetFullTensorRegion(), ptr, async));
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status Tensor::ReadData(void* ptr, CLCommandQueue* queue) const {
  switch (tensor_desc_.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      RETURN_IF_ERROR(
          queue->EnqueueReadBuffer(memory_, GetMemorySizeInBytes(), ptr));
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      cl_mem mem = buffer_based_ ? image_buffer_memory_ : memory_;
      RETURN_IF_ERROR(queue->EnqueueReadImage(
          mem, tensor_desc_.GetFullTensorRegion(), ptr));
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status CreateTensor(const CLContext& context,
                          const TensorDescriptor& tensor_desc, Tensor* result) {
  CLMemory mem;
  RETURN_IF_ERROR(AllocateTensorMemoryInternal(context, tensor_desc, &mem));
  cl_mem memory = mem.Release();
  cl_mem image_memory = nullptr;
  if (tensor_desc.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    std::vector<uint64_t> storage_dims = tensor_desc.GetStorageDims();
    RETURN_IF_ERROR(
        CreateImageBufferFromBuffer(context, memory, tensor_desc.GetDataType(),
                                    storage_dims[0], &image_memory));
  }
  TensorDescriptor desc_copy;
  tensor_desc.CopyWithoutData(&desc_copy);
  *result = Tensor(memory, /*memory_owner*/ true, image_memory, desc_copy);
  return absl::OkStatus();
}

absl::Status CreateTensorShared(const CLContext& context, cl_mem memory,
                                const TensorDescriptor& tensor_desc,
                                Tensor* result) {
  const bool memory_owner = false;
  if (tensor_desc.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    std::vector<uint64_t> storage_dims = tensor_desc.GetStorageDims();
    cl_mem image_memory;
    RETURN_IF_ERROR(
        CreateImageBufferFromBuffer(context, memory, tensor_desc.GetDataType(),
                                    storage_dims[0], &image_memory));
    *result = Tensor(memory, memory_owner, image_memory, tensor_desc);
  } else {
    *result = Tensor(memory, memory_owner, tensor_desc);
  }
  return absl::OkStatus();
}

absl::Status CreateTensorSharedImage2DBuffer(
    const CLContext& context, cl_mem memory,
    const TensorDescriptor& tensor_desc, int width_pixel_alignment,
    Tensor* result) {
  std::vector<uint64_t> storage_dims = tensor_desc.GetStorageDims();
  const int width = storage_dims[0];
  const int height = storage_dims[1];
  const int channels = tensor_desc.GetElementSize();
  cl_mem image_memory;
  RETURN_IF_ERROR(CreateImage2DFromBuffer(
      context, memory, tensor_desc.GetDataType(), width, height, channels,
      width_pixel_alignment, &image_memory));
  *result = Tensor(memory, false, image_memory, tensor_desc);
  result->aligned_texture_width_ = AlignByN(width, width_pixel_alignment);
  return absl::OkStatus();
}

absl::Status AllocateTensorMemory(const CLContext& context,
                                  const TensorDescriptor& tensor_desc,
                                  CLMemory* result) {
  return AllocateTensorMemoryInternal(context, tensor_desc, result);
}

}  // namespace cl
}  // namespace ml_drift
