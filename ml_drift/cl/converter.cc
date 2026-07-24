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

#include "ml_drift/cl/converter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_errors.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/program_cache.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/cl/tensor_type_util.h"
#include "ml_drift/common/api_cl_gl_vk.h"
#include "ml_drift/common/api_common.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conversion.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/spi.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace cl {
namespace {

bool IsSupportedDataType(DataType type) {
  return type == DataType::FLOAT16 || type == DataType::FLOAT32 ||
         type == DataType::INT32 || type == DataType::BOOL ||
         type == DataType::BFLOAT16 || type == DataType::INT16 ||
         type == DataType::UINT8 || type == DataType::INT8;
}

bool IsBHWCOpenCLBuffer(const ObjectDef& def) {
  return IsSupportedDataType(def.data_type) &&
         def.object_type == ObjectType::OPENCL_BUFFER &&
         def.data_layout == DataLayout::BHWC;
}

bool IsOpenCLTensor(const ObjectDef& def) {
  const bool is_buffer_tensor = def.object_type == ObjectType::OPENCL_BUFFER &&
                                def.data_layout == DataLayout::DHWC4;
  const bool is_image2d_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::HDWC4;
  const bool is_image2d_array_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::DHWC4;
  const bool is_single_image_tensor =
      def.object_type == ObjectType::OPENCL_TEXTURE &&
      def.data_layout == DataLayout::BHWC;
  return IsSupportedDataType(def.data_type) &&
         (is_buffer_tensor || is_image2d_tensor || is_image2d_array_tensor ||
          is_single_image_tensor);
}

absl::Status GetOpenCLMemory(const TensorObject& obj, cl_mem* memory) {
  auto texture = std::get_if<OpenClTexture>(&obj);
  auto buffer = std::get_if<OpenClBuffer>(&obj);
  if (texture && texture->memobj) {
    *memory = texture->memobj;
  } else if (buffer && buffer->memobj) {
    *memory = buffer->memobj;
  } else {
    return absl::InvalidArgumentError("Missing OpenCL object.");
  }
  return absl::OkStatus();
}

}  // namespace

bool TensorToTensorConverter::IsSupported(const ObjectDef& input,
                                          const ObjectDef& output) {
  return IsOpenCLTensor(input) && IsOpenCLTensor(output);
}

absl::Status TensorToTensorConverter::InitExplicit(
    const CLDevice* device, CLContext* context, ProgramCache* cache,
    const TensorDescriptor& src_desc, const TensorDescriptor& dst_desc) {
  const GpuInfo& gpu_info = device->GetInfo();
  TensorToTensor gpu_op = CreateTensorToTensorOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  op_.Init(std::make_unique<GPUOperation>(std::move(gpu_op)));
  ABSL_RETURN_IF_ERROR(op_.Compile(device, context, cache));
  return absl::OkStatus();
}

absl::Status TensorToTensorConverter::Init(const TensorObjectDef& input_def,
                                           const TensorObjectDef& output_def,
                                           Environment* environment) {
  src_tensor_descriptor_ =
      TensorDescriptor(input_def.object_def.data_type,
                       ToTensorStorageType(input_def.object_def.object_type,
                                           input_def.object_def.data_layout),
                       Layout::BHWC);

  dst_tensor_descriptor_ =
      TensorDescriptor(output_def.object_def.data_type,
                       ToTensorStorageType(output_def.object_def.object_type,
                                           output_def.object_def.data_layout),
                       Layout::BHWC);

  shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                input_def.dimensions.w, input_def.dimensions.c);
  queue_ = environment->queue();
  context_ = &environment->context();
  return InitExplicit(environment->GetDevicePtr(), &environment->context(),
                      environment->program_cache(), src_tensor_descriptor_,
                      dst_tensor_descriptor_);
}

absl::Status TensorToTensorConverter::ConvertExplicit(CLCommandQueue* queue,
                                                      Tensor* src, Tensor* dst,
                                                      CLEvent* event) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(0, src));
  ABSL_RETURN_IF_ERROR(op_.SetDstTensor(0, dst));
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  return op_.AddToQueue(queue, event);
}

absl::Status TensorToTensorConverter::Convert(const TensorObject& input_obj,
                                              const TensorObject& output_obj) {
  cl_mem in_memory;
  ABSL_RETURN_IF_ERROR(GetOpenCLMemory(input_obj, &in_memory));
  cl_mem out_memory;
  ABSL_RETURN_IF_ERROR(GetOpenCLMemory(output_obj, &out_memory));

  Tensor src_tensor;
  TensorDescriptor descriptor_with_shape = src_tensor_descriptor_;
  descriptor_with_shape.SetBHWCShape(shape_);
  ABSL_RETURN_IF_ERROR(CreateTensorShared(*context_, in_memory,
                                          descriptor_with_shape, &src_tensor));
  Tensor dst_tensor;
  descriptor_with_shape = dst_tensor_descriptor_;
  descriptor_with_shape.SetBHWCShape(shape_);
  ABSL_RETURN_IF_ERROR(CreateTensorShared(*context_, out_memory,
                                          descriptor_with_shape, &dst_tensor));
  return ConvertExplicit(queue_, &src_tensor, &dst_tensor);
}

bool TensorToBHWCBufferConverter::IsSupported(const ObjectDef& input,
                                              const ObjectDef& output) {
  return IsOpenCLTensor(input) && IsBHWCOpenCLBuffer(output);
}

absl::Status TensorToBHWCBufferConverter::InitExplicit(
    const CLDevice* device, CLContext* context, ProgramCache* cache,
    const TensorDescriptor& src_desc, const BufferDescriptor& dst_desc) {
  const GpuInfo& gpu_info = device->GetInfo();
  TensorToBhwcBuffer gpu_op =
      CreateTensorToBhwcBufferOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, src_desc.GetBHWCShape());

  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  op_.Init(std::make_unique<TensorToBhwcBuffer>(std::move(gpu_op)));
  ABSL_RETURN_IF_ERROR(op_.Compile(device, context, cache));
  return absl::OkStatus();
}

absl::Status TensorToBHWCBufferConverter::Init(
    const TensorObjectDef& input_def, const TensorObjectDef& output_def,
    Environment* environment) {
  TensorStorageType src_tensor_type = ToTensorStorageType(
      input_def.object_def.object_type, input_def.object_def.data_layout);
  DataType src_data_type = input_def.object_def.data_type;
  DataType dst_data_type = output_def.object_def.data_type;
  if (src_data_type == DataType::BFLOAT16 &&
      dst_data_type == DataType::BFLOAT16) {
    src_data_type = DataType::UINT16;
    dst_data_type = DataType::UINT16;
  }
  src_tensor_descriptor_ =
      TensorDescriptor(src_data_type, src_tensor_type, Layout::BHWC);

  dst_buffer_descriptor_.element_type = dst_data_type;
  dst_buffer_descriptor_.element_size = 1;
  dst_buffer_descriptor_.memory_type = MemoryType::GLOBAL;

  shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                input_def.dimensions.w, input_def.dimensions.c);

  queue_ = environment->queue();
  context_ = &environment->context();
  return InitExplicit(environment->GetDevicePtr(), &environment->context(),
                      environment->program_cache(), src_tensor_descriptor_,
                      dst_buffer_descriptor_);
}

absl::Status TensorToBHWCBufferConverter::Convert(
    const TensorObject& input_obj, const TensorObject& output_obj) {
  auto output = std::get_if<OpenClBuffer>(&output_obj);
  if (!output || !output->memobj) {
    return absl::InvalidArgumentError(
        "Missing output in tensor_to_bhwc converter");
  }

  cl_mem in_memory;
  ABSL_RETURN_IF_ERROR(GetOpenCLMemory(input_obj, &in_memory));
  Tensor tensor;
  TensorDescriptor descriptor_with_shape = src_tensor_descriptor_;
  descriptor_with_shape.SetBHWCShape(shape_);
  ABSL_RETURN_IF_ERROR(
      CreateTensorShared(*context_, in_memory, descriptor_with_shape, &tensor));
  Buffer buffer = CreateBufferShared(output->memobj);

  return ConvertExplicit(queue_, &tensor, &buffer);
}

absl::Status TensorToBHWCBufferConverter::ConvertExplicit(CLCommandQueue* queue,
                                                          Tensor* src,
                                                          Buffer* dst,
                                                          CLEvent* event) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(0, src));
  ABSL_RETURN_IF_ERROR(op_.SetDstBuffer(0, dst));
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  return op_.AddToQueue(queue, event);
}

bool BHWCBufferToTensorConverter::IsSupported(const ObjectDef& input,
                                              const ObjectDef& output) {
  return IsBHWCOpenCLBuffer(input) && IsOpenCLTensor(output);
}

absl::Status BHWCBufferToTensorConverter::InitExplicit(
    const CLDevice* device, CLContext* context, ProgramCache* cache,
    const BufferDescriptor& src_desc, const TensorDescriptor& dst_desc) {
  const GpuInfo& gpu_info = device->GetInfo();
  BhwcBufferToTensor gpu_op =
      CreateBhwcBufferToTensorOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  op_.Init(std::make_unique<GPUOperation>(std::move(gpu_op)));
  ABSL_RETURN_IF_ERROR(op_.Compile(device, context, cache));
  return absl::OkStatus();
}

absl::Status BHWCBufferToTensorConverter::Init(
    const TensorObjectDef& input_def, const TensorObjectDef& output_def,
    Environment* environment) {
  DataType src_data_type = input_def.object_def.data_type;
  DataType dst_data_type = output_def.object_def.data_type;
  if (src_data_type == DataType::BFLOAT16 &&
      dst_data_type == DataType::BFLOAT16) {
    src_data_type = DataType::UINT16;
    dst_data_type = DataType::UINT16;
  }

  src_buffer_descriptor_.element_type = src_data_type;
  src_buffer_descriptor_.element_size = 1;
  src_buffer_descriptor_.memory_type = MemoryType::GLOBAL;

  TensorStorageType dst_tensor_type = ToTensorStorageType(
      output_def.object_def.object_type, output_def.object_def.data_layout);
  dst_tensor_descriptor_ = TensorDescriptor(dst_data_type,

                                            dst_tensor_type, Layout::BHWC);

  shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                output_def.dimensions.w, output_def.dimensions.c);

  queue_ = environment->queue();
  context_ = &environment->context();
  return InitExplicit(environment->GetDevicePtr(), &environment->context(),
                      environment->program_cache(), src_buffer_descriptor_,
                      dst_tensor_descriptor_);
}

absl::Status BHWCBufferToTensorConverter::ConvertExplicit(CLCommandQueue* queue,
                                                          Buffer* src,
                                                          Tensor* dst,
                                                          CLEvent* event) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcBuffer(0, src));
  ABSL_RETURN_IF_ERROR(op_.SetDstTensor(0, dst));
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  return op_.AddToQueue(queue, event);
}

absl::Status BHWCBufferToTensorConverter::Convert(
    const TensorObject& input_obj, const TensorObject& output_obj) {
  auto input = std::get_if<OpenClBuffer>(&input_obj);
  if (!input || !input->memobj) {
    return absl::InvalidArgumentError(
        "Missing input in bhwc_to_tensor converter");
  }
  cl_mem out_memory;
  ABSL_RETURN_IF_ERROR(GetOpenCLMemory(output_obj, &out_memory));
  Tensor dst_tensor;
  TensorDescriptor descriptor_with_shape = dst_tensor_descriptor_;
  descriptor_with_shape.SetBHWCShape(shape_);
  ABSL_RETURN_IF_ERROR(CreateTensorShared(*context_, out_memory,
                                          descriptor_with_shape, &dst_tensor));
  Buffer src_buffer = CreateBufferShared(input->memobj);
  return ConvertExplicit(queue_, &src_buffer, &dst_tensor);
}

namespace {

std::array<size_t, 3> CalculateTextureRegion(const TensorObjectDef& def) {
  const auto& dims = def.dimensions;
  std::array<size_t, 3> region = {0, 0, 1};
  switch (ToTensorStorageType(def.object_def.object_type,
                              def.object_def.data_layout)) {
    case TensorStorageType::SINGLE_TEXTURE_2D:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h);
      break;
    case TensorStorageType::TEXTURE_2D:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h * dims.d());
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      region[0] = static_cast<size_t>(dims.w * dims.b);
      region[1] = static_cast<size_t>(dims.h);
      region[2] = static_cast<size_t>(dims.d());
      break;
    default:
      break;
  }
  return region;
}

bool IsOpenClTextureOrBuffer(ObjectType type) {
  return type == ObjectType::OPENCL_BUFFER ||
         type == ObjectType::OPENCL_TEXTURE;
}

// Copies data from one object of the same type and layout to another object.
class TrivialCopier : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return IsOpenClTextureOrBuffer(input.object_type) &&
           input.data_type == output.data_type &&
           input.object_type == output.object_type &&
           input.data_layout == output.data_layout;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    shape_ = BHWC(input_def.dimensions.b, input_def.dimensions.h,
                  input_def.dimensions.w, input_def.dimensions.c);
    data_type_ = input_def.object_def.data_type;
    queue_ = environment->queue();
    region_ = CalculateTextureRegion(output_def);
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto texture_input = std::get_if<OpenClTexture>(&input_obj);
    auto texture_output = std::get_if<OpenClTexture>(&output_obj);
    if (texture_input && texture_output) {
      return Copy(*texture_input, *texture_output);
    }
    auto buffer_input = std::get_if<OpenClBuffer>(&input_obj);
    auto buffer_output = std::get_if<OpenClBuffer>(&output_obj);
    if (buffer_input && buffer_output) {
      return Copy(*buffer_input, *buffer_output);
    }
    return absl::InternalError("Unexpected object");
  }

  absl::Status Copy(const OpenClBuffer& input, const OpenClBuffer& output) {
    if (input.memobj == output.memobj) {
      return absl::OkStatus();
    }
    return GetOpenCLError(
        clEnqueueCopyBuffer(queue_->queue(), input.memobj, output.memobj, 0, 0,
                            SizeOf(data_type_) * shape_.w * shape_.h *
                                AlignByN(shape_.c, 4) * shape_.b,
                            0, nullptr, nullptr));
  }

  absl::Status Copy(const OpenClTexture& input, const OpenClTexture& output) {
    if (input.memobj == output.memobj) {
      return absl::OkStatus();
    }
    size_t origin[3] = {0, 0, 0};
    return GetOpenCLError(
        clEnqueueCopyImage(queue_->queue(), input.memobj, output.memobj, origin,
                           origin, region_.data(), 0, nullptr, nullptr));
  }

 private:
  DataType data_type_ = DataType::UNKNOWN;
  std::array<size_t, 3> region_;
};

// Copies data from/to CPU into a tensor.
class CpuCopier : public OpenClConverterImpl {
 public:
  explicit CpuCopier(bool asynchronous = false) : async_(asynchronous) {}
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::CPU_MEMORY &&
             IsOpenClTextureOrBuffer(output.object_type)) ||
            (output.object_type == ObjectType::CPU_MEMORY &&
             IsOpenClTextureOrBuffer(input.object_type)));
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final {
    region_ = CalculateTextureRegion(
        input_def.object_def.object_type == ObjectType::CPU_MEMORY ? output_def
                                                                   : input_def);
    input_data_type_ = input_def.object_def.data_type;
    output_data_type_ = output_def.object_def.data_type;
    queue_ = environment->queue();
    return absl::OkStatus();
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
    auto cpu_input = std::get_if<CpuMemory>(&input_obj);
    auto cpu_output = std::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      if (output_data_type_ == DataType::BOOL) {
        return CopyFromBoolCpu(cpu_input, output_obj);
      }
      auto texture_output = std::get_if<OpenClTexture>(&output_obj);
      if (texture_output) {
        return queue_->EnqueueWriteImage(
            texture_output->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_input->data, async_);
      }
      auto buffer_output = std::get_if<OpenClBuffer>(&output_obj);
      if (buffer_output) {
        return queue_->EnqueueWriteBuffer(buffer_output->memobj,
                                          cpu_input->size_bytes,
                                          cpu_input->data, async_);
      }
    } else if (cpu_output) {
      if (input_data_type_ == DataType::BOOL) {
        return CopyToBoolCpu(input_obj, cpu_output);
      }
      auto texture_input = std::get_if<OpenClTexture>(&input_obj);
      if (texture_input) {
        return queue_->EnqueueReadImage(
            texture_input->memobj, int3(region_[0], region_[1], region_[2]),
            cpu_output->data, async_);
      }
      auto buffer_input = std::get_if<OpenClBuffer>(&input_obj);
      if (buffer_input) {
        return queue_->EnqueueReadBuffer(buffer_input->memobj,
                                         cpu_output->size_bytes,
                                         cpu_output->data, async_);
      }
    }
    return absl::InternalError("Unexpected object");
  }

 private:
  absl::Status CopyToBoolCpu(const TensorObject& tensor_obj,
                             const CpuMemory* cpu_memory) {
    const size_t num_elements = cpu_memory->size_bytes;
    std::vector<uint8_t> tmp_data(num_elements);
    auto texture_input = std::get_if<OpenClTexture>(&tensor_obj);
    if (texture_input) {
      ABSL_RETURN_IF_ERROR(queue_->EnqueueReadImage(
          texture_input->memobj, int3(region_[0], region_[1], region_[2]),
          tmp_data.data(), false));
    } else {
      auto buffer_input = std::get_if<OpenClBuffer>(&tensor_obj);
      if (!buffer_input) {
        return absl::InternalError("Unexpected object");
      }
      ABSL_RETURN_IF_ERROR(queue_->EnqueueReadBuffer(
          buffer_input->memobj, tmp_data.size(), tmp_data.data(), false));
    }
    bool* output_data = reinterpret_cast<bool*>(cpu_memory->data);
    for (int i = 0; i < num_elements; ++i) {
      output_data[i] = tmp_data[i];
    }
    return absl::OkStatus();
  }

  absl::Status CopyFromBoolCpu(const CpuMemory* cpu_memory,
                               const TensorObject& tensor_obj) {
    const size_t num_elements = cpu_memory->size_bytes;
    const bool* bool_data = reinterpret_cast<bool*>(cpu_memory->data);
    copy_from_bool_data_.resize(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      copy_from_bool_data_[i] = bool_data[i];
    }
    auto texture_output = std::get_if<OpenClTexture>(&tensor_obj);
    if (texture_output) {
      return queue_->EnqueueWriteImage(texture_output->memobj,
                                       int3(region_[0], region_[1], region_[2]),
                                       copy_from_bool_data_.data(), async_);
    }
    auto buffer_output = std::get_if<OpenClBuffer>(&tensor_obj);
    if (buffer_output) {
      return queue_->EnqueueWriteBuffer(buffer_output->memobj,
                                        copy_from_bool_data_.size(),
                                        copy_from_bool_data_.data(), async_);
    }
    return absl::InternalError("Unexpected object");
  }

  std::vector<uint8_t>
      copy_from_bool_data_;  // stays here to be alive when async write will be
                             // actually executed.
  std::array<size_t, 3> region_;
  bool async_;
  DataType input_data_type_;
  DataType output_data_type_;
};

class OpenClTensorConverterBuilder : public TensorObjectConverterBuilder {
 public:
  explicit OpenClTensorConverterBuilder(Environment* environment)
      : environment_(environment) {}

  bool IsSupported(const TensorObjectDef& input,
                   const TensorObjectDef& output) const final {
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    return input.dimensions == output.dimensions &&
           (TrivialCopier::IsSupported(input_def, output_def) ||
            TensorToTensorConverter::IsSupported(input_def, output_def) ||
            CpuCopier::IsSupported(input_def, output_def) ||
            TensorToBHWCBufferConverter::IsSupported(input_def, output_def) ||
            BHWCBufferToTensorConverter::IsSupported(input_def, output_def));
  }

  absl::Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) final {
    std::unique_ptr<OpenClConverterImpl> impl;
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    if (TrivialCopier::IsSupported(input_def, output_def)) {
      impl = std::make_unique<TrivialCopier>();
    } else if (TensorToTensorConverter::IsSupported(input_def, output_def)) {
      impl = std::make_unique<TensorToTensorConverter>();
    } else if (CpuCopier::IsSupported(input_def, output_def)) {
      impl = std::make_unique<CpuCopier>(/*asynchronous*/ true);
    } else if (TensorToBHWCBufferConverter::IsSupported(input_def,
                                                        output_def)) {
      impl = std::make_unique<TensorToBHWCBufferConverter>();
    } else if (BHWCBufferToTensorConverter::IsSupported(input_def,
                                                        output_def)) {
      impl = std::make_unique<BHWCBufferToTensorConverter>();
    } else {
      return absl::UnimplementedError("Unsupported conversion");
    }
    ABSL_RETURN_IF_ERROR(impl->Init(input, output, environment_));
    *converter = std::move(impl);
    return absl::OkStatus();
  }

  Environment* environment_;
};

}  // namespace

std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    Environment* environment) {
  return std::make_unique<OpenClTensorConverterBuilder>(environment);
}

}  // namespace cl
}  // namespace ml_drift
