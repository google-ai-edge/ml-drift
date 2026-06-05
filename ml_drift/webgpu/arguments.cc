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

#include "ml_drift/webgpu/arguments.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"  // IWYU pragma: keep
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"  // IWYU pragma: keep

namespace ml_drift {
namespace webgpu {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
  size_t position = str->find(old_word);
  while (position != std::string::npos) {
    char prev = position == 0 ? '.' : (*str)[position - 1];
    char next = position + old_word.size() < str->size()
                    ? (*str)[position + old_word.size()]
                    : '.';
    if (IsWordSymbol(prev) || IsWordSymbol(next)) {
      position = str->find(old_word, position + 1);
      continue;
    }
    str->replace(position, old_word.size(), new_word);
    position = str->find(old_word, position + new_word.size());
  }
}

absl::Status CreateWebGpuObject(const wgpu::Device& device,
                                GPUObjectDescriptor* desc,
                                GpuObjectPtr* result) {
  const BufferDescriptor* buffer_desc = AsBufferDescriptor(desc);
  if (buffer_desc) {
    Buffer gpu_buffer;
    gpu_buffer.CreateFromBufferDescriptor(device, *buffer_desc);
    *result = std::make_unique<Buffer>(std::move(gpu_buffer));
    return absl::OkStatus();
  }

  const TensorDescriptor* tensor_desc = AsTensorDescriptor(desc);
  if (tensor_desc) {
    SpatialTensor gpu_tensor;
    RETURN_IF_ERROR(gpu_tensor.CreateFromDescriptor(device, *tensor_desc));
    *result = std::make_unique<SpatialTensor>(std::move(gpu_tensor));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unknown GPU descriptor.");
}

std::string DataTypeToWgslType(DataType data_type, int vec_size,
                               bool supports_fp16) {
  if (data_type == DataType::FLOAT32) {
    if (vec_size == 1) {
      return "f32";
    } else if (vec_size == 2) {
      return "vec2<f32>";
    } else if (vec_size == 4) {
      return "vec4<f32>";
    } else if (vec_size == 16) {
      return "mat4x4<f32>";
    }
  } else if (data_type == DataType::FLOAT16) {
    if (supports_fp16) {
      if (vec_size == 1) {
        return "f16";
      } else if (vec_size == 2) {
        return "vec2<f16>";
      } else if (vec_size == 4) {
        return "vec4<f16>";
      } else if (vec_size == 16) {
        return "mat4x4<f16>";
      }
    } else {
      if (vec_size == 2) {
        return "u32";
      } else if (vec_size == 4) {
        return "vec2<u32>";
      } else if (vec_size == 8) {
        return "vec4<u32>";
      } else if (vec_size == 16) {
        return "vec4<u32>";
      }
    }
  } else if (data_type == DataType::INT32) {
    if (vec_size == 1) {
      return "i32";
    } else if (vec_size == 2) {
      return "vec2<i32>";
    } else if (vec_size == 4) {
      return "vec4<i32>";
    }
  } else if (data_type == DataType::INT16) {
    if (vec_size == 2) {
      return "i32";
    } else if (vec_size == 4) {
      return "vec2<i32>";
    } else if (vec_size == 8) {
      return "vec4<i32>";
    }
  } else if (data_type == DataType::INT8) {
    if (vec_size == 4) {
      return "i32";
    } else if (vec_size == 8) {
      return "vec2<i32>";
    } else if (vec_size == 16) {
      return "vec4<i32>";
    }
  } else if (data_type == DataType::UINT32) {
    if (vec_size == 1) {
      return "u32";
    } else if (vec_size == 2) {
      return "vec2<u32>";
    } else if (vec_size == 4) {
      return "vec4<u32>";
    }
  } else if (data_type == DataType::UINT16 || data_type == DataType::BFLOAT16) {
    if (vec_size == 2) {
      return "u32";
    } else if (vec_size == 4) {
      return "vec2<u32>";
    } else if (vec_size == 8) {
      return "vec4<u32>";
    }
  } else if (data_type == DataType::UINT8 || data_type == DataType::BOOL) {
    if (vec_size == 4) {
      return "u32";
    } else if (vec_size == 8) {
      return "vec2<u32>";
    } else if (vec_size == 16) {
      return "vec4<u32>";
    }
  }
  return "";
}

std::string DataTypeToTextureShaderType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
      return "rgba32float";
    case DataType::FLOAT16:
      return "rgba16float";
    case DataType::UINT32:
      return "rgba32uint";
    case DataType::BFLOAT16:
    case DataType::UINT16:
      return "rgba16uint";
    case DataType::UINT8:
      return "rgba8uint";
    case DataType::INT32:
      return "rgba32sint";
    case DataType::INT16:
      return "rgba16sint";
    case DataType::INT8:
      return "rgba8sint";
    case DataType::BOOL:
      return "rgba8uint";
    default:
      return "";
  }
}

wgpu::TextureSampleType DataTypeToTextureSampleType(DataType data_type) {
  switch (data_type) {
    case ml_drift::DataType::UNKNOWN:
    case ml_drift::DataType::UINT4:
    case ml_drift::DataType::UINT3:
    case ml_drift::DataType::UINT2:
    case ml_drift::DataType::UINT1:
    case ml_drift::DataType::INT4:
    case ml_drift::DataType::INT3:
    case ml_drift::DataType::INT2:
    case ml_drift::DataType::INT1:
      return wgpu::TextureSampleType::BindingNotUsed;
    case ml_drift::DataType::BOOL:
      return wgpu::TextureSampleType::Uint;
    case ml_drift::DataType::FLOAT16:
    case ml_drift::DataType::FLOAT32:
    case ml_drift::DataType::FLOAT64:
      return wgpu::TextureSampleType::UnfilterableFloat;
    case ml_drift::DataType::BFLOAT16:
    case ml_drift::DataType::UINT8:
    case ml_drift::DataType::UINT16:
    case ml_drift::DataType::UINT32:
    case ml_drift::DataType::UINT64:
      return wgpu::TextureSampleType::Uint;
    case ml_drift::DataType::INT8:
    case ml_drift::DataType::INT16:
    case ml_drift::DataType::INT32:
    case ml_drift::DataType::INT64:
      return wgpu::TextureSampleType::Sint;
  }
  return wgpu::TextureSampleType::Undefined;
}

std::string TextureSampleTypeToString(wgpu::TextureSampleType sample_type) {
  switch (sample_type) {
    case wgpu::TextureSampleType::Float:
    case wgpu::TextureSampleType::UnfilterableFloat:
      return "f32";
    case wgpu::TextureSampleType::Sint:
      return "i32";
    case wgpu::TextureSampleType::Uint:
      return "u32";
    default:
      return "Undefined";
  }
}

}  // namespace

// Static
constexpr char WebGpuArguments::kArgsPrefix[];

absl::Status WebGpuArguments::Init(
    const wgpu::Device& device, const GpuInfo& gpu_info, Arguments* args,
    std::string* code, UniformBufferCreator* uniform_buffer_creator) {
  if (!pipeline_layout_.bindings_.empty()) {
    return absl::InvalidArgumentError(
        "WebGpuArguments::Init() is called twice.");
  }

  RETURN_IF_ERROR(AllocateObjects(device, *args));
  RETURN_IF_ERROR(AddObjectArgs(gpu_info, *args));
  std::map<std::string, GPUObjectDescriptorPtr, std::less<>> object_refs;
  args->MoveObjectRefs(&object_refs);
  for (auto& [name, ref] : object_refs) {
    object_refs_[name] = std::move(ref);
  }
  std::string struct_desc = ScalarArgumentsToStructWithVec4Fields(
      *args, gpu_info.webgpu_info.supports_fp16, code);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  args->ResolveArgsPass(code);
  std::string arguments = ToWgslArguments(gpu_info.webgpu_info.supports_fp16);
  if (!const_data_.empty()) {
    const int binding_index = pipeline_layout_.AddUniformBufferBinding("U");
    arguments += struct_desc + "\n@group(0) @binding(" +
                 std::to_string(binding_index) + ") var<uniform> U: Scalars;\n";
    if (uniform_buffer_creator) {
      scalars_ = uniform_buffer_creator->CreateUniformBuffer(
          device, const_data_.size());
    } else {
      scalars_ =
          CreateBufferUniform(device, const_data_.size(), /*data_ptr=*/nullptr);
    }
    RETURN_IF_ERROR(UpdateScalars(device));
  }
  *code = arguments + *code;
  RETURN_IF_ERROR(AssembleLayout(device));
  return absl::OkStatus();
}

absl::Status WebGpuArguments::InitWithoutCodeGeneration(
    const wgpu::Device& device, const GpuInfo& gpu_info, Arguments* args,
    UniformBufferCreator* uniform_buffer_creator) {
  if (!pipeline_layout_.bindings_.empty()) {
    return absl::InvalidArgumentError(
        "WebGpuArguments::Init() is called twice.");
  }

  RETURN_IF_ERROR(AllocateObjects(device, *args));
  RETURN_IF_ERROR(AddObjectArgs(gpu_info, *args));
  std::map<std::string, GPUObjectDescriptorPtr, std::less<>> object_refs;
  args->MoveObjectRefs(&object_refs);
  for (auto& [name, ref] : object_refs) {
    object_refs_[name] = std::move(ref);
  }
  std::string temp_code;
  ScalarArgumentsToStructWithVec4Fields(
      *args, gpu_info.webgpu_info.supports_fp16, &temp_code);
  RETURN_IF_ERROR(SetObjectsResources(*args));

  // Needs to be called to update the pipeline_layout_.
  ToWgslArguments(gpu_info.webgpu_info.supports_fp16);
  if (!const_data_.empty()) {
    pipeline_layout_.AddUniformBufferBinding("U");
    if (uniform_buffer_creator) {
      scalars_ = uniform_buffer_creator->CreateUniformBuffer(
          device, const_data_.size());
    } else {
      scalars_ =
          CreateBufferUniform(device, const_data_.size(), /*data_ptr=*/nullptr);
    }
    RETURN_IF_ERROR(UpdateScalars(device));
  }
  RETURN_IF_ERROR(AssembleLayout(device));
  return absl::OkStatus();
}

absl::Status WebGpuArguments::AssembleLayout(const wgpu::Device& device) {
  return pipeline_layout_.Assemble(device);
}

absl::Status WebGpuArguments::UpdateScalars(const wgpu::Device& device) {
  if (const_data_ == last_uploaded_const_data_) {
    return absl::OkStatus();
  }
  last_uploaded_const_data_ = const_data_;
  return scalars_.WriteData(device.GetQueue(), absl::MakeSpan(const_data_));
}

void WebGpuArguments::UpdateBindingsOnGPU(const wgpu::Device& device) {
  pipeline_layout_.UpdateBindingsOnGPU(device, buffers_, images2d_,
                                       image2d_arrays_, images3d_, scalars_);
}

void WebGpuArguments::Bind(const wgpu::ComputePassEncoder& encoder) const {
  encoder.SetBindGroup(0, pipeline_layout_.bind_group_);
}

std::string WebGpuArguments::ScalarArgumentsToStructWithVec4Fields(
    const Arguments& args, bool explicit_fp16, std::string* code) {
  std::string struct_desc = "struct Scalars {\n";
  int pos = 0;
  constexpr const char* channels[] = {".x", ".y", ".z", ".w"};
  for (const auto& fvalue : args.GetFloatValues()) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        absl::StrAppend(&struct_desc, "  f", pos / 4, " : vec4<f32>,\n");
      }
      std::string new_name = absl::StrCat("U.f", pos / 4, channels[pos % 4]);
      ReplaceAllWords(kArgsPrefix + fvalue.first, new_name, code);
      pos++;
    }
  }
  for (const auto& hfvalue : args.GetHalfValues()) {
    auto& new_val = float_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        absl::StrAppend(&struct_desc, "  f", pos / 4, " : vec4<f32>,\n");
      }
      std::string new_name = absl::StrCat("U.f", pos / 4, channels[pos % 4]);
      if (explicit_fp16) {
        new_name = absl::StrCat("f16(", new_name, ")");
      }
      ReplaceAllWords(kArgsPrefix + hfvalue.first, new_name, code);
      pos++;
    }
  }
  pos = AlignByN(pos, 4);
  for (const auto& ivalue : args.GetIntValues()) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        absl::StrAppend(&struct_desc, "  i", pos / 4, " : vec4<i32>,\n");
      }
      std::string new_name = absl::StrCat("U.i", pos / 4, channels[pos % 4]);
      ReplaceAllWords(kArgsPrefix + ivalue.first, new_name, code);
      pos++;
    }
  }
  pos = AlignByN(pos, 4);
  for (const auto& uivalue : args.GetUintValues()) {
    auto& new_val = uint_values_[uivalue.first];
    new_val.value = uivalue.second.value;
    new_val.active = uivalue.second.active;
    if (uivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        absl::StrAppend(&struct_desc, "  u", pos / 4, " : vec4<u32>,\n");
      }
      std::string new_name = absl::StrCat("U.u", pos / 4, channels[pos % 4]);
      ReplaceAllWords(kArgsPrefix + uivalue.first, new_name, code);
      pos++;
    }
  }
  if (pos != 0) {
    int aligned_pos = AlignByN(pos, 4);
    absl::StrAppend(&struct_desc, "};");
    const_data_.resize(aligned_pos * 4);
    for (auto& it : float_values_) {
      if (it.second.active) {
        float* ptr =
            reinterpret_cast<float*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
    for (auto& it : int_values_) {
      if (it.second.active) {
        int32_t* ptr =
            reinterpret_cast<int32_t*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
    for (auto& it : uint_values_) {
      if (it.second.active) {
        uint32_t* ptr =
            reinterpret_cast<uint32_t*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
  } else {
    struct_desc = "";
  }
  return struct_desc;
}

std::string WebGpuArguments::ToWgslArguments(bool supports_fp16) {
  std::string result;
  for (auto& t : images2d_) {
    if (t.second.desc.access_type == AccessType::WRITE) {
      wgpu::TextureFormat texture_format =
          DataTypeToTextureFormat(t.second.desc.data_type);
      wgpu::TextureViewDimension view_dimension =
          wgpu::TextureViewDimension::e2D;
      const int binding_index = pipeline_layout_.AddStorageTextureBinding(
          t.first, texture_format, view_dimension);
      const std::string shader_type =
          DataTypeToTextureShaderType(t.second.desc.data_type);
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var " + t.first + " : texture_storage_2d<" + shader_type +
                ", write>;\n";
    } else {
      wgpu::TextureViewDimension view_dimension =
          wgpu::TextureViewDimension::e2D;
      const auto texture_sample_type =
          DataTypeToTextureSampleType(t.second.desc.data_type);
      const int binding_index = pipeline_layout_.AddTextureBinding(
          t.first, view_dimension, texture_sample_type);
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var " + t.first + " : texture_2d<" +
                TextureSampleTypeToString(texture_sample_type) + ">;\n";
    }
  }
  for (auto& t : images3d_) {
    if (t.second.desc.access_type == AccessType::WRITE) {
      wgpu::TextureFormat texture_format =
          DataTypeToTextureFormat(t.second.desc.data_type);
      wgpu::TextureViewDimension view_dimension =
          wgpu::TextureViewDimension::e3D;
      const int binding_index = pipeline_layout_.AddStorageTextureBinding(
          t.first, texture_format, view_dimension);
      const std::string shader_type =
          DataTypeToTextureShaderType(t.second.desc.data_type);
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var " + t.first + " : texture_storage_3d<" + shader_type +
                ", write>;\n";
    } else {
      wgpu::TextureViewDimension view_dimension =
          wgpu::TextureViewDimension::e3D;
      const auto texture_sample_type =
          DataTypeToTextureSampleType(t.second.desc.data_type);
      const int binding_index = pipeline_layout_.AddTextureBinding(
          t.first, view_dimension, texture_sample_type);
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var " + t.first + " : texture_3d<" +
                TextureSampleTypeToString(texture_sample_type) + ">;\n";
    }
  }
  for (auto& t : image2d_arrays_) {
    if (t.second.desc.access_type == AccessType::WRITE) {
      wgpu::TextureFormat texture_format =
          DataTypeToTextureFormat(t.second.desc.data_type);
      wgpu::TextureViewDimension view_dimension =
          wgpu::TextureViewDimension::e2DArray;
      const int binding_index = pipeline_layout_.AddStorageTextureBinding(
          t.first, texture_format, view_dimension);
      const std::string shader_type =
          DataTypeToTextureShaderType(t.second.desc.data_type);
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var " + t.first + " : texture_storage_2d_array<" +
                shader_type + ", write>;\n";
    } else {
      wgpu::TextureViewDimension view_dimension =
          wgpu::TextureViewDimension::e2DArray;
      const auto texture_sample_type =
          DataTypeToTextureSampleType(t.second.desc.data_type);
      const int binding_index = pipeline_layout_.AddTextureBinding(
          t.first, view_dimension, texture_sample_type);
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var " + t.first + " : texture_2d_array<" +
                TextureSampleTypeToString(texture_sample_type) + ">;\n";
    }
  }
  for (auto& b : buffers_) {
    if (b.second.desc.memory_type == MemoryType::CONSTANT) {
      const int binding_index =
          pipeline_layout_.AddUniformBufferBinding(b.first);
      const std::string wgsl_type = DataTypeToWgslType(
          b.second.desc.data_type, b.second.desc.element_size, supports_fp16);
      result += "struct " + b.first + "_vector {\n";
      result += "  data: array<" + wgsl_type + ", " +
                b.second.desc.attributes[0] + ">,\n";
      result += "};\n";
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var<uniform> " + b.first + " : " + b.first + "_vector;\n";
    } else {
      const int binding_index = pipeline_layout_.AddStorageBufferBinding(
          b.first, b.second.desc.access_type == AccessType::READ);
      const std::string wgsl_type = DataTypeToWgslType(
          b.second.desc.data_type, b.second.desc.element_size, supports_fp16);
      const std::string access_type =
          b.second.desc.access_type == AccessType::READ ? "read" : "read_write";
      result += "struct " + b.first + "_vector {\n";
      result += "  data: array<" + wgsl_type + ">,\n";
      result += "};\n";
      result += "@group(0) @binding(" + std::to_string(binding_index) +
                ") var<storage, " + access_type + "> " + b.first + " : " +
                b.first + "_vector;\n";
    }
  }
  return result;
}

absl::Status WebGpuArguments::SetInt(const std::string& name, int value) {
  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    int32_t* ptr =
        reinterpret_cast<int32_t*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}
absl::Status WebGpuArguments::SetUint(const std::string& name,
                                      unsigned int value) {
  auto it = uint_values_.find(name);
  if (it == uint_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No uint argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    uint32_t* ptr =
        reinterpret_cast<uint32_t*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}
absl::Status WebGpuArguments::SetFloat(const std::string& name, float value) {
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    float* ptr =
        reinterpret_cast<float*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}

absl::Status WebGpuArguments::SetHalf(const std::string& name, half value) {
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No half argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    float* ptr =
        reinterpret_cast<float*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}

absl::Status WebGpuArguments::AllocateObjects(const wgpu::Device& device,
                                              const Arguments& args) {
  objects_.resize(args.GetObjects().size());
  int i = 0;
  for (auto& t : args.GetObjects()) {
    GpuObjectPtr object;
    RETURN_IF_ERROR(CreateWebGpuObject(device, t.second.get(), &object));
    objects_[i] = std::move(object);
    i++;
  }
  return absl::OkStatus();
}

absl::Status WebGpuArguments::AddObjectArgs(const GpuInfo& gpu_info,
                                            const Arguments& args) {
  for (const auto& t : args.GetObjects()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  for (const auto& t : args.GetObjectRefs()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  return absl::OkStatus();
}

void WebGpuArguments::AddBuffer(const std::string& name,
                                const GPUBufferDescriptor& desc) {
  buffers_[name].desc = desc;
}

void WebGpuArguments::AddImage2D(const std::string& name,
                                 const GPUImage2DDescriptor& desc) {
  images2d_[name].desc = desc;
}

void WebGpuArguments::AddImage2DArray(const std::string& name,
                                      const GPUImage2DArrayDescriptor& desc) {
  image2d_arrays_[name].desc = desc;
}

void WebGpuArguments::AddImage3D(const std::string& name,
                                 const GPUImage3DDescriptor& desc) {
  images3d_[name].desc = desc;
}

void WebGpuArguments::AddGPUResources(const std::string& name,
                                      const GPUResources& resources) {
  for (const auto& r : resources.buffers) {
    AddBuffer(absl::StrCat(name, "_", r.first), r.second);
  }

  for (const auto& r : resources.images2d) {
    AddImage2D(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.image2d_arrays) {
    AddImage2DArray(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.images3d) {
    AddImage3D(absl::StrCat(name, "_", r.first), r.second);
  }
}

absl::Status WebGpuArguments::SetBuffer(const std::string& name,
                                        const wgpu::Buffer& handle, size_t size,
                                        size_t offset) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.handle = handle;
  it->second.size = size;
  it->second.offset = offset;
  return absl::OkStatus();
}

absl::Status WebGpuArguments::SetImage2D(const std::string& name,
                                         wgpu::TextureView handle) {
  auto it = images2d_.find(name);
  if (it == images2d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2d argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status WebGpuArguments::SetImage2DArray(const std::string& name,
                                              wgpu::TextureView handle) {
  auto it = image2d_arrays_.find(name);
  if (it == image2d_arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2d array argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status WebGpuArguments::SetImage3D(const std::string& name,
                                         wgpu::TextureView handle) {
  auto it = images3d_.find(name);
  if (it == images3d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image3d argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status WebGpuArguments::SetObjectRef(const std::string& name,
                                           const GpuObject& object) {
  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  GpuResourcesWithValue resources;
  RETURN_IF_ERROR(object.GetGPUResources(it->second.get(), &resources));
  return SetGPUResources(name, resources);
}

absl::Status WebGpuArguments::SetGPUResources(
    const std::string& name, const GpuResourcesWithValue& resources) {
  for (const auto& r : resources.generic.ints) {
    RETURN_IF_ERROR(SetInt(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.generic.uints) {
    RETURN_IF_ERROR(SetUint(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.generic.floats) {
    RETURN_IF_ERROR(SetFloat(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.buffers) {
    RETURN_IF_ERROR(SetBuffer(absl::StrCat(name, "_", r.first), r.second.buffer,
                              r.second.size, r.second.offset));
  }

  for (const auto& r : resources.images2d) {
    RETURN_IF_ERROR(SetImage2D(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.image2d_arrays) {
    RETURN_IF_ERROR(
        SetImage2DArray(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.images3d) {
    RETURN_IF_ERROR(SetImage3D(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
}

absl::Status WebGpuArguments::SetObjectsResources(const Arguments& args) {
  int i = 0;
  for (const auto& t : args.GetObjects()) {
    GpuResourcesWithValue resources;
    RETURN_IF_ERROR(objects_[i]->GetGPUResources(t.second.get(), &resources));
    RETURN_IF_ERROR(SetGPUResources(t.first, resources));
    i++;
  }
  return absl::OkStatus();
}

bool WebGpuArguments::HasFloat16Buffers() const {
  for (const auto& t : buffers_) {
    if (t.second.desc.data_type == DataType::FLOAT16) {
      return true;
    }
  }
  return false;
}

absl::Status WebGpuArguments::PipelineLayout::Assemble(
    const wgpu::Device& device) {
  std::vector<wgpu::BindGroupLayoutEntry> bindings;
  bindings.reserve(bindings_.size());
  for (const auto& it : bindings_) {
    bindings.push_back(it.second);
  }
  wgpu::BindGroupLayoutDescriptor layout_descriptor = {
      .entryCount = static_cast<uint32_t>(bindings.size()),
      .entries = bindings.data(),
  };
  group_layout_ = device.CreateBindGroupLayout(&layout_descriptor);
  wgpu::PipelineLayoutDescriptor pipeline_descriptor = {
      .bindGroupLayoutCount = 1u,
      .bindGroupLayouts = &group_layout_,
  };
  layout_ = device.CreatePipelineLayout(&pipeline_descriptor);
  return absl::OkStatus();
}

void WebGpuArguments::PipelineLayout::UpdateBindingsOnGPU(
    const wgpu::Device& device,
    const std::map<std::string, WebGpuBufferDescriptor>& buffers,
    const std::map<std::string, WebGpuImage2DDescriptor>& images2d,
    const std::map<std::string, WebGpuImage2DArrayDescriptor>& image2d_arrays,
    const std::map<std::string, WebGpuImage3DDescriptor>& images3d,
    const Buffer& scalars) {
  if (bindings_.empty()) {
    return;
  }
  std::vector<wgpu::BindGroupEntry> bind_group_entries;
  for (const auto& b : buffers) {
    wgpu::BindGroupEntry entry;
    entry.binding = bindings_.at(b.first).binding;
    entry.buffer = b.second.handle;
    entry.size = b.second.size;
    entry.offset = b.second.offset;
    bind_group_entries.push_back(entry);
  }
  for (const auto& t : images2d) {
    wgpu::BindGroupEntry entry;
    entry.binding = bindings_.at(t.first).binding;
    entry.textureView = t.second.handle;
    bind_group_entries.push_back(entry);
  }
  for (const auto& t : image2d_arrays) {
    wgpu::BindGroupEntry entry;
    entry.binding = bindings_.at(t.first).binding;
    entry.textureView = t.second.handle;
    bind_group_entries.push_back(entry);
  }
  for (const auto& t : images3d) {
    wgpu::BindGroupEntry entry;
    entry.binding = bindings_.at(t.first).binding;
    entry.textureView = t.second.handle;
    bind_group_entries.push_back(entry);
  }
  if (scalars.GetMemoryHandle()) {
    wgpu::BindGroupEntry entry;
    entry.binding = bindings_.at("U").binding;
    entry.buffer = scalars.GetMemoryHandle();
    entry.offset = scalars.offset();
    entry.size = scalars.GetMemorySizeInBytes();
    bind_group_entries.push_back(entry);
  }
  wgpu::BindGroupDescriptor bind_descriptor = {
      .layout = group_layout_,
      .entryCount = static_cast<uint32_t>(bind_group_entries.size()),
      .entries = bind_group_entries.data(),
  };
  bind_group_ = device.CreateBindGroup(&bind_descriptor);
}

void WebGpuArguments::CopyFrom(const WebGpuArguments& other) {
  int_values_ = other.int_values_;
  uint_values_ = other.uint_values_;
  float_values_ = other.float_values_;
  buffers_ = other.buffers_;
  images2d_ = other.images2d_;
  image2d_arrays_ = other.image2d_arrays_;
  images3d_ = other.images3d_;
  object_refs_ = other.object_refs_;
  objects_ = other.objects_;
  const_data_ = other.const_data_;
  last_uploaded_const_data_ = other.last_uploaded_const_data_;
  scalars_ =
      Buffer(other.scalars_.GetMemoryHandle(),
             other.scalars_.GetMemorySizeInBytes(), other.scalars_.offset());
  pipeline_layout_ = other.pipeline_layout_;
}

}  // namespace webgpu
}  // namespace ml_drift
