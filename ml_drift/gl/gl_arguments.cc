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

#include "ml_drift/gl/gl_arguments.h"

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_spatial_tensor.h"
#include "ml_drift/gl/gl_texture_helper.h"
#include "ml_drift/gl/gpu_object.h"
#include "ml_drift/gl/portable_gl31.h"  // IWYU pragma: keep

namespace ml_drift {
namespace gl {
namespace {

inline bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

absl::string_view GetNextWord(absl::string_view code, size_t first_position) {
  size_t pos = first_position;
  while (pos < code.size() && IsWordSymbol(code[pos])) {
    pos++;
  }
  return code.substr(first_position, pos - first_position);
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

absl::Status CreateGlObject(ml_drift::GPUObjectDescriptor* desc,
                            GPUObjectPtr* result) {
  const auto* buffer_desc =
      dynamic_cast<const ml_drift::BufferDescriptor*>(desc);
  if (buffer_desc) {
    GlBuffer gpu_buffer;
    gpu_buffer.CreateFromBufferDescriptor(*buffer_desc);
    *result = std::make_unique<GlBuffer>(std::move(gpu_buffer));
    return absl::OkStatus();
  }

  const auto* tensor_spatial_desc =
      dynamic_cast<const ml_drift::TensorDescriptor*>(desc);
  if (tensor_spatial_desc) {
    GlSpatialTensor gpu_tensor;
    ABSL_RETURN_IF_ERROR(gpu_tensor.CreateFromDescriptor(*tensor_spatial_desc));
    *result = std::make_unique<GlSpatialTensor>(std::move(gpu_tensor));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unknown GPU descriptor.");
}

std::string DataTypeToGLType(ml_drift::DataType data_type, int vec_size) {
  if (data_type == ml_drift::DataType::kFloat32) {
    if (vec_size == 1) {
      return "float";
    } else if (vec_size == 2) {
      return "vec2";
    } else if (vec_size == 4) {
      return "vec4";
    } else if (vec_size == 8) {
      return "mat4x2";
    } else if (vec_size == 16) {
      return "mat4x4";
    }
  } else if (data_type == ml_drift::DataType::kFloat16) {
    if (vec_size == 2) {
      return "uint";
    } else if (vec_size == 4) {
      return "uvec2";
    } else if (vec_size == 8) {
      return "uvec4";
    } else if (vec_size == 16) {
      return "uvec8";
    }
  } else if (data_type == ml_drift::DataType::kInt32) {
    if (vec_size == 1) {
      return "int";
    } else if (vec_size == 2) {
      return "ivec2";
    } else if (vec_size == 4) {
      return "ivec4";
    }
  } else if (data_type == ml_drift::DataType::kInt16) {
    if (vec_size == 1) {
      return "no_type";
    } else if (vec_size == 2) {
      return "int";
    } else if (vec_size == 4) {
      return "ivec2";
    }
  } else if (data_type == ml_drift::DataType::kInt8) {
    if (vec_size == 1) {
      return "no_type";
    } else if (vec_size == 2) {
      return "no_type";
    } else if (vec_size == 4) {
      return "int";
    }
  } else if (data_type == ml_drift::DataType::kUint32) {
    if (vec_size == 1) {
      return "uint";
    } else if (vec_size == 2) {
      return "uvec2";
    } else if (vec_size == 4) {
      return "uvec4";
    }
  } else if (data_type == ml_drift::DataType::kUint16 ||
             data_type == ml_drift::DataType::kBfloat16) {
    if (vec_size == 1) {
      return "no_type";
    } else if (vec_size == 2) {
      return "uint";
    } else if (vec_size == 4) {
      return "uvec2";
    }
  } else if (data_type == ml_drift::DataType::kUint8) {
    if (vec_size == 1) {
      return "no_type";
    } else if (vec_size == 2) {
      return "no_type";
    } else if (vec_size == 4) {
      return "uint";
    }
  } else if (data_type == ml_drift::DataType::kBool) {
    if (vec_size == 1) {
      return "no_type";
    } else if (vec_size == 2) {
      return "no_type";
    } else if (vec_size == 4) {
      return "uint";
    }
  }
  return "";
}

std::string GetTypePrefix(ml_drift::DataType data_type) {
  if (data_type == ml_drift::DataType::kInt32 ||
      data_type == ml_drift::DataType::kInt16 ||
      data_type == ml_drift::DataType::kInt8) {
    return "i";
  } else if (data_type == ml_drift::DataType::kUint32 ||
             data_type == ml_drift::DataType::kBfloat16 ||
             data_type == ml_drift::DataType::kUint16 ||
             data_type == ml_drift::DataType::kUint8 ||
             data_type == ml_drift::DataType::kBool) {
    return "u";
  } else {
    return "";
  }
}

std::string GetPrecisionModifier(ml_drift::DataType data_type) {
  if (data_type == ml_drift::DataType::kInt32 ||
      data_type == ml_drift::DataType::kUint32 ||
      data_type == ml_drift::DataType::kFloat32) {
    return "highp";
  } else if (data_type == ml_drift::DataType::kBfloat16 ||
             data_type == ml_drift::DataType::kInt16 ||
             data_type == ml_drift::DataType::kUint16 ||
             data_type == ml_drift::DataType::kFloat16) {
    return "mediump";
  } else if (data_type == ml_drift::DataType::kInt8 ||
             data_type == ml_drift::DataType::kUint8 ||
             data_type == ml_drift::DataType::kBool) {
    return "lowp";
  } else {
    return "highp";
  }
}

std::string GetShaderImageType(ml_drift::DataType data_type) {
  switch (data_type) {
    case ml_drift::DataType::kFloat32:
      return "rgba32f";
    case ml_drift::DataType::kFloat16:
      return "rgba16f";
    case ml_drift::DataType::kInt32:
      return "rgba32i";
    case ml_drift::DataType::kInt16:
      return "rgba16i";
    case ml_drift::DataType::kInt8:
      return "rgba8i";
    case ml_drift::DataType::kUint32:
      return "rgba32ui";
    case ml_drift::DataType::kBfloat16:
    case ml_drift::DataType::kUint16:
      return "rgba16ui";
    case ml_drift::DataType::kUint8:
      return "rgba8ui";
    case ml_drift::DataType::kBool:
      return "rgba8ui";
    default:
      return "unknown";
  }
}

absl::Status MoveLocalMemToGlobalScope(std::string* code) {
  std::string local_mem_declarations;
  size_t loc_mem_start = code->find("__local");
  while (loc_mem_start != std::string::npos) {
    size_t loc_mem_end = code->find(';', loc_mem_start);
    if (loc_mem_end == std::string::npos) {
      return absl::InternalError("Wrong local memory declaration.");
    }
    loc_mem_end += 1;
    local_mem_declarations +=
        "shared" +
        code->substr(loc_mem_start + 7, loc_mem_end - loc_mem_start - 7) + "\n";
    code->erase(loc_mem_start, loc_mem_end - loc_mem_start);
    loc_mem_start = code->find("__local", loc_mem_start);
  }
  if (!local_mem_declarations.empty()) {
    size_t main_function_pos = code->find("MAIN_FUNCTION($0)");
    if (main_function_pos == std::string::npos) {
      return absl::InternalError("Can not find MAIN_FUNCTION.");
    }
    code->insert(main_function_pos, local_mem_declarations);
  }
  return absl::OkStatus();
}
}  // namespace

void GlArguments::AddFloat(const std::string& name, float value) {
  float_values_[name].value = value;
}
void GlArguments::AddInt(const std::string& name, int value) {
  int_values_[name].value = value;
}
void GlArguments::AddUint(const std::string& name, uint value) {
  uint_values_[name].value = value;
}
void GlArguments::AddBuffer(const std::string& name,
                            const ml_drift::GPUBufferDescriptor& desc) {
  buffers_[name].desc = desc;
}
void GlArguments::AddTexture2D(const std::string& name,
                               const ml_drift::GPUImage2DDescriptor& desc) {
  textures2d_[name].desc = desc;
}
void GlArguments::AddTexture2DArray(
    const std::string& name, const ml_drift::GPUImage2DArrayDescriptor& desc) {
  texture2d_arrays_[name].desc = desc;
}

void GlArguments::AddImage3D(const std::string& name,
                             const ml_drift::GPUImage3DDescriptor& desc) {
  images3d_[name].desc = desc;
}
void GlArguments::AddImageBuffer(
    const std::string& name, const ml_drift::GPUImageBufferDescriptor& desc) {
  image_buffers_[name].desc = desc;
}

void GlArguments::AddGPUResources(const std::string& name,
                                  const ml_drift::GPUResources& resources) {
  for (const auto& r : resources.buffers) {
    AddBuffer(name + "_" + r.first, r.second);
  }
  for (const auto& r : resources.images2d) {
    AddTexture2D(name + "_" + r.first, r.second);
  }
  for (const auto& r : resources.image2d_arrays) {
    AddTexture2DArray(name + "_" + r.first, r.second);
  }
  for (const auto& r : resources.images3d) {
    AddImage3D(name + "_" + r.first, r.second);
  }
  for (const auto& r : resources.image_buffers) {
    AddImageBuffer(name + "_" + r.first, r.second);
  }
}

absl::Status GlArguments::SetFloat(const std::string& name, float value) {
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(absl::StrCat("No float with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_float4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status GlArguments::SetInt(const std::string& name, int value) {
  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(absl::StrCat("No int with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_int4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status GlArguments::SetUint(const std::string& name, uint value) {
  auto it = uint_values_.find(name);
  if (it == uint_values_.end()) {
    return absl::NotFoundError(absl::StrCat("No uint with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_uint4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status GlArguments::SetHalf(const std::string& name,
                                  ml_drift::half value) {
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(absl::StrCat("No half with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_float4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status GlArguments::SetBuffer(const std::string& name, GLuint id) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(absl::StrCat("No buffer with name - ", name));
  }
  it->second.id = id;
  return absl::OkStatus();
}

absl::Status GlArguments::SetTexture2D(const std::string& name, GLuint id) {
  auto it = textures2d_.find(name);
  if (it == textures2d_.end()) {
    return absl::NotFoundError(absl::StrCat("No texture2d with name - ", name));
  }
  it->second.id = id;
  return absl::OkStatus();
}

absl::Status GlArguments::SetTexture2DArray(const std::string& name,
                                            GLuint id) {
  auto it = texture2d_arrays_.find(name);
  if (it == texture2d_arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No texture2d array with name - ", name));
  }
  it->second.id = id;
  return absl::OkStatus();
}

absl::Status GlArguments::SetImage3D(const std::string& name, GLuint id) {
  auto it = images3d_.find(name);
  if (it == images3d_.end()) {
    return absl::NotFoundError(absl::StrCat("No image3d with name - ", name));
  }
  it->second.id = id;
  return absl::OkStatus();
}
absl::Status GlArguments::SetImageBuffer(const std::string& name, GLuint id) {
  auto it = image_buffers_.find(name);
  if (it == image_buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image_buffer with name - ", name));
  }
  it->second.id = id;
  return absl::OkStatus();
}

absl::Status GlArguments::SetObjectRef(const std::string& name,
                                       const GPUObjectPtr& object) {
  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  GPUResourcesWithValue resources;
  ABSL_RETURN_IF_ERROR(object->GetGPUResources(it->second.get(), &resources));
  return SetGPUResources(name, resources);
}

absl::Status GlArguments::SetObjectRef(const std::string& name,
                                       const GPUObject* object) {
  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  GPUResourcesWithValue resources;
  ABSL_RETURN_IF_ERROR(object->GetGPUResources(it->second.get(), &resources));
  return SetGPUResources(name, resources);
}

absl::Status GlArguments::SetGPUResources(
    const std::string& name, const GPUResourcesWithValue& resources) {
  for (const auto& r : resources.generic.ints) {
    ABSL_RETURN_IF_ERROR(SetInt(name + "_" + r.first, r.second));
  }
  for (const auto& r : resources.generic.uints) {
    ABSL_RETURN_IF_ERROR(SetUint(name + "_" + r.first, r.second));
  }
  for (const auto& r : resources.generic.floats) {
    ABSL_RETURN_IF_ERROR(SetFloat(name + "_" + r.first, r.second));
  }
  for (const auto& r : resources.buffers) {
    ABSL_RETURN_IF_ERROR(SetBuffer(name + "_" + r.first, r.second));
  }
  for (const auto& r : resources.images2d) {
    ABSL_RETURN_IF_ERROR(SetTexture2D(name + "_" + r.first, r.second));
  }
  for (const auto& r : resources.image2d_arrays) {
    ABSL_RETURN_IF_ERROR(SetTexture2DArray(name + "_" + r.first, r.second));
  }
  for (const auto& r : resources.images3d) {
    ABSL_RETURN_IF_ERROR(SetImage3D(name + "_" + r.first, r.second));
  }
  for (const auto& r : resources.image_buffers) {
    ABSL_RETURN_IF_ERROR(SetImageBuffer(name + "_" + r.first, r.second));
  }
  return absl::OkStatus();
}

std::string GlArguments::ToGLSLUniforms() {
  std::string result;
  int image_slot = 0;
  for (auto& t : textures2d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kWrite) {
      result +=
          "layout(" + GetShaderImageType(t.second.desc.data_type) +
          ", binding = " + std::to_string(image_slot) + ") writeonly uniform " +
          GetPrecisionModifier(t.second.desc.data_type) + " " +
          GetTypePrefix(t.second.desc.data_type) + "image2D " + t.first + ";\n";
      image_slots_[t.first] = image_slot++;
    }
  }
  for (auto& t : texture2d_arrays_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kWrite) {
      result += "layout(" + GetShaderImageType(t.second.desc.data_type) +
                ", binding = " + std::to_string(image_slot) +
                ") writeonly uniform " +
                GetPrecisionModifier(t.second.desc.data_type) + " " +
                GetTypePrefix(t.second.desc.data_type) + "image2DArray " +
                t.first + ";\n";
      image_slots_[t.first] = image_slot++;
    }
  }
  for (auto& t : images3d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kWrite) {
      result +=
          "layout(" + GetShaderImageType(t.second.desc.data_type) +
          ", binding = " + std::to_string(image_slot) + ") writeonly uniform " +
          GetPrecisionModifier(t.second.desc.data_type) + " " +
          GetTypePrefix(t.second.desc.data_type) + "image3D " + t.first + ";\n";
      image_slots_[t.first] = image_slot++;
    }
  }
  int ssbo_slot = 0;
  for (auto& b : buffers_) {
    if (b.second.desc.memory_type == ml_drift::MemoryType::kConstant) {
      result += "layout(std140) uniform " + b.first + "_ubo {\n";
      if (b.second.desc.data_type == ml_drift::DataType::kFloat32) {
        result +=
            "  vec4 " + b.first + "[" + b.second.desc.attributes[0] + "];\n";
      } else if (b.second.desc.data_type == ml_drift::DataType::kFloat16) {
        const int elements_count = std::stoi(b.second.desc.attributes[0]);
        result += "  uvec4 " + b.first + "[" +
                  std::to_string(ml_drift::DivideRoundUp(elements_count, 2)) +
                  "];\n";
      }
      result += "};\n";
    } else {
      const std::string gl_type =
          DataTypeToGLType(b.second.desc.data_type, b.second.desc.element_size);
      result += "layout(std430, binding = " + std::to_string(ssbo_slot) +
                ") buffer " + b.first + "_ssbo {\n";
      result += "  " + gl_type + " " + b.first + "[];\n";
      result += "};\n";
      ssbo_slots_[b.first] = ssbo_slot++;
    }
  }
  for (auto& t : textures2d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      result += "uniform " + GetPrecisionModifier(t.second.desc.data_type) +
                " " + GetTypePrefix(t.second.desc.data_type) + "sampler2D " +
                t.first + ";\n";
    }
  }
  for (auto& t : texture2d_arrays_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      result += "uniform " + GetPrecisionModifier(t.second.desc.data_type) +
                " " + GetTypePrefix(t.second.desc.data_type) +
                "sampler2DArray " + t.first + ";\n";
    }
  }
  for (auto& t : images3d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      result += "uniform " + GetPrecisionModifier(t.second.desc.data_type) +
                " " + GetTypePrefix(t.second.desc.data_type) + "sampler3D " +
                t.first + ";\n";
    }
  }
  for (auto& t : image_buffers_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      result += "uniform " + GetPrecisionModifier(t.second.desc.data_type) +
                " " + GetTypePrefix(t.second.desc.data_type) +
                "samplerBuffer " + t.first + ";\n";
    }
  }

  if (!shared_int4s_data_.empty()) {
    result += "uniform highp ivec4 shared_int4s[" +
              std::to_string(shared_int4s_data_.size() / 4) + "];\n";
  }
  if (!shared_uint4s_data_.empty()) {
    result += "uniform highp uvec4 shared_uint4s[" +
              std::to_string(shared_uint4s_data_.size() / 4) + "];\n";
  }
  if (!shared_float4s_data_.empty()) {
    result += "uniform highp vec4 shared_float4s[" +
              std::to_string(shared_float4s_data_.size() / 4) + "];\n";
  }
  return result;
}

void GlArguments::GetLocations(GLuint pipeline) {
  int ubo_slot = 0;
  for (auto& b : buffers_) {
    if (b.second.desc.memory_type == ml_drift::MemoryType::kConstant) {
      GLuint loc = glGetUniformBlockIndex(pipeline, (b.first + "_ubo").c_str());
      glUniformBlockBinding(pipeline, loc, ubo_slot);
      ubo_slots_[b.first] = ubo_slot++;
    }
  }
  int slot = 0;
  for (auto& t : textures2d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      GLuint loc = glGetUniformLocation(pipeline, t.first.c_str());
      glUniform1i(loc, slot);
      texture_slots_[t.first] = slot++;
    }
  }
  for (auto& t : texture2d_arrays_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      GLuint loc = glGetUniformLocation(pipeline, t.first.c_str());
      glUniform1i(loc, slot);
      texture_slots_[t.first] = slot++;
    }
  }
  for (auto& t : images3d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      GLuint loc = glGetUniformLocation(pipeline, t.first.c_str());
      glUniform1i(loc, slot);
      texture_slots_[t.first] = slot++;
    }
  }
  for (auto& t : image_buffers_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      GLuint loc = glGetUniformLocation(pipeline, t.first.c_str());
      glUniform1i(loc, slot);
      texture_slots_[t.first] = slot++;
    }
  }
  if (!shared_int4s_data_.empty()) {
    shared_int4s_loc_ = glGetUniformLocation(pipeline, "shared_int4s");
  }
  if (!shared_uint4s_data_.empty()) {
    shared_uint4s_loc_ = glGetUniformLocation(pipeline, "shared_uint4s");
  }
  if (!shared_float4s_data_.empty()) {
    shared_float4s_loc_ = glGetUniformLocation(pipeline, "shared_float4s");
  }
}

void GlArguments::Bind() {
  for (auto& b : buffers_) {
    if (b.second.desc.memory_type == ml_drift::MemoryType::kConstant) {
      glBindBufferBase(GL_UNIFORM_BUFFER, ubo_slots_[b.first], b.second.id);
    } else {
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_slots_[b.first],
                       b.second.id);
    }
  }
  for (auto& t : textures2d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      glActiveTexture(GL_TEXTURE0 + texture_slots_[t.first]);
      glBindTexture(GL_TEXTURE_2D, t.second.id);
    } else if (t.second.desc.access_type == ml_drift::AccessType::kWrite) {
      GLenum format =
          ml_drift::gl::ToTextureInternalFormat(t.second.desc.data_type);
      glBindImageTexture(image_slots_[t.first], t.second.id, 0, GL_FALSE, 0,
                         GL_WRITE_ONLY, format);
    }
  }
  for (auto& t : texture2d_arrays_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      glActiveTexture(GL_TEXTURE0 + texture_slots_[t.first]);
      glBindTexture(GL_TEXTURE_2D_ARRAY, t.second.id);
    } else if (t.second.desc.access_type == ml_drift::AccessType::kWrite) {
      GLenum format =
          ml_drift::gl::ToTextureInternalFormat(t.second.desc.data_type);
      glBindImageTexture(image_slots_[t.first], t.second.id, 0, GL_TRUE, 0,
                         GL_WRITE_ONLY, format);
    }
  }
  for (auto& t : images3d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      glActiveTexture(GL_TEXTURE0 + texture_slots_[t.first]);
      glBindTexture(GL_TEXTURE_3D, t.second.id);
    } else if (t.second.desc.access_type == ml_drift::AccessType::kWrite) {
      GLenum format =
          ml_drift::gl::ToTextureInternalFormat(t.second.desc.data_type);
      glBindImageTexture(image_slots_[t.first], t.second.id, 0, GL_TRUE, 0,
                         GL_WRITE_ONLY, format);
    }
  }
  for (auto& t : image_buffers_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      glActiveTexture(GL_TEXTURE0 + texture_slots_[t.first]);
      glBindTexture(GL_TEXTURE_BUFFER, t.second.id);
    }
  }
  for (auto& t : images3d_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      glActiveTexture(GL_TEXTURE0 + texture_slots_[t.first]);
      glBindTexture(GL_TEXTURE_3D, t.second.id);
    }
  }
  for (auto& t : image_buffers_) {
    if (t.second.desc.access_type == ml_drift::AccessType::kRead) {
      glActiveTexture(GL_TEXTURE0 + texture_slots_[t.first]);
      glBindTexture(GL_TEXTURE_BUFFER, t.second.id);
    }
  }
  if (!shared_float4s_data_.empty()) {
    glUniform4fv(shared_float4s_loc_, shared_float4s_data_.size() / 4,
                 shared_float4s_data_.data());
  }
  if (!shared_int4s_data_.empty()) {
    glUniform4iv(shared_int4s_loc_, shared_int4s_data_.size() / 4,
                 shared_int4s_data_.data());
  }
  if (!shared_uint4s_data_.empty()) {
    glUniform4uiv(shared_uint4s_loc_, shared_uint4s_data_.size() / 4,
                  shared_uint4s_data_.data());
  }
}

void GlArguments::RenameArgs(const std::string& postfix,
                             std::string* code) const {
  size_t position = 0;
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position + std::strlen(kArgsPrefix);
    absl::string_view arg_name = GetNextWord(*code, arg_pos);
    size_t length = arg_name.size();
    code->replace(arg_pos, length, absl::StrCat(arg_name, postfix));
    position = arg_pos + length;
    next_position = code->find(kArgsPrefix, position);
  }
}

void GlArguments::CopyArguments(const ml_drift::Arguments& args) {
  for (const auto& fvalue : args.GetFloatValues()) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.offset = shared_float4s_data_.size();
      shared_float4s_data_.push_back(new_val.value);
    }
  }
  for (const auto& fvalue : args.GetHalfValues()) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.offset = shared_float4s_data_.size();
      shared_float4s_data_.push_back(new_val.value);
    }
  }
  for (const auto& ivalue : args.GetIntValues()) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.offset = shared_int4s_data_.size();
      shared_int4s_data_.push_back(new_val.value);
    }
  }
  for (const auto& uivalue : args.GetUintValues()) {
    auto& new_val = uint_values_[uivalue.first];
    new_val.value = uivalue.second.value;
    new_val.active = uivalue.second.active;
    if (uivalue.second.active) {
      new_val.offset = shared_uint4s_data_.size();
      shared_uint4s_data_.push_back(new_val.value);
    }
  }
  int shared_int4s_aligned_size =
      ml_drift::AlignByN(shared_int4s_data_.size(), 4);
  shared_int4s_data_.resize(shared_int4s_aligned_size);
  int shared_uint4s_aligned_size =
      ml_drift::AlignByN(shared_uint4s_data_.size(), 4);
  shared_uint4s_data_.resize(shared_uint4s_aligned_size);
  int shared_float4s_aligned_size =
      ml_drift::AlignByN(shared_float4s_data_.size(), 4);
  shared_float4s_data_.resize(shared_float4s_aligned_size);
}

void GlArguments::RenameArgumentsInCode(std::string* code) const {
  const std::string postfixes[4] = {"x", "y", "z", "w"};
  for (const auto& fvalue : float_values_) {
    if (fvalue.second.active) {
      std::string index = std::to_string(fvalue.second.offset / 4);
      std::string new_name = "shared_float4s[" + index + "]." +
                             postfixes[fvalue.second.offset % 4];
      ReplaceAllWords(kArgsPrefix + fvalue.first, new_name, code);
    }
  }
  for (const auto& ivalue : int_values_) {
    if (ivalue.second.active) {
      std::string index = std::to_string(ivalue.second.offset / 4);
      std::string new_name =
          "shared_int4s[" + index + "]." + postfixes[ivalue.second.offset % 4];
      ReplaceAllWords(kArgsPrefix + ivalue.first, new_name, code);
    }
  }
  for (const auto& uivalue : uint_values_) {
    if (uivalue.second.active) {
      std::string index = std::to_string(uivalue.second.offset / 4);
      std::string new_name = "shared_uint4s[" + index + "]." +
                             postfixes[uivalue.second.offset % 4];
      ReplaceAllWords(kArgsPrefix + uivalue.first, new_name, code);
    }
  }
}

absl::Status GlArguments::AddObjectArgs(const ml_drift::GpuInfo& gpu_info,
                                        const ml_drift::Arguments& args) {
  for (const auto& t : args.GetObjects()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  for (const auto& t : args.GetObjectRefs()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  return absl::OkStatus();
}

absl::Status GlArguments::AllocateObjects(const ml_drift::Arguments& args) {
  objects_.resize(args.GetObjects().size());
  int i = 0;
  for (const auto& t : args.GetObjects()) {
    ABSL_RETURN_IF_ERROR(CreateGlObject(t.second.get(), &objects_[i]));
    i++;
  }
  return absl::OkStatus();
}

absl::Status GlArguments::SetObjectsResources(const ml_drift::Arguments& args) {
  int i = 0;
  for (const auto& t : args.GetObjects()) {
    GPUResourcesWithValue resources;
    ABSL_RETURN_IF_ERROR(
        objects_[i]->GetGPUResources(t.second.get(), &resources));
    ABSL_RETURN_IF_ERROR(SetGPUResources(t.first, resources));
    i++;
  }
  return absl::OkStatus();
}

absl::Status GlArguments::Init(const ml_drift::GpuInfo& gpu_info,
                               ml_drift::Arguments* args, std::string* code) {
  ABSL_RETURN_IF_ERROR(AllocateObjects(*args));
  ABSL_RETURN_IF_ERROR(AddObjectArgs(gpu_info, *args));
  args->MoveObjectRefs(&object_refs_);
  CopyArguments(*args);
  ABSL_RETURN_IF_ERROR(SetObjectsResources(*args));
  RenameArgumentsInCode(code);
  args->ResolveArgsPass(code);
  *code = ToGLSLUniforms() + *code;
  // GLSL-specific.
  ABSL_RETURN_IF_ERROR(MoveLocalMemToGlobalScope(code));
  return absl::OkStatus();
}

absl::Status GlArguments::Init(const ml_drift::GpuInfo& gpu_info,
                               ml_drift::Arguments* args) {
  ABSL_RETURN_IF_ERROR(AllocateObjects(*args));
  ABSL_RETURN_IF_ERROR(AddObjectArgs(gpu_info, *args));
  args->MoveObjectRefs(&object_refs_);
  CopyArguments(*args);
  ABSL_RETURN_IF_ERROR(SetObjectsResources(*args));
  ToGLSLUniforms();
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace ml_drift
