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

#include "ml_drift/common/api_cl_gl_vk.h"

#include <cstdint>
#include <variant>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/util.h"

#ifndef CL_DELEGATE_NO_GL
#define GL_NO_PROTOTYPES
#define EGL_NO_PROTOTYPES
#include "ml_drift/pelong/portable_gl31.h"
#undef GL_NO_PROTOTYPES
#undef EGL_NO_PROTOTYPES
#endif

namespace ml_drift {
namespace {

struct ObjectTypeGetter {
  ObjectType operator()(std::monostate) const { return ObjectType::UNKNOWN; }
#ifndef CL_DELEGATE_NO_GL
  ObjectType operator()(OpenGlBuffer) const { return ObjectType::OPENGL_SSBO; }
  ObjectType operator()(OpenGlTexture) const {
    return ObjectType::OPENGL_TEXTURE;
  }
#endif
  ObjectType operator()(OpenClBuffer) const {
    return ObjectType::OPENCL_BUFFER;
  }
  ObjectType operator()(OpenClTexture) const {
    return ObjectType::OPENCL_TEXTURE;
  }
  ObjectType operator()(VulkanBuffer) const {
    return ObjectType::VULKAN_BUFFER;
  }
  ObjectType operator()(VulkanTexture) const {
    return ObjectType::VULKAN_TEXTURE;
  }
  ObjectType operator()(CpuMemory) const { return ObjectType::CPU_MEMORY; }
};

struct ObjectValidityChecker {
  bool operator()(std::monostate) const { return false; }
#ifndef CL_DELEGATE_NO_GL
  bool operator()(OpenGlBuffer obj) const { return obj.id != GL_INVALID_INDEX; }
  bool operator()(OpenGlTexture obj) const {
    return obj.id != GL_INVALID_INDEX && obj.format != GL_INVALID_ENUM;
  }
#endif
  bool operator()(OpenClBuffer obj) const { return obj.memobj; }
  bool operator()(OpenClTexture obj) const { return obj.memobj; }
  bool operator()(VulkanBuffer obj) const { return obj.memory; }
  bool operator()(VulkanTexture obj) const { return obj.memory; }
  bool operator()(CpuMemory obj) const {
    return obj.data != nullptr && obj.size_bytes > 0 &&
           (data_type == DataType::UNKNOWN || data_type == DataType::BOOL ||
            obj.size_bytes % SizeOf(data_type) == 0);
  }
  DataType data_type;
};

}  // namespace

bool IsValid(const ObjectDef& def) {
  return def.data_type != DataType::UNKNOWN &&
         def.data_layout != DataLayout::UNKNOWN &&
         def.object_type != ObjectType::UNKNOWN;
}

ObjectType GetType(const TensorObject& object) {
  return std::visit(ObjectTypeGetter{}, object);
}

bool IsValid(const TensorObjectDef& def) { return IsValid(def.object_def); }

bool IsValid(const TensorObjectDef& def, const TensorObject& object) {
  return GetType(object) == def.object_def.object_type &&
         std::visit(ObjectValidityChecker{def.object_def.data_type}, object);
}

bool IsObjectPresent(ObjectType type, const TensorObject& obj) {
  switch (type) {
    case ObjectType::CPU_MEMORY:
      return std::holds_alternative<CpuMemory>(obj);
#ifndef CL_DELEGATE_NO_GL
    case ObjectType::OPENGL_SSBO:
      return std::holds_alternative<OpenGlBuffer>(obj);
    case ObjectType::OPENGL_TEXTURE:
      return std::holds_alternative<OpenGlTexture>(obj);
#endif
    case ObjectType::OPENCL_BUFFER:
      return std::holds_alternative<OpenClBuffer>(obj);
    case ObjectType::OPENCL_TEXTURE:
      return std::holds_alternative<OpenClTexture>(obj);
    case ObjectType::VULKAN_BUFFER:
      return std::holds_alternative<VulkanBuffer>(obj);
    case ObjectType::VULKAN_TEXTURE:
      return std::holds_alternative<VulkanTexture>(obj);
    case ObjectType::UNKNOWN:
      return false;
  }
}

bool IsObjectInitialized(const TensorObject& obj) {
  return GetType(obj) != ObjectType::UNKNOWN;
}

uint32_t NumElements(const TensorObjectDef& def) {
  const auto& d = def.dimensions;
  switch (def.object_def.data_layout) {
    case DataLayout::BHWC:
      return d.product();
    case DataLayout::HWDC4:
    case DataLayout::HDWC4:
    case DataLayout::DHWC4:
      return d.b * d.h * d.w * AlignByN(d.c, 4);
    case DataLayout::UNKNOWN:
      return 0;
  }
  return 0;
}

}  // namespace ml_drift
