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

#ifndef ML_DRIFT_COMMON_API_CL_GL_VK_H_
#define ML_DRIFT_COMMON_API_CL_GL_VK_H_

// Usage example:
//
//   // Builder is created from a model using GPU-specific parameters.
//   std::unique_ptr<InferenceBuilder> builder = ...;
//
//   // input data is coming from a texture
//   // output data goes to CPU
//   builder->SetInputObjectDef(0, {DataType::FLOAT16, DataLayout::PHWC4,
//                                  ObjectType::OPENGL_TEXTURE, true});
//   builder->SetOutputObjectDef(0, {DataType::FLOAT32, DataLayout::BHWC,
//                                  ObjectType::CPU_MEMORY, false});
//   ABSL_ASSIGN_OR_RETURN(auto runner, builder->Build());  // may be slow
//   ABSL_RETURN_IF_ERROR(
//       runner->SetInputObject(0, OpenGlTexture{texture_id, texture_format}));
//   ABSL_RETURN_IF_ERROR(runner->Run());

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/api_common.h"  // IWYU pragma: export
#include "ml_drift/common/data_type.h"
#include <CL/cl.h>
#include "vulkan/vulkan.h"  // IWYU pragma: keep

#ifndef CL_DELEGATE_NO_GL
#endif

namespace ml_drift {

enum class ObjectType {
  UNKNOWN,
#ifndef CL_DELEGATE_NO_GL
  OPENGL_SSBO,
  OPENGL_TEXTURE,
#endif
  CPU_MEMORY,
  OPENCL_TEXTURE,
  OPENCL_BUFFER,
  VULKAN_BUFFER,
  VULKAN_TEXTURE
};

#ifndef CL_DELEGATE_NO_GL
struct OpenGlBuffer {
  OpenGlBuffer() = default;
  explicit OpenGlBuffer(GLuint new_id) : id(new_id) {}

  GLuint id = GL_INVALID_INDEX;
};

struct OpenGlTexture {
  OpenGlTexture() = default;
  OpenGlTexture(GLuint new_id, GLenum new_format)
      : id(new_id), format(new_format) {}

  GLuint id = GL_INVALID_INDEX;
  GLenum format = GL_INVALID_ENUM;
};
#endif

struct OpenClBuffer {
  OpenClBuffer() = default;
  explicit OpenClBuffer(cl_mem new_memobj) : memobj(new_memobj) {}

  cl_mem memobj = nullptr;
};

struct OpenClTexture {
  OpenClTexture() = default;
  explicit OpenClTexture(cl_mem new_memobj) : memobj(new_memobj) {}

  cl_mem memobj = nullptr;
  // TODO(akulik): should it specify texture format?
};

struct VulkanBuffer {
  VulkanBuffer() = default;
  explicit VulkanBuffer(VkBuffer buffer_, VkDeviceSize size_,
                        VkDeviceMemory memory_, VkDeviceSize offset_)
      : buffer(buffer_), size(size_), memory(memory_), offset(offset_) {}

  VkBuffer buffer;
  VkDeviceSize size;
  VkDeviceMemory memory;
  VkDeviceSize offset;
};

struct VulkanTexture {
  VulkanTexture() = default;
  explicit VulkanTexture(VkDeviceMemory new_memory) : memory(new_memory) {}

  VkImage image;
  VkImageView image_view;
  VkFormat format;
  VkExtent3D extent;
  VkDeviceMemory memory;
  VkDeviceSize offset;
};

struct VulkanMemory {
  VulkanMemory() = default;
  explicit VulkanMemory(VkDeviceMemory new_memory) : memory(new_memory) {}

  VkDeviceMemory memory;
  VkDeviceSize size;
  VkDeviceSize offset;
};

// Defines object representation.
struct ObjectDef {
  DataType data_type = DataType::UNKNOWN;
  DataLayout data_layout = DataLayout::UNKNOWN;
  ObjectType object_type = ObjectType::UNKNOWN;

  // If true, then object is managed externally and needs to be provided to
  // InferenceRunner by a user before running inference.
  //
  // User-provided objects will not be re-used internally for any purpose to
  // lower overall memory usage.
  bool user_provided = false;

  bool operator==(const ObjectDef& other) const {
    return data_type == other.data_type && data_layout == other.data_layout &&
           object_type == other.object_type &&
           user_provided == other.user_provided;
  }
};

bool IsValid(const ObjectDef& def);

// Connects tensor shape with corresponding object definition.
struct TensorObjectDef {
  // Dimensions semantic is defined by corresponding DataLayout.
  Dimensions dimensions;
  ObjectDef object_def;

  bool operator==(const TensorObjectDef& other) const {
    return dimensions == other.dimensions && object_def == other.object_def;
  }
};

// @return true if tensor object def is defined.
bool IsValid(const TensorObjectDef& def);

// @return the number of elements in a tensor object.
uint32_t NumElements(const TensorObjectDef& def);

using TensorObject = std::variant<std::monostate,
#ifndef CL_DELEGATE_NO_GL
                                  OpenGlBuffer, OpenGlTexture,
#endif
                                  CpuMemory, OpenClBuffer, OpenClTexture,
                                  VulkanBuffer, VulkanTexture>;

// @return true if object is set and corresponding values are defined.
bool IsValid(const TensorObjectDef& def, const TensorObject& object);

ObjectType GetType(const TensorObject& object);

// @return true if corresponding object is set for the given type
bool IsObjectPresent(ObjectType type, const TensorObject& obj);

// @return true if corresponding object has already been initialized and
// assigned with a specific ObjectType.
bool IsObjectInitialized(const TensorObject& obj);

class InferenceRunner;

// Allows to inspect and change input and output definitions before a graph is
// prepared for the inference.
class InferenceBuilder {
 public:
  virtual ~InferenceBuilder() = default;

  // Returns inference graph inputs and outputs definitions.
  virtual std::vector<TensorObjectDef> inputs() const = 0;
  virtual std::vector<TensorObjectDef> outputs() const = 0;

  // Sets new shape for the input if underlying implementation and graph
  // structure allows dynamic tensors.
  virtual absl::Status SetInputShape(int index,
                                     const Dimensions& dimensions) = 0;

  // Updates object definitions for the given index. Implementation may allow
  // to use different layouts and/or data type conversions between objects
  // defined in a graph and given objects, for example:
  //   input '0' is DataType::FLOAT32, DataLayout::BHWC.
  //   A user, however, has an input in DataType::FLOAT16, DataLayout::PHWC4.
  //   An implementation may allow this transformation to happen automatically
  //   under the hood.
  virtual absl::Status SetInputObjectDef(int index, ObjectDef def) = 0;
  virtual absl::Status SetOutputObjectDef(int index, ObjectDef def) = 0;
  virtual absl::Status SetAllInputObjectDefsTo(ObjectDef def) {
    auto input_defs = inputs();
    for (int i = 0; i < input_defs.size(); ++i) {
      ABSL_RETURN_IF_ERROR(SetInputObjectDef(i, def));
    }
    return absl::OkStatus();
  }
  virtual absl::Status SetAllOutputObjectDefsTo(ObjectDef def) {
    auto output_defs = outputs();
    for (int i = 0; i < output_defs.size(); ++i) {
      ABSL_RETURN_IF_ERROR(SetOutputObjectDef(i, def));
    }
    return absl::OkStatus();
  }

  // Creates new instance of the inference runner. InferenceBuilder stays valid
  // and could be used to create another inference runner if needed.
  //
  // This method may take significant time to prepare new inference runner. For
  // example, it may require to compile OpenGL shaders.
  virtual absl::StatusOr<std::unique_ptr<InferenceRunner>> Build() = 0;
};

// Runs prepared inference. Every object marked as external needs to be set
// prior calling Run method.
class InferenceRunner {
 public:
  virtual ~InferenceRunner() = default;

  // Returns inference graph inputs and outputs definitions.
  virtual std::vector<TensorObjectDef> inputs() const = 0;
  virtual std::vector<TensorObjectDef> outputs() const = 0;

  // Getters provide access to underlying objects for the given index.
  // Setters allow to set or change external object for the given index. Note,
  // object need to match object definition set before in InferenceBuilder.

  virtual absl::Status GetInputObject(int index, TensorObject* object) = 0;
  virtual absl::Status GetOutputObject(int index, TensorObject* object) = 0;
  virtual absl::Status SetInputObject(int index, TensorObject object) = 0;
  virtual absl::Status SetOutputObject(int index, TensorObject object) = 0;

  virtual absl::Status Run() = 0;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_API_CL_GL_VK_H_
