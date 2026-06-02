// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_COMMON_TASK_GPU_OBJECT_DESC_H_
#define ML_DRIFT_COMMON_TASK_GPU_OBJECT_DESC_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/serialization_base_generated.h"

namespace ml_drift {

struct GPUImage2DDescriptor {
  DataType data_type;
  bool normalized = false;   // used with INT data types, if normalized, we read
                             // in kernel float data.
  DataType normalized_type;  // can be FLOAT32 or FLOAT16, using with normalized
                             // = true
  AccessType access_type;
};

struct GPUImage3DDescriptor {
  DataType data_type;
  AccessType access_type;
};

struct GPUImage2DArrayDescriptor {
  DataType data_type;
  AccessType access_type;
};

struct GPUImageBufferDescriptor {
  DataType data_type;
  AccessType access_type;
};

struct GPUCustomMemoryDescriptor {
  std::string type_name;
};

enum class MemoryType { GLOBAL, CONSTANT, LOCAL };

struct GPUBufferDescriptor {
  DataType data_type;
  AccessType access_type;
  int element_size;
  MemoryType memory_type = MemoryType::GLOBAL;
  std::vector<std::string> attributes;
};

struct GPUResources {
  std::vector<std::string> ints;
  std::vector<std::string> floats;
  std::vector<std::pair<std::string, GPUBufferDescriptor>> buffers;
  std::vector<std::pair<std::string, GPUImage2DDescriptor>> images2d;
  std::vector<std::pair<std::string, GPUImage2DArrayDescriptor>> image2d_arrays;
  std::vector<std::pair<std::string, GPUImage3DDescriptor>> images3d;
  std::vector<std::pair<std::string, GPUImageBufferDescriptor>> image_buffers;
  std::vector<std::pair<std::string, GPUCustomMemoryDescriptor>>
      custom_memories;

  std::vector<std::string> GetNames() const {
    std::vector<std::string> names = ints;
    names.insert(names.end(), floats.begin(), floats.end());
    for (const auto& obj : buffers) {
      names.push_back(obj.first);
    }
    for (const auto& obj : images2d) {
      names.push_back(obj.first);
    }
    for (const auto& obj : image2d_arrays) {
      names.push_back(obj.first);
    }
    for (const auto& obj : images3d) {
      names.push_back(obj.first);
    }
    for (const auto& obj : image_buffers) {
      names.push_back(obj.first);
    }
    for (const auto& obj : custom_memories) {
      names.push_back(obj.first);
    }
    return names;
  }

  int GetReadImagesCount() const {
    int counter = 0;
    for (const auto& t : images2d) {
      if (t.second.access_type == AccessType::READ) {
        counter++;
      }
    }
    for (const auto& t : image2d_arrays) {
      if (t.second.access_type == AccessType::READ) {
        counter++;
      }
    }
    for (const auto& t : images3d) {
      if (t.second.access_type == AccessType::READ) {
        counter++;
      }
    }
    for (const auto& t : image_buffers) {
      if (t.second.access_type == AccessType::READ) {
        counter++;
      }
    }
    return counter;
  }

  int GetWriteImagesCount() const {
    int counter = 0;
    for (const auto& t : images2d) {
      if (t.second.access_type == AccessType::WRITE) {
        counter++;
      }
    }
    for (const auto& t : image2d_arrays) {
      if (t.second.access_type == AccessType::WRITE) {
        counter++;
      }
    }
    for (const auto& t : images3d) {
      if (t.second.access_type == AccessType::WRITE) {
        counter++;
      }
    }
    for (const auto& t : image_buffers) {
      if (t.second.access_type == AccessType::WRITE) {
        counter++;
      }
    }
    return counter;
  }
};

struct GenericGPUResourcesWithValue {
  std::vector<std::pair<std::string, int>> ints;
  std::vector<std::pair<std::string, unsigned int>> uints;
  std::vector<std::pair<std::string, float>> floats;

  void AddFloat(const std::string& name, float value) {
    floats.push_back({name, value});
  }
  void AddInt(const std::string& name, int value) {
    ints.push_back({name, value});
  }
  void AddUint(const std::string& name, unsigned int value) {
    uints.push_back({name, value});
  }
};

class GPUObjectDescriptor {
 public:
  GPUObjectDescriptor() = default;
  GPUObjectDescriptor(const GPUObjectDescriptor&) = default;
  GPUObjectDescriptor& operator=(const GPUObjectDescriptor&) = default;
  GPUObjectDescriptor(GPUObjectDescriptor&& obj_desc) = default;
  GPUObjectDescriptor& operator=(GPUObjectDescriptor&& obj_desc) = default;
  virtual ~GPUObjectDescriptor() = default;

  virtual bool IsTensorDescriptor() const { return false; }
  virtual bool IsBufferDescriptor() const { return false; }

  void SetStateVar(const std::string& key, const std::string& value) const {
    state_vars_[key] = value;
  }

  virtual absl::Status PerformConstExpr(const GpuInfo& gpu_info,
                                        absl::string_view const_expr,
                                        std::string* result) const {
    return absl::UnimplementedError(
        "No implementation of perform const expression");
  }

  virtual absl::Status PerformSelector(
      const GpuInfo& gpu_info, absl::string_view selector,
      const std::vector<std::string>& args,
      const std::vector<std::string>& template_args,
      std::string* result) const {
    return absl::UnimplementedError("No implementation of perform selector");
  }
  virtual GPUResources GetGPUResources(const GpuInfo& gpu_info) const {
    return GPUResources();
  }

  // For internal use, will work correct only for const objects and before
  // cpu representation release.
  virtual uint64_t GetSizeInBytes() const { return 0; }

  void SetAccess(AccessType access_type) { access_type_ = access_type; }
  AccessType GetAccess() const { return access_type_; }

 protected:
  friend flatbuffers::Offset<data::GPUObjectDescriptor> Encode(
      const GPUObjectDescriptor& desc, flatbuffers::FlatBufferBuilder* builder);
  friend void Decode(const data::GPUObjectDescriptor* fb_obj,
                     GPUObjectDescriptor* obj);
  mutable std::map<std::string, std::string, std::less<>> state_vars_;
  AccessType access_type_;
};

using GPUObjectDescriptorPtr = std::unique_ptr<GPUObjectDescriptor>;

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_GPU_OBJECT_DESC_H_
