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

#ifndef ML_DRIFT_GL_GPU_OBJECT_H_
#define ML_DRIFT_GL_GPU_OBJECT_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {

struct GPUResourcesWithValue {
  ml_drift::GenericGPUResourcesWithValue generic;

  std::vector<std::pair<std::string, GLuint>> buffers;
  std::vector<std::pair<std::string, GLuint>> images2d;
  std::vector<std::pair<std::string, GLuint>> image2d_arrays;
  std::vector<std::pair<std::string, GLuint>> images3d;
  std::vector<std::pair<std::string, GLuint>> image_buffers;

  void AddFloat(const std::string& name, float value) {
    generic.AddFloat(name, value);
  }
  void AddInt(const std::string& name, int value) {
    generic.AddInt(name, value);
  }
};

class GPUObject {
 public:
  GPUObject() = default;
  // Move only
  GPUObject(GPUObject&& obj_desc) = default;
  GPUObject& operator=(GPUObject&& obj_desc) = default;
  GPUObject(const GPUObject&) = delete;
  GPUObject& operator=(const GPUObject&) = delete;
  virtual ~GPUObject() = default;
  virtual absl::Status GetGPUResources(
      const ml_drift::GPUObjectDescriptor* obj_ptr,
      GPUResourcesWithValue* resources) const = 0;
};

using GPUObjectPtr = std::unique_ptr<GPUObject>;

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GPU_OBJECT_H_
