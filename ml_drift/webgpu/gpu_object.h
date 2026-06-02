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

#ifndef ML_DRIFT_WEBGPU_GPU_OBJECT_H_
#define ML_DRIFT_WEBGPU_GPU_OBJECT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

struct BufferResource {
  wgpu::Buffer buffer;
  uint64_t size;
  uint64_t offset;
};

struct GpuResourcesWithValue {
  GenericGPUResourcesWithValue generic;

  std::vector<std::pair<std::string, BufferResource>> buffers;
  std::vector<std::pair<std::string, wgpu::TextureView>> images2d;
  std::vector<std::pair<std::string, wgpu::TextureView>> images3d;
  std::vector<std::pair<std::string, wgpu::TextureView>> image2d_arrays;

  void AddFloat(const std::string& name, float value) {
    generic.AddFloat(name, value);
  }
  void AddInt(const std::string& name, int value) {
    generic.AddInt(name, value);
  }
};

class GpuObject {
 public:
  GpuObject() = default;
  // Move only
  GpuObject(GpuObject&& obj_desc) = default;
  GpuObject& operator=(GpuObject&& obj_desc) = default;
  GpuObject(const GpuObject&) = delete;
  GpuObject& operator=(const GpuObject&) = delete;
  virtual ~GpuObject() = default;
  virtual absl::Status GetGPUResources(
      const GPUObjectDescriptor* obj_ptr,
      GpuResourcesWithValue* resources) const = 0;
};

using GpuObjectPtr = std::unique_ptr<GpuObject>;

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_GPU_OBJECT_H_
