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

#ifndef ML_DRIFT_WEBGPU_ENVIRONMENT_H_
#define ML_DRIFT_WEBGPU_ENVIRONMENT_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/ascii.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/compute_pipeline_cache.h"
#include "ml_drift/webgpu/instance.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

struct DeviceUnhandledErrors {
  int total_count = 0;
  int create_compute_pipeline_count = 0;
};

class Environment {
 public:
  Environment();
  explicit Environment(wgpu::BackendType preferred_backend_type);
  virtual ~Environment() = default;

  // Sets a function that can be used to handle fatal errors.
  using ErrorFn = void (*)(const char*);
  static void SetErrorFn(ErrorFn error_fn);
  static void OnError(const std::string& msg);

  absl::Status Initialize(const wgpu::Device& device,
                          const wgpu::AdapterInfo& adapter_info);

#ifndef __EMSCRIPTEN__
  struct InitParams {
    const wgpu::Limits* limits;
    bool use_low_power;
    bool enable_host_mapped_pointer;
    const wgpu::DawnCacheDeviceDescriptor* cache_descriptor;
  };
  absl::Status Initialize(const InitParams& params = {});
#endif

  void RequestExtension(const std::string& extensions);

  void SetUseAsyncCreateCalls(bool use_async_create_calls) {
    use_async_create_calls_ = use_async_create_calls;
  }

  void Tick() const;

  const wgpu::Device& device() const { return device_; }
  const wgpu::Queue& queue() const { return queue_; }
  wgpu::Instance instance() const { return Instance::Get(device_); }
  bool use_async_create_calls() const { return use_async_create_calls_; }
  const GpuInfo& GetInfo() const { return gpu_info_; }

  ComputePipelineCache* GetComputePipelineCache() const {
    return &compute_pipeline_cache_;
  }

  UniformBufferCreator* GetUniformBufferCreator() const {
    return &uniform_buffer_creator_;
  }

  std::string GetPlatformDescription() const { return platform_description_; }

  int GetDeviceUnhandledErrorsTotalCount() const {
    return device_unhandled_errors_.total_count;
  }

  int GetDeviceUnhandledErrorsCreateComputePipelineCount() const {
    return device_unhandled_errors_.create_compute_pipeline_count;
  }

 private:
#ifndef __EMSCRIPTEN__
  absl::Status InitializeInstanceAndAdapter(wgpu::Adapter* adapter,
                                            bool use_low_power);
#endif  // __EMSCRIPTEN__
  void SetPlatformDescription(const std::string& platform_description) {
    platform_description_ = platform_description;
    absl::AsciiStrToLower(&platform_description_);
  }

  static ErrorFn error_fn_;

  wgpu::BackendType preferred_backend_type_;
  wgpu::Device device_;
  // The platform description of the device. This is used to check if the
  // serialized model is compatible with the current platform and is stored in
  // lower case for comparison.
  std::string platform_description_;
  wgpu::Queue queue_;
  mutable ComputePipelineCache compute_pipeline_cache_;
  mutable UniformBufferCreator uniform_buffer_creator_;
  GpuInfo gpu_info_ = {};
  std::vector<std::string> requested_extensions_;
  bool use_async_create_calls_ = false;
  DeviceUnhandledErrors device_unhandled_errors_;
};

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_ENVIRONMENT_H_
