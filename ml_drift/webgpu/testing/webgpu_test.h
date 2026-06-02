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

#ifndef ML_DRIFT_WEBGPU_TESTING_WEBGPU_TEST_H_
#define ML_DRIFT_WEBGPU_TESTING_WEBGPU_TEST_H_

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/webgpu/execution_environment.h"
#include "ml_drift/webgpu/webgpu_headers.h"

#ifdef __EMSCRIPTEN__
#include "third_party/mediapipe/gpu/webgpu/webgpu_service.h"
#endif  // __EMSCRIPTEN__

namespace ml_drift {
namespace webgpu {

class WebGpuExecutionEnvironment : public TestExecutionEnvironment {
 public:
#ifdef __EMSCRIPTEN__
  WebGpuExecutionEnvironment() : env_(wgpu::BackendType::WebGPU) {}
  ~WebGpuExecutionEnvironment() override = default;
  absl::Status Init() {
    auto service_or_status = mediapipe::WebGpuService::Create();
    auto service = *service_or_status;
    return env_.Initialize(service->device(), service->adapter_info());
  }
#elif defined __APPLE__
  WebGpuExecutionEnvironment() : env_(wgpu::BackendType::Metal) {}
  ~WebGpuExecutionEnvironment() override = default;
  absl::Status Init() { return env_.Initialize(); }
#else
  WebGpuExecutionEnvironment() : env_(wgpu::BackendType::Vulkan) {}
  ~WebGpuExecutionEnvironment() override = default;
  absl::Status Init() { return env_.Initialize(); }
#endif  // __EMSCRIPTEN__

  std::vector<DataType> GetSupportedDataTypes() const override {
    return {DataType::FLOAT32, DataType::FLOAT16};
  }
  std::vector<TensorStorageType> GetSupportedStorages(
      DataType data_type) const override {
    return {TensorStorageType::BUFFER, TensorStorageType::TEXTURE_2D,
            TensorStorageType::TEXTURE_ARRAY, TensorStorageType::TEXTURE_3D};
  }

  const GpuInfo& GetGpuInfo() const override { return env_.GetInfo(); }
  const ExecutionEnvironment& GetEnv() const { return env_; }

  absl::Status ExecuteGpuOperationInternal(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation) override;

 private:
  ExecutionEnvironment env_;
};

class WebGpuOperationTest : public ::testing::Test {
 public:
  void SetUp() override { MLD_ASSERT_OK(exec_env_.Init()); }

 protected:
  WebGpuExecutionEnvironment exec_env_;
};

class WebGpuOperationTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { MLD_ASSERT_OK(exec_env_.Init()); }
  WebGpuExecutionEnvironment exec_env_;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_TESTING_WEBGPU_TEST_H_
