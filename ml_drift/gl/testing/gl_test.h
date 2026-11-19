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

#ifndef ML_DRIFT_GL_TESTING_GL_TEST_H_
#define ML_DRIFT_GL_TESTING_GL_TEST_H_

#include <dlfcn.h>

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/gl/egl_environment.h"
#include "ml_drift/gl/portable_gl31.h"
#include "ml_drift/gl/request_gpu_info.h"

#ifndef ABSL_ASSERT_OK
#define ABSL_ASSERT_OK(x) ASSERT_TRUE(x.ok());
#endif

namespace ml_drift {
namespace gl {

class GlExecutionEnvironment : public ml_drift::TestExecutionEnvironment {
 public:
  GlExecutionEnvironment() = default;
  ~GlExecutionEnvironment() override {
    if (fb_ != -1) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &fb_);
      fb_ = -1;
    }
  }

  absl::Status Init() {
    ABSL_RETURN_IF_ERROR(
        ml_drift::gl::EglEnvironment::NewEglEnvironment(&egl_env_));

    glGenFramebuffers(1, &fb_);
    glBindFramebuffer(GL_FRAMEBUFFER, fb_);
    ABSL_RETURN_IF_ERROR(ml_drift::gl::RequestGpuInfo(&gpu_info_));
    return absl::OkStatus();
  }

  std::vector<ml_drift::DataType> GetSupportedDataTypes() const override {
    return {ml_drift::DataType::kFloat32, ml_drift::DataType::kFloat16};
  }

  std::vector<ml_drift::TensorStorageType> GetSupportedStorages(
      ml_drift::DataType data_type) const override {
    std::vector<ml_drift::TensorStorageType> storages = {
        ml_drift::TensorStorageType::kTexture2D,
        ml_drift::TensorStorageType::kBuffer,
        ml_drift::TensorStorageType::kTextureArray,
        ml_drift::TensorStorageType::kTexture3D};
    if (gpu_info_.opengl_info.IsApiOpenGl32OrAbove()) {
      storages.push_back(ml_drift::TensorStorageType::kImageBuffer);
    }
    return storages;
  }

  const ml_drift::GpuInfo& GetGpuInfo() const override { return gpu_info_; }

  absl::Status ExecuteGpuOperationInternal(
      const std::vector<ml_drift::TensorDescriptor*>& src_cpu,
      const std::vector<ml_drift::TensorDescriptor*>& dst_cpu,
      std::unique_ptr<ml_drift::GPUOperation>&& operation) override;

 private:
  GLuint fb_ = -1;
  ml_drift::GpuInfo gpu_info_;
  std::unique_ptr<ml_drift::gl::EglEnvironment> egl_env_;
};

class OpenGlOperationTest : public ::testing::Test {
 public:
  void SetUp() override { ABSL_ASSERT_OK(exec_env_.Init()); }
  void TearDown() override {}
  static void SetUpTestSuite() {
#ifndef __ANDROID__
    // Keeping dbus loaded allows leak checking ignore memory allocated by it.
    (void)dlopen("libdbus-1.so.3", RTLD_NOW | RTLD_GLOBAL);
#endif  // !__ANDROID__
  }

 protected:
  GlExecutionEnvironment exec_env_;
};

class OpenGLOperationTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { ABSL_ASSERT_OK(exec_env_.Init()); }
  void TearDown() override {}
  static void SetUpTestSuite() {
#ifndef __ANDROID__
    // Keeping dbus loaded allows leak checking ignore memory allocated by it.
    (void)dlopen("libdbus-1.so.3", RTLD_NOW | RTLD_GLOBAL);
#endif  // !__ANDROID__
  }
  GlExecutionEnvironment exec_env_;
};

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_TESTING_GL_TEST_H_
