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

#ifndef ML_DRIFT_CL_TESTING_CL_TEST_H_
#define ML_DRIFT_CL_TESTING_CL_TEST_H_

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace cl {

class ClExecutionEnvironment : public TestExecutionEnvironment {
 public:
  ClExecutionEnvironment() = default;
  ~ClExecutionEnvironment() override = default;

  absl::Status Init();

  std::vector<DataType> GetSupportedDataTypes() const override;
  std::vector<TensorStorageType> GetSupportedStorages(
      DataType data_type) const override;

  const GpuInfo& GetGpuInfo() const override;

  absl::Status ExecuteGpuOperationInternal(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation) override;

  absl::StatusOr<ProfilingInfo> GetGpuOperationTimeMs(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWC>& dst_sizes, int num_repeats) override;

  Environment* GetEnvironmentPtr() { return &env_; }

 private:
  Environment env_;
};

class OpenCLOperationTest : public ::testing::Test {
 public:
  void SetUp() override {
    ABSL_ASSERT_OK(LoadOpenCL());
    ABSL_ASSERT_OK(exec_env_.Init());
  }

 protected:
  ClExecutionEnvironment exec_env_;
};

class OpenCLOperationTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    ABSL_ASSERT_OK(LoadOpenCL());
    ABSL_ASSERT_OK(exec_env_.Init());
  }

  ClExecutionEnvironment exec_env_;
};
}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_TESTING_CL_TEST_H_
