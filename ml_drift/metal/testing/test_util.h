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

#ifndef ML_DRIFT_METAL_TESTING_TEST_UTIL_H_
#define ML_DRIFT_METAL_TESTING_TEST_UTIL_H_

#import <Metal/Metal.h>

#include <memory>
#include <vector>

#include "testing/base/public/gunit.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/metal/metal_device.h"

namespace ml_drift {
namespace metal {

class MetalExecutionEnvironment : public TestExecutionEnvironment {
 public:
  MetalExecutionEnvironment() = default;
  ~MetalExecutionEnvironment() = default;

  std::vector<DataType> GetSupportedDataTypes() const override;
  std::vector<TensorStorageType> GetSupportedStorages(
      DataType data_type) const override;

  const GpuInfo& GetGpuInfo() const override { return device_.GetInfo(); }

  absl::Status ExecuteGpuOperationInternal(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation) override;

 private:
  MetalDevice device_;
};

// Note that GTM_GoogleTestRunner_GTM_USING_XCTEST (used for iOS tests) doesn't
// run the main function. As such, AddGlobalTestEnvironment() will not be
// called. This matters in case anyone ever adds global SetUp/TearDown methods
// to the test environment.
class MetalOperationTestEnvironment : public ::testing::Environment {
 public:
  MetalExecutionEnvironment exec_env_;
};

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_TESTING_TEST_UTIL_H_
