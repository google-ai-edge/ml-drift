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



#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/cl/testing/cl_test.h"
#include "ml_drift/common/gpu_model_external_weights_test_util.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, DynamicConvolution) {
  auto status = TestDynamicConvolution(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ExternalConvWeights) {
  auto status = TestExternalConvWeights(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, FullyConnectedSmallSpatial) {
  auto status = TestFullyConnected(&exec_env_, BHWC(1, 1, 1, 16));
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, FullyConnectedBigSpatial) {
  auto status = TestFullyConnected(&exec_env_, BHWC(1, 1, 256, 16));
  ASSERT_TRUE(status.ok()) << status.message();
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
