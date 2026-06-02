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

#ifndef ML_DRIFT_CL_CL_TEST_H_
#define ML_DRIFT_CL_CL_TEST_H_

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/opencl_wrapper.h"

namespace ml_drift {
namespace cl {

#ifndef MLD_ASSERT_OK
#define MLD_ASSERT_OK(x) ASSERT_TRUE(x.ok());
#endif

class OpenCLTest : public ::testing::Test {
 public:
  void SetUp() override {
    MLD_ASSERT_OK(LoadOpenCL());
    MLD_ASSERT_OK(CreateEnvironment(&env_));
  }

 protected:
  Environment env_;
};

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_TEST_H_
