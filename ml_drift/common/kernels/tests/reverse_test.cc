// Copyright 2025 The ML Drift Authors.
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/reverse_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class ReverseFloatTest : public DataTypeTest {};

TEST_P(ReverseFloatTest, ReverseHWCTest) {
  if (!exec_env->IsSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReverseHWCTest(*exec_env, data_type(), storage()));
}

TEST_P(ReverseFloatTest, ReverseBHWCTest) {
  if (!exec_env->IsSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReverseBHWCTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ReverseFloatTestSuite, ReverseFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ReverseFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
