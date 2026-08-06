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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/rope_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class RopeFloatTest : public DataTypeTest {};

TEST_P(RopeFloatTest, RoPETest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(RoPETest(*exec_env, data_type(), storage()));
}

TEST_P(RopeFloatTest, SplitRoPEConcatTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SplitRoPEConcatTest(*exec_env, data_type(), storage()));
}

TEST_P(RopeFloatTest, SplitRoPEConcatIntPositionTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SplitRoPEConcatIntPositionTest(*exec_env, data_type(), storage()));
}

TEST_P(RopeFloatTest, SplitRoPEConcatIntChannelPositionTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      SplitRoPEConcatIntChannelPositionTest(*exec_env, data_type(), storage()));
}

TEST_P(RopeFloatTest, SplitRoPEConcatIntSinglePositionTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      SplitRoPEConcatIntSinglePositionTest(*exec_env, data_type(), storage()));
}

TEST_P(RopeFloatTest, SplitRoPEConcatInterleavedAxialTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      SplitRoPEConcatInterleavedAxialTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    RopeFloatTestSuite, RopeFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<RopeFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
