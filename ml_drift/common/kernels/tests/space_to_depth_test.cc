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
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/space_to_depth_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class SpaceToDepthFloatTest : public DataTypeTest {};

TEST_P(SpaceToDepthFloatTest, SpaceToDepthTensorShape1x2x2x1BlockSize2Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SpaceToDepthTensorShape1x2x2x1BlockSize2Test(*exec_env, data_type(),
                                                         storage()));
}

TEST_P(SpaceToDepthFloatTest, SpaceToDepthTensorShape1x2x2x2BlockSize2Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SpaceToDepthTensorShape1x2x2x2BlockSize2Test(*exec_env, data_type(),
                                                         storage()));
}

TEST_P(SpaceToDepthFloatTest, SpaceToDepthTensorShape1x2x2x3BlockSize2Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SpaceToDepthTensorShape1x2x2x3BlockSize2Test(*exec_env, data_type(),
                                                         storage()));
}

TEST_P(SpaceToDepthFloatTest, SpaceToDepthTensorShape1x4x4x1BlockSize2Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SpaceToDepthTensorShape1x4x4x1BlockSize2Test(*exec_env, data_type(),
                                                         storage()));
}

TEST_P(SpaceToDepthFloatTest, DepthToSpaceFrom1x1x1x4To1x2x2x1Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      DepthToSpaceFrom1x1x1x4To1x2x2x1Test(*exec_env, data_type(), storage()));
}

TEST_P(SpaceToDepthFloatTest, DepthToSpaceFrom1x1x1x16To1x2x2x4Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      DepthToSpaceFrom1x1x1x16To1x2x2x4Test(*exec_env, data_type(), storage()));
}

TEST_P(SpaceToDepthFloatTest, DepthToSpaceFrom1x2x2x16To1x4x4x4Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      DepthToSpaceFrom1x2x2x16To1x4x4x4Test(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    SpaceToDepthFloatTestSuite, SpaceToDepthFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<SpaceToDepthFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
