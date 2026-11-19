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
#include "ml_drift/common/kernels/tests/reshape_test_util.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class ReshapeFloatTest : public DataTypeTest {};

TEST_P(ReshapeFloatTest, ReshapeTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ReshapeTest(*exec_env, data_type(), storage()));
}

TEST_P(ReshapeFloatTest, ReshapeBigTest0) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ReshapeBigTest(*exec_env, data_type(), storage(), BHWC(1, 3, 2, 7),
                           BHWC(1, 7, 1, 6)));
}

TEST_P(ReshapeFloatTest, ReshapeBigTest1) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ReshapeBigTest(*exec_env, data_type(), storage(), BHWC(1, 3, 2, 7),
                           BHWC(1, 1, 1, 42)));
}

TEST_P(ReshapeFloatTest, ReshapeBigTest2) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ReshapeBigTest(*exec_env, data_type(), storage(), BHWC(1, 3, 2, 7),
                           BHWC(1, 42, 1, 1)));
}

TEST_P(ReshapeFloatTest, ReshapeBatchedBigTest0) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ReshapeBigTest(*exec_env, data_type(), storage(), BHWC(4, 3, 2, 7),
                           BHWC(2, 7, 1, 12)));
}

TEST_P(ReshapeFloatTest, ReshapeBatchedBigTest1) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ReshapeBigTest(*exec_env, data_type(), storage(), BHWC(3, 3, 2, 7),
                           BHWC(1, 1, 3, 42)));
}

TEST_P(ReshapeFloatTest, ReshapeBatchedBigTest2) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ReshapeBigTest(*exec_env, data_type(), storage(), BHWC(1, 3, 2, 7),
                           BHWC(2, 21, 1, 1)));
}

TEST_P(ReshapeFloatTest, Reshape3DBigTest0) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshape3DBigTest(*exec_env, data_type(), storage(),
                             BHWDC(1, 3, 2, 2, 7), BHWDC(1, 7, 1, 2, 6)));
}

TEST_P(ReshapeFloatTest, Reshape3DBigTest1) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshape3DBigTest(*exec_env, data_type(), storage(),
                             BHWDC(1, 3, 2, 2, 7), BHWDC(1, 1, 1, 1, 84)));
}

TEST_P(ReshapeFloatTest, Reshape3DBigTest2) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshape3DBigTest(*exec_env, data_type(), storage(),
                             BHWDC(4, 3, 2, 2, 7), BHWDC(2, 7, 1, 2, 12)));
}

TEST_P(ReshapeFloatTest, Reshapex4Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshapex4Test(*exec_env, data_type(), storage()));
}

TEST_P(ReshapeFloatTest, Reshapex4BigTest0) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshapex4Test(*exec_env, data_type(), storage(), BHWC(1, 5, 4, 12),
                          BHWC(1, 12, 1, 20)));
}

TEST_P(ReshapeFloatTest, Reshapex4BigTest1) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshapex4Test(*exec_env, data_type(), storage(), BHWC(1, 3, 2, 12),
                          BHWC(1, 6, 3, 4)));
}

TEST_P(ReshapeFloatTest, Reshapex4BigTest2) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshapex4Test(*exec_env, data_type(), storage(), BHWC(1, 3, 16, 12),
                          BHWC(1, 12, 3, 16)));
}

TEST_P(ReshapeFloatTest, Reshapex4BatchedBigTest0) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshapex4Test(*exec_env, data_type(), storage(), BHWC(3, 5, 4, 12),
                          BHWC(1, 12, 3, 20)));
}

TEST_P(ReshapeFloatTest, Reshapex4BatchedBigTest1) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshapex4Test(*exec_env, data_type(), storage(), BHWC(1, 3, 2, 12),
                          BHWC(3, 2, 3, 4)));
}

TEST_P(ReshapeFloatTest, Reshapex4BatchedBigTest2) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Reshapex4Test(*exec_env, data_type(), storage(), BHWC(8, 3, 16, 12),
                          BHWC(2, 12, 12, 16)));
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeFloatTestSuite, ReshapeFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ReshapeFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
