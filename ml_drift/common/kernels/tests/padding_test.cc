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
#include "ml_drift/common/kernels/tests/padding_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class PaddingFloatTest : public DataTypeTest {};

TEST_P(PaddingFloatTest, PaddingAppendWidthTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingAppendWidthTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingAppendWidthConstValuesTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      PaddingAppendWidthConstValuesTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingPrependWidthTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingPrependWidthTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingAppendHeightTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingAppendHeightTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingPrependHeightTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingPrependHeightTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingAppendChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingAppendChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingAppendChannelsUnalignedSrcTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      PaddingAppendChannelsUnalignedSrcTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingPrependChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingPrependChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingPrependChannelsX4Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingPrependChannelsX4Test(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingComplexTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingComplexTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingReflectWidthTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingReflectWidthTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingReflectChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingReflectChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PaddingFloatTest, PaddingBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PaddingBatchedBigTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    PaddingFloatTestSuite, PaddingFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<PaddingFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
