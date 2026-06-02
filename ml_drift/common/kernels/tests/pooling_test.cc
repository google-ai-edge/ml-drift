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
#include "ml_drift/common/kernels/tests/pooling_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class PoolingFloatTest : public DataTypeTest {};

TEST_P(PoolingFloatTest, AveragePoolingTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(AveragePoolingTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, AveragePoolingNonEmptyPaddingTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      AveragePoolingNonEmptyPaddingTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPoolingTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingIndicesTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPoolingIndicesTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, AveragePoolingBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(AveragePoolingBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, AveragePoolingBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(AveragePoolingBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, AveragePoolingNonEmptyPaddingBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      AveragePoolingNonEmptyPaddingBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, AveragePooling3DBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(AveragePooling3DBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, AveragePooling3DBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(AveragePooling3DBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPoolingBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPoolingBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPooling3DBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPooling3DBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPooling3DBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPooling3DBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingIndicesBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPoolingIndicesBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingIndicesBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPoolingIndicesBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingIndices3DBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MaxPoolingIndices3DBigTest(*exec_env, data_type(), storage()));
}

TEST_P(PoolingFloatTest, MaxPoolingIndices3DBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      MaxPoolingIndices3DBatchedBigTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    PoolingFloatTestSuite, PoolingFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<PoolingFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
