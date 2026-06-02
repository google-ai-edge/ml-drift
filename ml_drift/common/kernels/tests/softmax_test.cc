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
#include "ml_drift/common/kernels/tests/softmax_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class SoftmaxFloatTest : public DataTypeTest {};

TEST_P(SoftmaxFloatTest, SoftmaxTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxWGTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxWGTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxReduceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxReduceTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxBigNumberTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxBigNumberTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxRuntimeChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxRuntimeChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxReduceBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxReduceBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxReduceRuntimeChannelsBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      SoftmaxReduceRuntimeChannelsBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, SoftmaxRuntimeChannelsBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(SoftmaxRuntimeChannelsBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1Test(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1BigNumberTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1BigNumberTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1RuntimeChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1RuntimeChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1Custom1Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1Custom1Test(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1BigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1BigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1BatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1BatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1ReduceBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1ReduceBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1ReduceRuntimeChannelsBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x1ReduceRuntimeChannelsBigTest(*exec_env, data_type(),
                                                   storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x1RuntimeChannelsBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      Softmax1x1RuntimeChannelsBigTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax5DTest(*exec_env, data_type(), storage()));
}

TEST_P(SoftmaxFloatTest, Softmax1x15DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Softmax1x15DTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    SoftmaxFloatTestSuite, SoftmaxFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<SoftmaxFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
