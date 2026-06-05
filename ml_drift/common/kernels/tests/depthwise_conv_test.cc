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
#include "ml_drift/common/kernels/tests/depthwise_conv_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class DepthwiseConvFloatTest : public FloatTest {};

TEST_P(DepthwiseConvFloatTest, DepthwiseConvSimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvSimpleWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvNoMultiplierTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvNoMultiplierTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvMultiplier2Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvMultiplier2Test(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvBigTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvBatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvExternalWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      DepthwiseConvExternalWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConv3DTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConv3DTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConv3DBatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConv3DBatchedTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConv3DNoMultiplierTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConv3DNoMultiplierTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConv3x3SimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      DepthwiseConv3x3SimpleWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConv3x3Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConv3x3Test(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConv3x3BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConv3x3BigTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConv3x3BatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConv3x3BatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvTiledSimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      DepthwiseConvTiledSimpleWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvTiledTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvTiledTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvTiledBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvTiledBigTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvTiledBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      DepthwiseConvTiledBatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvFloatTest, DepthwiseConvTiledWithDilationBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      DepthwiseConvTiledWithDilationBigTest(*exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    DepthwiseConvFloatTestSuite, DepthwiseConvFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<DepthwiseConvFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
