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
#include "ml_drift/common/kernels/tests/conv_transpose_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class ConvTransposeFloatTest : public FloatTest {};

TEST_P(ConvTransposeFloatTest, ConvTransposedSimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvTransposedSimpleWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvTransposedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvTransposedTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvTransposedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvTransposedBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvTransposedBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvTransposedBatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvTransposed3DBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvTransposed3DBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvTransposed3DBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvTransposed3DBatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvTransposedExternalWeightsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ConvTransposedExternalWeightsBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed2x2Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed2x2Test(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed2x2BatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ConvolutionTransposed2x2BatchedTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed2x2DynamicWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed2x2DynamicWeightsTest(*exec_env, precision(),
                                                       storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed3x3SimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed3x3SimpleWeightsTest(*exec_env, precision(),
                                                      storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed3x3Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed3x3Test(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed3x3BatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ConvolutionTransposed3x3BatchedTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed3x3ExternalWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed3x3ExternalWeightsTest(*exec_env, precision(),
                                                        storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed4x4SimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed4x4SimpleWeightsTest(*exec_env, precision(),
                                                      storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed4x4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed4x4Test(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed4x4BatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ConvolutionTransposed4x4BatchedTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeFloatTest, ConvolutionTransposed4x4ExternalWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvolutionTransposed4x4ExternalWeightsTest(*exec_env, precision(),
                                                        storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvTransposeFloatTestSuite, ConvTransposeFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ConvTransposeFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
