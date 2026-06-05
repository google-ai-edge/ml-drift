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
#include "ml_drift/common/kernels/tests/conv_transpose_thin_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class ConvTransposeThinFloatTest : public FloatTest {};

TEST_P(ConvTransposeThinFloatTest, ConvolutionTransposedThinSimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvolutionTransposedThinSimpleWeightsTest(*exec_env, precision(),
                                                       storage()));
}

TEST_P(ConvTransposeThinFloatTest, ConvolutionTransposedThinTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvolutionTransposedThinTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeThinFloatTest, ConvolutionTransposedThinBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      ConvolutionTransposedThinBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeThinFloatTest, ConvolutionTransposedThinBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvolutionTransposedThinBatchedBigTest(*exec_env, precision(),
                                                    storage()));
}

TEST_P(ConvTransposeThinFloatTest,
       ConvolutionTransposed3x3ThinSimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvolutionTransposed3x3ThinSimpleWeightsTest(
      *exec_env, precision(), storage()));
}

TEST_P(ConvTransposeThinFloatTest, ConvolutionTransposed3x3ThinTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      ConvolutionTransposed3x3ThinTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeThinFloatTest, ConvolutionTransposed3x3ThinBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      ConvolutionTransposed3x3ThinBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvTransposeThinFloatTest, ConvolutionTransposed3x3ThinBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvolutionTransposed3x3ThinBatchedBigTest(*exec_env, precision(),
                                                       storage()));
}

TEST_P(ConvTransposeThinFloatTest,
       ConvolutionTransposed3x3ThinExternalWeightsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvolutionTransposed3x3ThinExternalWeightsBigTest(
      *exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvTransposeThinFloatTestSuite, ConvTransposeThinFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ConvTransposeThinFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
