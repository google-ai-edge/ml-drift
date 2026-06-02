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

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/conv_constants_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class ConvConstantsFloatTest : public FloatTest {};

TEST_P(ConvConstantsFloatTest, SimpleWeights) {
  const auto data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvConstantsSimpleWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(ConvConstantsFloatTest, ConvConstantsTest) {
  const auto data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvConstantsTest(*exec_env, precision(), storage()));
}

TEST_P(ConvConstantsFloatTest, ConvConstantsBatchedBigTest) {
  const auto data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvConstantsBatchedBigTest(*exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvConstantsFloatTestSuite, ConvConstantsFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ConvConstantsFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class ConvConstantsChannelTest
    : public Test,
      public WithParamInterface<
          std::tuple<CalculationsPrecision, TensorStorageType, int, int>> {
  // Convenience accessor methods
 protected:
  CalculationsPrecision precision() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
  int src_channel() const { return std::get<2>(GetParam()); }
  int dst_channel() const { return std::get<3>(GetParam()); }
};

TEST_P(ConvConstantsChannelTest, ConvConstantsBigTest) {
  const auto data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  if (precision() != CalculationsPrecision::F32 &&
      (src_channel() != 5 || dst_channel() != 7)) {
    GTEST_SKIP() << "Only test non F32 for channels = (5, 7)";
  }
  MLD_ASSERT_OK(ConvConstantsBigTest(*exec_env, precision(), storage(),
                                 src_channel(), dst_channel()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvConstantsChannelTestSuite, ConvConstantsChannelTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes()),
            ValuesIn({1, 2, 3, 4, 5, 6, 7, 8}),
            ValuesIn({1, 2, 3, 4, 5, 6, 7, 8})),
    [](const TestParamInfo<ConvConstantsChannelTest::ParamType>& info) {
      FloatTest::ParamType param;
      param = std::make_tuple(std::get<0>(info.param), std::get<1>(info.param));
      return absl::StrCat(ToString(param), "_src_ch_", std::get<2>(info.param),
                          "_dst_ch_", std::get<3>(info.param));
    });

class ConvConstantsExternalWeightsTest
    : public Test,
      public WithParamInterface<
          std::tuple<CalculationsPrecision, TensorStorageType>> {
  // Convenience accessor methods
 protected:
  CalculationsPrecision precision() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
};

TEST_P(ConvConstantsExternalWeightsTest, BigTest) {
  const auto data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      ConvConstantsExternalWeightsBigTest(*exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvConstantsExternalWeightsTestSuite, ConvConstantsExternalWeightsTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ConvConstantsExternalWeightsTest::ParamType>& info) {
      FloatTest::ParamType param;
      param = std::make_tuple(std::get<0>(info.param), std::get<1>(info.param));
      return absl::StrCat(ToString(param));
    });

}  // namespace ml_drift
