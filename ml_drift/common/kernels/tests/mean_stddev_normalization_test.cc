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

#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/mean_stddev_normalization_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class MeanStddevGroupTest
    : public Test,
      public WithParamInterface<std::tuple<DataType, TensorStorageType, int>> {
};

TEST_P(MeanStddevGroupTest, HWCGroupNormalizationTest) {
  auto [data_type, storage, group_size] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage);
  }
  MLD_ASSERT_OK(
      HWCGroupNormalizationTest(*exec_env, data_type, storage, group_size));
}

TEST_P(MeanStddevGroupTest, HWCGroupNormalizationBatchTest) {
  auto [data_type, storage, group_size] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage);
  }
  MLD_ASSERT_OK(HWCGroupNormalizationBatchTest(*exec_env, data_type, storage,
                                           group_size));
}

INSTANTIATE_TEST_SUITE_P(
    MeanStddevGroupTestSuite, MeanStddevGroupTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes()),
            ValuesIn({1, 2, 4, 8, 12})),
    [](const TestParamInfo<MeanStddevGroupTest::ParamType>& info) {
      // Test names can only contain alphanumeric
      // characters and underscores. ToString(storage_type)
      // = "TensorStorageType::x" We get rid of the colons
      // with StrReplaceAll.
      const std::string name =
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_",
                       std::to_string(std::get<2>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

class MeanStddevFloatTest : public DataTypeTest {};

TEST_P(MeanStddevFloatTest, MeanStddevNormSeparateBatchesTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      MeanStddevNormSeparateBatchesTest(*exec_env, data_type(), storage()));
}

TEST_P(MeanStddevFloatTest, MeanStddevNormalizationAllBatchesTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      MeanStddevNormalizationAllBatchesTest(*exec_env, data_type(), storage()));
}

TEST_P(MeanStddevFloatTest, MeanStddevNormalizationLargeVectorTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MeanStddevNormalizationLargeVectorTest(*exec_env, data_type(),
                                                   storage()));
}

TEST_P(MeanStddevFloatTest, RMSNormalizationBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(RMSNormalizationBigTest(*exec_env, data_type(), storage()));
}

TEST_P(MeanStddevFloatTest, StatisticalTopKTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(StatisticalTopKTest(*exec_env, data_type(), storage()));
}

TEST_P(MeanStddevFloatTest, MeanStddevNormalization5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MeanStddevNormalization5DTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    MeanStddevFloatTestSuite, MeanStddevFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<MeanStddevFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
