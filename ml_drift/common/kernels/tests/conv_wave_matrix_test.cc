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

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/conv_wave_matrix_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class ConvWaveMatrixFloatTest : public FloatTest {};

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrix1x1Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMatrix1x1Test(*exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMatrixTest(*exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrix1x1BatchTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMatrix1x1BatchTest(*exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrix1x1ExternalWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status =
      ConvWaveMatrix1x1ExternalWeightsTest(*exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixExternalWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status =
      ConvWaveMatrixExternalWeightsTest(*exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixExternalBatchedWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMatrixExternalBatchedWeightsTest(*exec_env, precision(),
                                                         storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixExternalBatchedWi4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status =
      ConvWaveMatrixExternalBatchedWi4Test(*exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixExternalBatchedGroupedWi4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMatrixExternalBatchedGroupedWi4Test(
      *exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixWinograd4x4To6x6Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status =
      ConvWaveMatrixWinograd4x4To6x6Test(*exec_env, precision(), storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixRuntimeSrcEndChannelsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMatrixRuntimeSrcEndChannelsTest(*exec_env, precision(),
                                                        storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(ConvWaveMatrixFloatTest, ConvWaveMatrixRuntimeDstEndChannelsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMatrixRuntimeDstEndChannelsTest(*exec_env, precision(),
                                                        storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

INSTANTIATE_TEST_SUITE_P(
    ConvWaveMatrixFloatTestSuite, ConvWaveMatrixFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<ConvWaveMatrixFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class Int8Test : public testing::Test,
                 public testing::WithParamInterface<
                     std::tuple<TensorStorageType, TensorStorageType>> {
 protected:
  TensorStorageType src_storage() const { return std::get<0>(GetParam()); }
  TensorStorageType dst_storage() const { return std::get<1>(GetParam()); }
};

TEST_P(Int8Test, ConvWaveMatrixInt8Test) {
  // Check dst storage. Check src storage w/i helper.
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported dst storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  } else if (!exec_env->IsStorageSupported(src_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported src storage type: " << ToString(src_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  auto status = ConvWaveMatrixInt8Test(*exec_env, src_storage(), dst_storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
}

TEST_P(Int8Test, ConvWaveMatrixInt8ExternalWeightsTest) {
  // Check dst storage. Check src storage w/i helper.
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported dst storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  } else if (!exec_env->IsStorageSupported(src_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported src storage type: " << ToString(src_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  auto status = ConvWaveMatrixInt8ExternalWeightsTest(*exec_env, src_storage(),
                                                      dst_storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

TEST_P(Int8Test, ConvWaveMatrixInt8WithSrcQuantizationTest) {
  // Check dst storage. Check src storage w/i helper.
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported dst storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  } else if (!exec_env->IsStorageSupported(src_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported src storage type: " << ToString(src_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  auto status = ConvWaveMatrixInt8WithSrcQuantizationTest(
      *exec_env, src_storage(), dst_storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  MLD_ASSERT_OK(status);
}

INSTANTIATE_TEST_SUITE_P(
    Int8TestSuite, Int8Test,
    Combine(ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<Int8Test::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param))),
          {{":", "_"}});
    });

}  // namespace ml_drift
