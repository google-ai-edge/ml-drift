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

#include "ml_drift/common/kernels/conv_generic.h"

#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/conv_generic_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class ConvGenericFloatTest : public FloatTest {};

TEST_P(ConvGenericFloatTest, ConvGeneric1x1SimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGeneric1x1SimpleWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGeneric1x1Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGeneric1x1Test(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGenericSimpleWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGenericSimpleWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGenericTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGenericTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGenericGroupedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGenericGroupedTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGeneric1x1BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGeneric1x1BigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGeneric1x1BatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGeneric1x1BatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGenericBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGenericBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGenericBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGenericBatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGenericGroupedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGenericGroupedBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGenericPackedGroupsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 1, 383, 128);
  const int dst_channels = 64;
  auto status = ConvGenericPackedGroupsTest(*exec_env, precision(), storage(),
                                            src_shape, dst_channels);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalWfloatTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 1, 24, 128);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWfloatTest(*exec_env, precision(), storage(),
                                              src_shape, dst_channels);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalBatchedWfloatTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 64);
  const int dst_channels = 128;
  auto status = ConvGenericExternalWfloatTest(*exec_env, precision(), storage(),
                                              src_shape, dst_channels,
                                              /*batched_weights=*/true);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalWi8Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 1, 24, 128);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi8Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalBatchedWi8Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 64);
  const int dst_channels = 128;
  auto status = ConvGenericExternalWi8Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/true);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalGroupedWi8Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 16 * 12);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi8Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/false,
                                           /*group_size=*/12);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalGroupedBatchedWi8Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 16 * 12);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi8Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/true,
                                           /*group_size=*/12);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest,
       ConvGenericExternalBatchedWi8NonBatchedScalesTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 16 * 12);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi8Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/true,
                                           /*group_size=*/-1,
                                           /*scale_zp_batch=*/1);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalWi4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 1, 24, 128);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi4Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalBatchedWi4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 64);
  const int dst_channels = 128;
  auto status = ConvGenericExternalWi4Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/true);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalGroupedWi4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 16 * 12);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi4Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/false,
                                           /*group_size=*/12);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalGroupedBatchedWi4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 16 * 12);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi4Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/true,
                                           /*group_size=*/12);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalWi2Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 1, 24, 128);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi2Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalBatchedWi2Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 64);
  const int dst_channels = 128;
  auto status = ConvGenericExternalWi2Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/true);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalGroupedWi2Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 16 * 12);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi2Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/false,
                                           /*group_size=*/12);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGenericExternalGroupedBatchedWi2Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const BHWC src_shape(1, 6, 12, 16 * 12);
  const int dst_channels = 64;
  auto status = ConvGenericExternalWi2Test(*exec_env, precision(), storage(),
                                           src_shape, dst_channels,
                                           /*batched_weights=*/true,
                                           /*group_size=*/12);
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
  ABSL_ASSERT_OK(status);
}

TEST_P(ConvGenericFloatTest, ConvGeneric3d1x1x1BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGeneric3d1x1x1BigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGeneric3d1x1x1BatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ConvGeneric3d1x1x1BatchedBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGeneric3dBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGeneric3dBigTest(*exec_env, precision(), storage()));
}

TEST_P(ConvGenericFloatTest, ConvGeneric3dBatchedBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ConvGeneric3dBatchedBigTest(*exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvGenericFloatTestSuite, ConvGenericFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<ConvGenericFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class TiledTest
    : public Test,
      public WithParamInterface<
          std::tuple<CalculationsPrecision, TensorStorageType, int>> {};

TEST_P(TiledTest, ConvGenericWinograd3x3TileNxNTest) {
  auto [precision, storage, kTileSize] = GetParam();
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ConvGenericWinograd3x3TileNxNTest(*exec_env, precision, storage,
                                              kTileSize));
}

INSTANTIATE_TEST_SUITE_P(
    TiledTestSuite, TiledTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn({6, 8, 10})),
    [](const TestParamInfo<TiledTest::ParamType>& info) {
      const auto float_param =
          std::make_tuple(std::get<0>(info.param), std::get<1>(info.param));
      return absl::StrCat(ToString(float_param), "_tile_size_",
                          std::get<2>(info.param));
    });

class IntTest : public Test,
                public WithParamInterface<
                    std::tuple<TensorStorageType, TensorStorageType>> {
 protected:
  TensorStorageType src_storage() const { return std::get<0>(GetParam()); }
  TensorStorageType dst_storage() const { return std::get<1>(GetParam()); }
};

TEST_P(IntTest, ConvGeneric1x1Int8SymmetricTest) {
  if (!SupportsConvGenericInt8(exec_env->GetGpuInfo())) {
    GTEST_SKIP()
        << "ConvGeneric1x1Int8SymmetricTest not supported on this device.";
  }
  if (!exec_env->IsStorageSupported(src_storage(), DataType::INT8) ||
      !exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP()
        << "ConvGeneric1x1Int8SymmetricTest not supported for src storage: "
        << ToString(src_storage())
        << " and dst storage: " << ToString(dst_storage());
  }
  ABSL_ASSERT_OK(
      ConvGeneric1x1Int8SymmetricTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(IntTest, ConvGenericInt8BigTest) {
  if (!SupportsConvGenericInt8(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "ConvGenericInt8BigTest not supported on this device.";
  }

  // check dst
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  ABSL_ASSERT_OK(ConvGenericInt8BigTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(IntTest, ConvGenericInt8ExternalWeightsBigTest) {
  if (!SupportsConvGenericInt8(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "ConvGenericInt8ExternalWeightsBigTest not supported on "
                    "this device.";
  }
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  ABSL_ASSERT_OK(ConvGenericInt8ExternalWeightsBigTest(*exec_env, src_storage(),
                                                  dst_storage()));
}

TEST_P(IntTest, ConvGenericInt8WithSrcQuantizationBigTest) {
  if (!SupportsConvGenericInt8(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "ConvGenericInt8WeightsInt4WithSrcQuantization not "
                    "supported on this device.";
  }
  ABSL_ASSERT_OK(ConvGenericInt8WithSrcQuantizationBigTest(*exec_env, src_storage(),
                                                      dst_storage()));
}

TEST_P(IntTest, ConvGenericInt8WeightsInt4WithSrcQuantizationBigTest) {
  if (!SupportsConvGenericInt8(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "ConvGenericInt8WeightsInt4WithSrcQuantization not "
                    "supported on this device.";
  }
  ABSL_ASSERT_OK(ConvGenericInt8WeightsInt4WithSrcQuantizationBigTest(
      *exec_env, src_storage(), dst_storage()));
}

TEST_P(IntTest, ConvGenericInt4BigTest) {
  const BHWC src_shape =
      BHWC(1, 17, 13, 125);  // src_slices = 32,
                             // divisible by 16 for triggering Intel wave matmul
  if (!SupportsConvGenericInt4(exec_env->GetGpuInfo(), src_shape)) {
    GTEST_SKIP() << "ConvGenericInt4 not supported on this device.";
  }
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  ABSL_ASSERT_OK(ConvGenericInt4BigTest(*exec_env, src_storage(), dst_storage(),
                                   src_shape));
}

TEST_P(IntTest, ConvGenericInt4ExternalWeightsBigTest) {
  const BHWC src_shape =
      BHWC(1, 17, 13, 125);  // src_slices = 32,
                             // divisible by 16 for triggering Intel wave matmul
  if (!SupportsConvGenericInt4(exec_env->GetGpuInfo(), src_shape)) {
    GTEST_SKIP() << "ConvGenericInt4 not supported on this device.";
  }
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  ABSL_ASSERT_OK(ConvGenericInt4ExternalWeightsBigTest(*exec_env, src_storage(),
                                                  dst_storage(), src_shape));
}

TEST_P(IntTest, ConvGenericInt4WithSrcQuantizationBigTest) {
  const BHWC src_shape =
      BHWC(1, 17, 13, 125);  // src_slices = 32,
                             // divisible by 16 for triggering Intel wave matmul
  if (!SupportsConvGenericInt4(exec_env->GetGpuInfo(), src_shape)) {
    GTEST_SKIP() << "ConvGenericInt4 not supported on this device.";
  }
  ABSL_ASSERT_OK(ConvGenericInt4WithSrcQuantizationBigTest(
      *exec_env, src_storage(), dst_storage(), src_shape));
}

INSTANTIATE_TEST_SUITE_P(
    IntTestSuite, IntTest,
    Combine(ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<IntTest::ParamType>& info) {
      const std::string out =
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)));
      return absl::StrReplaceAll(out, {{":", "_"}});
    });

}  // namespace ml_drift
