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

#include "ml_drift/common/kernels/fully_connected.h"

#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/fully_connected_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class FullyConnectedFloatTest : public FloatTest {};

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8Test(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8BlockwiseAttributesTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8BlockwiseAttributesTest(*exec_env, precision(),
                                                      storage()));
}

TEST_P(FullyConnectedFloatTest,
       FullyConnectedInt8BlockwiseAttributesWithZeroPointsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8BlockwiseAttributesWithZeroPointsTest(
      *exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedWeightsAsSpatialTensorTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedWeightsAsSpatialTensorTest(
      *exec_env, precision(), storage(), OHWI(340, 1, 1, 128),
      BHWC(1, 1, 1, 128)));
}

TEST_P(FullyConnectedFloatTest,
       FullyConnectedWeightsAsSpatialTensorBatchedHeight) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedWeightsAsSpatialTensorTest(
      *exec_env, precision(), storage(), OHWI(340, 4, 1, 128),
      BHWC(1, 4, 2, 128)));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt4Sparse2x4) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt4Sparse2x4Test(*exec_env, precision(), storage(),
                                            BHWC(1, 1, 1, 256), 256));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedExternalWeightsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedExternalWeightsBigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedBatchedWeightsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedBatchedWeightsBigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedRingedOTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedRingedOTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedRingedITest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedRingedITest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8BigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8Width2Batch2BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedInt8Width2Batch2BigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8GroupedQuantizationBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8GroupedQuantizationBigTest(*exec_env, precision(),
                                                         storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8ExternalBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedInt8ExternalBigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8BatchedWeightsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  if (!exec_env->IsStorageSupported(storage(), DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(DataType::kInt32)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8BatchedWeightsBigTest(*exec_env, precision(),
                                                    storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt8BatchedWeightsIdsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8BatchedWeightsIdsBigTest(
      *exec_env, precision(), storage(), BHWC(1, 1, 1, 32), BHWC(1, 1, 1, 1)));
  ABSL_ASSERT_OK(FullyConnectedInt8BatchedWeightsIdsBigTest(
      *exec_env, precision(), storage(), BHWC(1, 8, 1, 32), BHWC(1, 1, 1, 8)));
  ABSL_ASSERT_OK(FullyConnectedInt8BatchedWeightsIdsBigTest(
      *exec_env, precision(), storage(), BHWC(1, 128, 1, 64),
      BHWC(1, 1, 1, 128)));
}

TEST_P(FullyConnectedFloatTest,
       FullyConnectedInt8BatchedWeightsIdsWithBroadcastingTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8BatchedWeightsIdsBigTest(
      *exec_env, precision(), storage(), BHWC(1, 1, 1, 32), BHWC(1, 1, 1, 4)));
}

TEST_P(FullyConnectedFloatTest,
       FullyConnectedInt8ExternalGroupedQuantizationBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt8ExternalGroupedQuantizationBigTest(
      *exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt4BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt4BigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt4BlockwiseTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt4BlockwiseTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt4WidthIs4BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedInt4WidthIs4BigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt4GroupedQuantizationBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt4GroupedQuantizationBigTest(*exec_env, precision(),
                                                         storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt4ExternalWeightsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt4ExternalWeightsBigTest(*exec_env, precision(),
                                                     storage()));
}

TEST_P(FullyConnectedFloatTest,
       FullyConnectedInt4ExternalGroupedQuantizationBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt4ExternalGroupedQuantizationBigTest(
      *exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt2BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt2BigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt2BlockwiseTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt2BlockwiseTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt2WidthIs4BigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedInt2WidthIs4BigTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt2GroupedQuantizationBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt2GroupedQuantizationBigTest(*exec_env, precision(),
                                                         storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedInt2ExternalWeightsBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt2ExternalWeightsBigTest(*exec_env, precision(),
                                                     storage()));
}

TEST_P(FullyConnectedFloatTest,
       FullyConnectedInt2ExternalGroupedQuantizationBigTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedInt2ExternalGroupedQuantizationBigTest(
      *exec_env, precision(), storage()));
}

TEST_P(FullyConnectedFloatTest, FullyConnectedPackedGroupsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedPackedGroupsTest(*exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    FullyConnectedFloatTestSuite, FullyConnectedFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<FullyConnectedFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class FullyConnectedIntTest
    : public testing::Test,
      public testing::WithParamInterface<
          std::tuple<TensorStorageType, TensorStorageType>> {
  // Convenience accessor methods
 protected:
  TensorStorageType src_storage() const { return std::get<0>(GetParam()); }
  TensorStorageType dst_storage() const { return std::get<1>(GetParam()); }
};

TEST_P(FullyConnectedIntTest, FullyConnectedSi8Wi8BigTest) {
  if (!SupportsFullyConnectedUint8Math(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Unsupported FullyConnectedIntTest";
  }
  if (!exec_env->IsStorageSupported(src_storage(), DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(DataType::kUint8);
  }
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::kUint32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(dst_storage())
                 << " to " << ToString(DataType::kUint32);
  }
  ABSL_ASSERT_OK(
      FullyConnectedSi8Wi8BigTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(FullyConnectedIntTest, FullyConnectedSi8Wi4BigTest) {
  if (!SupportsFullyConnectedUint8Math(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Unsupported FullyConnectedIntTest";
  }
  if (!exec_env->IsStorageSupported(src_storage(), DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(DataType::kUint8);
  }
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::kUint32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(dst_storage())
                 << " to " << ToString(DataType::kUint32);
  }
  ABSL_ASSERT_OK(
      FullyConnectedSi8Wi4BigTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(FullyConnectedIntTest, FullyConnectedSi8Wi2BigTest) {
  if (!SupportsFullyConnectedUint8Math(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Unsupported FullyConnectedIntTest";
  }
  if (!exec_env->IsStorageSupported(src_storage(), DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(DataType::kUint8);
  }
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::kUint32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(dst_storage())
                 << " to " << ToString(DataType::kUint32);
  }
  ABSL_ASSERT_OK(
      FullyConnectedSi8Wi2BigTest(*exec_env, src_storage(), dst_storage()));
}

INSTANTIATE_TEST_SUITE_P(
    Suite, FullyConnectedIntTest,
    Combine(ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<FullyConnectedIntTest::ParamType>& info) {
      const std::string name =
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

}  // namespace ml_drift
