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
#include "ml_drift/common/kernels/tests/fully_connected_oi_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

class FullyConnectedOIFloatTest : public FloatTest {};

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOII4O4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedOITest(*exec_env, precision(), storage(), /*isI4O4=*/true));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIO4I4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedOITest(*exec_env, precision(), storage(),
                                 /*isI4O4=*/false));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOII4O4Wi8) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedOIWi8(*exec_env, precision(), storage(), /*isI4O4=*/true));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIO4I4Wi8) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedOIWi8(*exec_env, precision(), storage(), /*isI4O4=*/false));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIWi4) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedOIWi4(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIRuntimeSrcTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedOIRuntimeSrcTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIRuntimeDstTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedOIRuntimeDstTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIPackedGroupsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      FullyConnectedOIPackedGroupsTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIRingedOTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedOIRingedOTest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIRingedITest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedOIRingedITest(*exec_env, precision(), storage()));
}

TEST_P(FullyConnectedOIFloatTest, FullyConnectedOIWi4Sparse2x4) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->GetGpuInfo().IsApiOpenCl()) {
    GTEST_SKIP() << "Sparse 2x4 weights are not supported for non-OpenCL APIs.";
  }
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FullyConnectedOIWi4Sparse2x4(*exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    FullyConnectedOIFloatTestSuite, FullyConnectedOIFloatTest,
    testing::Combine(
        testing::ValuesIn(GetCalculationsPrecisions()),
        testing::ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const testing::TestParamInfo<FullyConnectedOIFloatTest::ParamType>&
           info) { return ToString(info.param); });

}  // namespace ml_drift
