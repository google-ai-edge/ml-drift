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
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/winograd_test_util.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class WinogradFloatTest : public DataTypeTest {};

TEST_P(WinogradFloatTest, Winograd3x3ForwardTiledTile6Small) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Winograd3x3ForwardTiledTest(*exec_env, data_type(), storage(),
                                        BHWC(1, 4, 4, 1), /*tile_size=*/6));
}

TEST_P(WinogradFloatTest, Winograd3x3BackwardTiledTile6Small) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Winograd3x3BackwardTiledTest(*exec_env, data_type(), storage(),
                                         BHWC(1, 4, 4, 1), /*tile_size=*/6));
}

TEST_P(WinogradFloatTest, Winograd3x3ForwardTiledTile8Big) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Winograd3x3ForwardTiledTest(*exec_env, data_type(), storage(),
                                        BHWC(1, 60, 60, 4), /*tile_size=*/8));
}

TEST_P(WinogradFloatTest, Winograd3x3BackwardTiledTile8Big) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Winograd3x3BackwardTiledTest(*exec_env, data_type(), storage(),
                                         BHWC(1, 60, 60, 4), /*tile_size=*/8));
}

TEST_P(WinogradFloatTest, Winograd4x4To36) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Winograd4x4To36Test(*exec_env, data_type(), storage()));
}

TEST_P(WinogradFloatTest, Winograd4x4To36Batch) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Winograd4x4To36BatchTest(*exec_env, data_type(), storage()));
}

TEST_P(WinogradFloatTest, Winograd36To4x4) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(Winograd36To4x4Test(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    WinogradFloatTestSuite, WinogradFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<WinogradFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

TEST(WinogradWeightsConversionTest, Winograd3x3To36) {
  MLD_ASSERT_OK(Winograd3x3To36Test(*exec_env));
}

}  // namespace ml_drift
