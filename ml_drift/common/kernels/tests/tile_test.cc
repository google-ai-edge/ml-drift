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
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/tile_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class TileFloatTest : public DataTypeTest {};

TEST_P(TileFloatTest, TileChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TileChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(TileFloatTest, TileChannelsX4Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TileChannelsX4Test(*exec_env, data_type(), storage()));
}

TEST_P(TileFloatTest, TileWidthTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TileWidthTest(*exec_env, data_type(), storage()));
}

TEST_P(TileFloatTest, TileHeightTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TileHeightTest(*exec_env, data_type(), storage()));
}

TEST_P(TileFloatTest, TileHWCTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TileHWCTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    TileFloatTestSuite, TileFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<TileFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

// Non-float tests.

class TileTypedTest : public Test,
                      public WithParamInterface<TensorStorageType> {};

TEST_P(TileTypedTest, TileChannelsInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsIntTest<DataType::INT8>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsInt16Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsIntTest<DataType::INT16>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsInt32Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsIntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsUInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsIntTest<DataType::UINT8>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsUInt16Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsIntTest<DataType::UINT16>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsUInt32Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsIntTest<DataType::UINT32>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsX4Int8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsX4IntTest<DataType::INT8>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsX4Int16Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsX4IntTest<DataType::INT16>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsX4Int32Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsX4IntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsX4UInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsX4IntTest<DataType::UINT8>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsX4UInt16Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsX4IntTest<DataType::UINT16>(*exec_env, GetParam()));
}

TEST_P(TileTypedTest, TileChannelsX4UInt32Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TileChannelsX4IntTest<DataType::UINT32>(*exec_env, GetParam()));
}

TEST_P(TileFloatTest, Tile5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Tile5DTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    TileTypedTestSuite, TileTypedTest, ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<TileTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

}  // namespace ml_drift
