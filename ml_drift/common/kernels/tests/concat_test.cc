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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/concat_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

// Non float tests.
class ConcatTypedTest : public Test,
                        public WithParamInterface<TensorStorageType> {};

TEST_P(ConcatTypedTest, ChannelsBool) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ConcatChannelsBoolTest(*exec_env, GetParam()));
}

TEST_P(ConcatTypedTest, Int8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ConcatIntTest<DataType::INT8>(*exec_env, GetParam()));
}

TEST_P(ConcatTypedTest, Int16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ConcatIntTest<DataType::INT16>(*exec_env, GetParam()));
}

TEST_P(ConcatTypedTest, Int32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ConcatIntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(ConcatTypedTest, Uint8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT8)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ConcatIntTest<DataType::UINT8>(*exec_env, GetParam()));
}

TEST_P(ConcatTypedTest, Uint16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ConcatIntTest<DataType::UINT16>(*exec_env, GetParam()));
}

TEST_P(ConcatTypedTest, Uint32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ConcatIntTest<DataType::UINT32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ConcatTypedTestSuite, ConcatTypedTest, ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<ConcatTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class ConcatFloatTest : public DataTypeTest {};

TEST_P(ConcatFloatTest, Width) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatWidthTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, Height) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatHeightTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, Channels) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ChannelsAlignedx4) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatChannelsAlignedx4Test(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatWidthBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatWidthBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatWidthBatchedBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatWidthBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatHeightBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatHeightBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatHeightBatchedBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatHeightBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatBatchBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatBatchBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatDepthBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatDepthBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatDepthBatchedBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatDepthBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatChannelsBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatChannelsBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatChannelsBatchedBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatChannelsBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatChannelsx4Big) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatChannelsx4BigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatChannelsx4BatchedBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatChannelsx4BatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ConcatFloatTest, ConcatChannelsBHWDCBig) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConcatChannelsBHWDCBigTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ConcatFloatTestSuite, ConcatFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ConcatFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
