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
#include "ml_drift/common/kernels/tests/add_test_util.h"
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
class AddTypedTest : public Test,
                     public WithParamInterface<TensorStorageType> {};

TEST_P(AddTypedTest, TwoEqualTensors) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kBfloat16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(AddTwoEqualTensorsBFloatTest(*exec_env, GetParam()));
}

TEST_P(AddTypedTest, TwoEqualInt8Tensors) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(AddTwoEqualIntTensorsTest<DataType::kInt8>(*exec_env, GetParam()));
}

TEST_P(AddTypedTest, TwoEqualInt16Tensors) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(AddTwoEqualIntTensorsTest<DataType::kInt16>(*exec_env, GetParam()));
}

TEST_P(AddTypedTest, TwoEqualInt32Tensors) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(AddTwoEqualIntTensorsTest<DataType::kInt32>(*exec_env, GetParam()));
}

TEST_P(AddTypedTest, TwoEqualUint8Tensors) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(
      AddTwoEqualUintTensorsTest<DataType::kUint8>(*exec_env, GetParam()));
}

TEST_P(AddTypedTest, TwoEqualUint16Tensors) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(
      AddTwoEqualUintTensorsTest<DataType::kUint16>(*exec_env, GetParam()));
}

TEST_P(AddTypedTest, TwoEqualUint32Tensors) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(
      AddTwoEqualUintTensorsTest<DataType::kUint32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    AddTypedTestSuite, AddTypedTest, ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<AddTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class AddFloatTest : public DataTypeTest {};

TEST_P(AddFloatTest, TwoEqualTensors) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddTwoEqualTensorsTest(*exec_env, data_type(), storage()));
}

TEST_P(AddFloatTest, FirstTensorHasMoreChannelsThanSecond) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddFirstTensorHasMoreChannelsThanSecondTest(*exec_env, data_type(),
                                                        storage()));
}

TEST_P(AddFloatTest, FirstTensorHasLessChannelsThanSecond) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddFirstTensorHasLessChannelsThanSecondTest(*exec_env, data_type(),
                                                        storage()));
}

TEST_P(AddFloatTest, AddBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddBigTest(*exec_env, data_type(), storage()));
}

TEST_P(AddFloatTest, AddBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(AddFloatTest, AddNotEqualBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddNotEqualBigTest(*exec_env, data_type(), storage()));
}

TEST_P(AddFloatTest, AddNotEqualBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddNotEqualBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(AddFloatTest, AddNotEqualFirstTensorBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddNotEqualFirstTensorBigTest(*exec_env, data_type(), storage()));
}

TEST_P(AddFloatTest, AddNotEqualFirstTensorBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      AddNotEqualFirstTensorBatchedBigTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    AddFloatTestSuite, AddFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<AddFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

TEST_P(AddFloatTest, AddBroadcast5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddBroadcast5DTest(*exec_env, data_type(), storage()));
}

}  // namespace ml_drift
