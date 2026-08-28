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
#include "ml_drift/common/kernels/tests/dot_general_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::TestWithParam;

// Non float tests.
using DotGeneralTypedTest = TestWithParam<TensorStorageType>;

TEST_P(DotGeneralTypedTest, Bfloat) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BFLOAT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DotGeneral2DBfloatTest(*exec_env, GetParam()));
}

TEST_P(DotGeneralTypedTest, Int8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DotGeneral2DIntTest<DataType::INT8>(*exec_env, GetParam()));
}

TEST_P(DotGeneralTypedTest, Int16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DotGeneral2DIntTest<DataType::INT16>(*exec_env, GetParam()));
}

TEST_P(DotGeneralTypedTest, Int32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DotGeneral2DIntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(DotGeneralTypedTest, Uint8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DotGeneral2DIntTest<DataType::UINT8>(*exec_env, GetParam()));
}

TEST_P(DotGeneralTypedTest, Uint16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DotGeneral2DIntTest<DataType::UINT16>(*exec_env, GetParam()));
}

TEST_P(DotGeneralTypedTest, Uint32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DotGeneral2DIntTest<DataType::UINT32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    DotGeneralTypedTestSuite, DotGeneralTypedTest,
    ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<DotGeneralTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class DotGeneralFloatTest : public DataTypeTest {};

TEST_P(DotGeneralFloatTest, DotGeneral1DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DotGeneral1DTest(*exec_env, data_type(), storage()));
}

TEST_P(DotGeneralFloatTest, DotGeneral2DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DotGeneral2DTest(*exec_env, data_type(), storage()));
}

TEST_P(DotGeneralFloatTest, DotGeneral3DBatchTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DotGeneral3DBatchTest(*exec_env, data_type(), storage()));
}

TEST_P(DotGeneralFloatTest, DotGeneral4DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DotGeneral4DTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    DotGeneralFloatTestSuite, DotGeneralFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<DotGeneralFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
