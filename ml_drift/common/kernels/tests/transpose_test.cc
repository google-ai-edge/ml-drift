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
#include "ml_drift/common/kernels/tests/transpose_test_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Non float tests.
using TransposeTypedTest = TestWithParam<TensorStorageType>;

TEST_P(TransposeTypedTest, Int8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TransposeIntTest<DataType::INT8>(*exec_env, GetParam()));
}

TEST_P(TransposeTypedTest, Int16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TransposeIntTest<DataType::INT16>(*exec_env, GetParam()));
}

TEST_P(TransposeTypedTest, Int32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TransposeIntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(TransposeTypedTest, Uint8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TransposeUintTest<DataType::UINT8>(*exec_env, GetParam()));
}

TEST_P(TransposeTypedTest, Uint16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TransposeUintTest<DataType::UINT16>(*exec_env, GetParam()));
}

TEST_P(TransposeTypedTest, Uint32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(TransposeUintTest<DataType::UINT32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    TransposeTypedTestSuite, TransposeTypedTest,
    ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<TransposeTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class TransposeFloatTest : public DataTypeTest {};

TEST_P(TransposeFloatTest, TransposeTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TransposeTest(*exec_env, data_type(), storage()));
}

TEST_P(TransposeFloatTest, TransposeBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(1, 6, 10, 8);

  TransposeAttributes attr;
  attr.perm = BHWC(0, 3, 1, 2);

  ABSL_ASSERT_OK(TransposeTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(TransposeFloatTest, TransposeBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(3, 6, 10, 8);

  TransposeAttributes attr;
  attr.perm = BHWC(2, 3, 1, 0);

  ABSL_ASSERT_OK(TransposeTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(TransposeFloatTest, TransposeNoChannelsPermutationBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(1, 6, 10, 8);

  TransposeAttributes attr;
  attr.perm = BHWC(0, 2, 1, 3);

  ABSL_ASSERT_OK(TransposeTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(TransposeFloatTest, TransposeNoChannelsPermutationBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(3, 6, 10, 8);

  TransposeAttributes attr;
  attr.perm = BHWC(2, 0, 1, 3);

  ABSL_ASSERT_OK(TransposeTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(TransposeFloatTest, Transpose3DBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWDC(1, 2, 3, 4, 5);

  Transpose3DAttributes attr;
  attr.perm = BHWDC(0, 4, 1, 2, 3);

  ABSL_ASSERT_OK(
      Transpose3DTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(TransposeFloatTest, Transpose3DBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWDC(3, 2, 3, 4, 5);

  Transpose3DAttributes attr;
  attr.perm = BHWDC(2, 4, 1, 0, 3);

  ABSL_ASSERT_OK(
      Transpose3DTest(*exec_env, attr, src_shape, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    TransposeFloatTestSuite, TransposeFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<TransposeFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
