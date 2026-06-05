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

#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/reduce_test_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

// Non float tests.
class ReduceTypedTest : public Test,
                        public WithParamInterface<TensorStorageType> {};

TEST_P(ReduceTypedTest, ReduceSumChannelsInt8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ReduceSumChannelsIntTest<DataType::INT8>(*exec_env, GetParam()));
}

TEST_P(ReduceTypedTest, ReduceSumChannelsInt16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ReduceSumChannelsIntTest<DataType::INT16>(*exec_env, GetParam()));
}

TEST_P(ReduceTypedTest, ReduceSumChannelsInt32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ReduceSumChannelsIntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(ReduceTypedTest, ReduceProductChannelsUInt8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(
      ReduceProductChannelsUIntTest<DataType::UINT8>(*exec_env, GetParam()));
}

TEST_P(ReduceTypedTest, ReduceProductChannelsUInt16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(
      ReduceProductChannelsUIntTest<DataType::UINT16>(*exec_env, GetParam()));
}

TEST_P(ReduceTypedTest, ReduceProductChannelsUInt32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(
      ReduceProductChannelsUIntTest<DataType::UINT32>(*exec_env, GetParam()));
}

TEST_P(ReduceTypedTest, ReduceAny) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ReduceAnyTest(*exec_env, GetParam()));
}

TEST_P(ReduceTypedTest, ReduceBool) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ReduceAllTest(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ReduceTypedTestSuite, ReduceTypedTest, ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<ReduceTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class ReduceFloatTest : public DataTypeTest {};

TEST_P(ReduceFloatTest, MeanHWTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(MeanHWTest(*exec_env, data_type(), storage()));
}

TEST_P(ReduceFloatTest, ReduceSumChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceSumChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(ReduceFloatTest, ReduceProductChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceProductChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(ReduceFloatTest, ReduceMaxChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceMaxChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(ReduceFloatTest, ReduceMinChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceMinChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(ReduceFloatTest, ReduceMaxIndChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceMaxIndChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(ReduceFloatTest, ReduceMaxIndHeightTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceMaxIndHeightTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ReduceFloatTestSuite, ReduceFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ReduceFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class ReduceBigFloatTest
    : public Test,
      public WithParamInterface<
          std::tuple<DataType, TensorStorageType, OperationType>> {
  // Convenience accessor methods
 protected:
  DataType data_type() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
  OperationType op_type() const { return std::get<2>(GetParam()); }
};

TEST_P(ReduceBigFloatTest, ReduceHWBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceHWBigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceHWBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      ReduceHWBatchedBigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceHBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceHBigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceBHBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceBHBigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceCBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceCBigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceCx4BigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceCx4BigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceHWCBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceHWCBigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceBHDBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceBHDBigTest(*exec_env, data_type(), storage(), op_type()));
}

TEST_P(ReduceBigFloatTest, ReduceBHWDBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ReduceBHWDBigTest(*exec_env, data_type(), storage(), op_type()));
}

std::vector<OperationType> GetReduceOps() {
  return {OperationType::MEAN, OperationType::REDUCE_SUM,
          OperationType::REDUCE_PRODUCT, OperationType::REDUCE_MINIMUM,
          OperationType::REDUCE_MAXIMUM};
}

INSTANTIATE_TEST_SUITE_P(
    Suite, ReduceBigFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes()),
            ValuesIn(GetReduceOps())),
    [](const TestParamInfo<ReduceBigFloatTest::ParamType>& info) {
      const std::string name =
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

}  // namespace ml_drift
