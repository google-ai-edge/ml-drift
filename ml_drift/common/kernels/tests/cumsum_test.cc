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

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/cumsum_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::TestWithParam;
using ::testing::WithParamInterface;

// Non float tests.
using CumsumTypedTest = TestWithParam<TensorStorageType>;

TEST_P(CumsumTypedTest, Int8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(CumsumIntTest<DataType::kInt8>(*exec_env, GetParam()));
}

TEST_P(CumsumTypedTest, Int16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(CumsumIntTest<DataType::kInt16>(*exec_env, GetParam()));
}

TEST_P(CumsumTypedTest, Int32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(CumsumIntTest<DataType::kInt32>(*exec_env, GetParam()));
}

TEST_P(CumsumTypedTest, Uint8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(CumsumIntTest<DataType::kUint8>(*exec_env, GetParam()));
}

TEST_P(CumsumTypedTest, Uint16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(CumsumIntTest<DataType::kUint16>(*exec_env, GetParam()));
}

TEST_P(CumsumTypedTest, Uint32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(CumsumIntTest<DataType::kUint32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    CumsumTypedTestSuite, CumsumTypedTest, ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<CumsumTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class CumsumFloatTest
    : public testing::Test,
      public WithParamInterface<std::tuple<DataType, TensorStorageType, Axis>> {
 protected:
  DataType data_type() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
  Axis axis() const { return std::get<2>(GetParam()); }
};

TEST_P(CumsumFloatTest, HWC) {
  if (axis() == Axis::kBatch) {
    GTEST_SKIP() << "No batch axis for HWC layout.";
  }
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(CumsumHWCTest(*exec_env, data_type(), storage(), axis()));
}

TEST_P(CumsumFloatTest, BHWC) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(CumsumBHWCTest(*exec_env, data_type(), storage(), axis()));
}

TEST_P(CumsumFloatTest, Cumsum5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Cumsum5DTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    CumsumFloatTestSuite, CumsumFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes()),
            ValuesIn({Axis::kBatch, Axis::kHeight, Axis::kWidth,
                      Axis::kChannels})),
    [](const TestParamInfo<CumsumFloatTest::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param))),
          {{":", ""}});
    });

}  // namespace ml_drift
