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
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/scatter_nd_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Integer updates: scatter mapping and last-write-wins for duplicate indices.
using ScatterNdTypedTest = TestWithParam<TensorStorageType>;

TEST_P(ScatterNdTypedTest, ScatterInt32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(ScatterNdIntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(ScatterNdTypedTest, DuplicateInt32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(
      ScatterNdDuplicateIntTest<DataType::INT32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ScatterNdTypedTestSuite, ScatterNdTypedTest,
    ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
    [](const TestParamInfo<ScatterNdTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

// Float updates: duplicate indices accumulate (TFLite sum semantics).
class ScatterNdFloatTest : public DataTypeTest {};

TEST_P(ScatterNdFloatTest, DuplicateSum) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ScatterNdSumFloatTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ScatterNdFloatTestSuite, ScatterNdFloatTest,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<ScatterNdFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
