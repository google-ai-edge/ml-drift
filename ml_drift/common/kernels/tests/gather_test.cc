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
#include "ml_drift/common/kernels/tests/gather_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Non float tests.
using GatherTypedTest = TestWithParam<TensorStorageType>;

TEST_P(GatherTypedTest, Int8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GatherWidthIntTest<DataType::INT8>(*exec_env, GetParam()));
}

TEST_P(GatherTypedTest, Int16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GatherWidthIntTest<DataType::INT16>(*exec_env, GetParam()));
}

TEST_P(GatherTypedTest, Int32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GatherWidthIntTest<DataType::INT32>(*exec_env, GetParam()));
}

TEST_P(GatherTypedTest, Uint8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GatherWidthIntTest<DataType::UINT8>(*exec_env, GetParam()));
}

TEST_P(GatherTypedTest, Uint16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GatherWidthIntTest<DataType::UINT16>(*exec_env, GetParam()));
}

TEST_P(GatherTypedTest, Uint32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GatherWidthIntTest<DataType::UINT32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    GatherTypedTestSuite, GatherTypedTest,
    ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
    [](const TestParamInfo<GatherTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class GatherFloatTest : public DataTypeTest {};

TEST_P(GatherFloatTest, GatherWidthTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(GatherTest(*exec_env, data_type(), storage(), Axis::WIDTH));
}

TEST_P(GatherFloatTest, GatherHeightTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(GatherTest(*exec_env, data_type(), storage(), Axis::HEIGHT));
}

TEST_P(GatherFloatTest, GatherChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(GatherTest(*exec_env, data_type(), storage(), Axis::CHANNELS));
}

TEST_P(GatherFloatTest, GatherBatchTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(GatherTest(*exec_env, data_type(), storage(), Axis::BATCH));
}

INSTANTIATE_TEST_SUITE_P(
    GatherFloatTestSuite, GatherFloatTest,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<GatherFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
