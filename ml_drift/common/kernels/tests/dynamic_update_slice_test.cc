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
#include "ml_drift/common/kernels/tests/dynamic_update_slice_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Non float tests.
using DynamicUpdateSliceTypedTest = TestWithParam<TensorStorageType>;

TEST_P(DynamicUpdateSliceTypedTest, Bool) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceBoolTest(*exec_env, GetParam()));
}

TEST_P(DynamicUpdateSliceTypedTest, Int8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceIntTest<DataType::kInt8>(*exec_env, GetParam()));
}

TEST_P(DynamicUpdateSliceTypedTest, Int16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceIntTest<DataType::kInt16>(*exec_env, GetParam()));
}

TEST_P(DynamicUpdateSliceTypedTest, Int32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceIntTest<DataType::kInt32>(*exec_env, GetParam()));
}

TEST_P(DynamicUpdateSliceTypedTest, Uint8) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceIntTest<DataType::kUint8>(*exec_env, GetParam()));
}

TEST_P(DynamicUpdateSliceTypedTest, Uint16) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(
      DynamicUpdateSliceIntTest<DataType::kUint16>(*exec_env, GetParam()));
}

TEST_P(DynamicUpdateSliceTypedTest, Uint32) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kUint32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(
      DynamicUpdateSliceIntTest<DataType::kUint32>(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    DynamicUpdateSliceTypedTestSuite, DynamicUpdateSliceTypedTest,
    ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<DynamicUpdateSliceTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class DynamicUpdateSliceFloatTest : public DataTypeTest {};

TEST_P(DynamicUpdateSliceFloatTest, DynamicUpdateSliceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceTest(*exec_env, data_type(), storage()));
}

TEST_P(DynamicUpdateSliceFloatTest, DynamicUpdateSliceTwoDimensionSliceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceTwoDimensionSliceTest(*exec_env, data_type(),
                                                    storage()));
}

TEST_P(DynamicUpdateSliceFloatTest, DynamicUpdateSliceThreeDimensionSliceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceThreeDimensionSliceTest(*exec_env, data_type(),
                                                      storage()));
}

TEST_P(DynamicUpdateSliceFloatTest, DynamicUpdateSliceFourDimensionSliceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceFourDimensionSliceTest(*exec_env, data_type(),
                                                     storage()));
}

TEST_P(DynamicUpdateSliceFloatTest,
       DynamicUpdateSliceStartIndicesThreeValuesSliceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceStartIndicesThreeValuesSliceTest(
      *exec_env, data_type(), storage()));
}

TEST_P(DynamicUpdateSliceFloatTest,
       DynamicUpdateSliceStartIndicesTwoValuesSliceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceStartIndicesTwoValuesSliceTest(
      *exec_env, data_type(), storage()));
}

TEST_P(DynamicUpdateSliceFloatTest, DynamicUpdateSliceClampTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceClampTest(*exec_env, data_type(), storage()));
}

TEST_P(DynamicUpdateSliceFloatTest, DynamicUpdateSliceConversionTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  DataType src_type = (data_type() == DataType::kFloat32) ? DataType::kFloat16
                                                          : DataType::kFloat32;
  if (!exec_env->IsStorageSupported(storage(), src_type)) {
    GTEST_SKIP() << "Unsupported src data type: " << ToString(src_type);
  }
  ABSL_ASSERT_OK(DynamicUpdateSliceConversionTest(*exec_env, src_type, data_type(),
                                             storage()));
}

INSTANTIATE_TEST_SUITE_P(
    DynamicUpdateSliceFloatTestSuite, DynamicUpdateSliceFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<DynamicUpdateSliceFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
