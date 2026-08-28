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
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/elementwise_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Non float tests.
using ElementwiseTypedTest = TestWithParam<TensorStorageType>;

TEST_P(ElementwiseTypedTest, CosIntTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(CosIntTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, FloorDivIntTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(FloorDivIntTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, FloorModIntTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(FloorModIntTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, SignInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(SignInt8Test(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, ShiftLeftTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ShiftLeftTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, ShiftRightTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ShiftRightTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, Atan2IntTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(Atan2IntTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, MaximumInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(MaximumInt8Test(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, MaximumWithIntScalarTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(MaximumWithIntScalarTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, MaximumWithUintScalarTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::UINT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(MaximumWithUintScalarTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, MinimumInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::INT8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(MinimumInt8Test(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LessTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LessTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LessEqualTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LessEqualTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, GreaterTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GreaterTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, GreaterEqualTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(GreaterEqualTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, EqualTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(EqualTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, NotEqualTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(NotEqualTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LogicalAndTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalAndTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LogicalAndInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalAndInt8Test(*exec_env, GetParam()));
}
TEST_P(ElementwiseTypedTest, LogicalOrTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalOrTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LogicalOrInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalOrInt8Test(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LogicalNotTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalNotTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LogicalNotInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalNotInt8Test(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LogicalXorTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalXorTest(*exec_env, GetParam()));
}

TEST_P(ElementwiseTypedTest, LogicalXorInt8Test) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(LogicalXorInt8Test(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTypedTestSuite, ElementwiseTypedTest,
    ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<ElementwiseTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class ElementwiseFloatTest : public DataTypeTest {};

TEST_P(ElementwiseFloatTest, AbsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AbsTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, CosTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(CosTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, CopyTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(CopyTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, EluTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(EluTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, ExpTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ExpTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, FloorTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FloorTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, FloorDivTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FloorDivTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, FloorModTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(FloorModTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, GeluTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(GeluTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, HardSwishTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(HardSwishTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, LogTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(LogTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, NegTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(NegTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, RoundTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(RoundTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, RsqrtTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(RsqrtTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SigmoidTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SigmoidTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SignTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SignTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SinTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SinTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SqrtTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SqrtTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SquareTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SquareTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, TanhTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TanhTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SubTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SubTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SquaredDiffTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(SquaredDiffTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, DivTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(DivTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, PowTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PowTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, PowWithScalarTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(PowWithScalarTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, AddTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, AddWithConstantBHWCTensorTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddWithConstantBHWCTensorTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, AddTiledTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(AddTiledTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, Atan2Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Atan2Test(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MaximumTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MaximumTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MaximumWithScalarTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MaximumWithScalarTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MaximumWithConstantLinearTensorTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      MaximumWithConstantLinearTensorTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MaximumWithConstantBHWCTensorTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      MaximumWithConstantBHWCTensorTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest,
       MaximumWithConstantBHWCTensorBroadcastChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MaximumWithConstantBHWCTensorBroadcastChannelsTest(
      *exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MinimumTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MinimumTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MinimumWithScalarTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MinimumWithScalarTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MulTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MulTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MulBroadcastHWTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MulBroadcastHWTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MulBroadcastChannelsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MulBroadcastChannelsTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, SubWithScalarAtFirstPositionTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      SubWithScalarAtFirstPositionTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, CosBroadcastTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(CosBroadcastTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MaximumScalarBroadcastInputTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MaximumScalarBroadcastInputTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MulLinearBroadcastInputTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MulLinearBroadcastInputTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MulBroadcastBothInputsTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MulBroadcastBothInputsTest(*exec_env, data_type(), storage()));
}

TEST_P(ElementwiseFloatTest, MishTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(MishTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseFloatTestSuite, ElementwiseFloatTest,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ElementwiseFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class Elementwise5DTest
    : public ::testing::TestWithParam<std::tuple<DataType, TensorStorageType>> {
 public:
  DataType data_type() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
};

TEST_P(Elementwise5DTest, Add5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(Add5DTest(*exec_env, data_type(), storage()));
}

TEST_P(Elementwise5DTest, OneInputWithBroadcast5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(OneInputWithBroadcast5DTest(*exec_env, data_type(), storage()));
}

TEST_P(Elementwise5DTest, WithBroadcast5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(WithBroadcast5DTest(*exec_env, data_type(), storage()));
}

TEST_P(Elementwise5DTest, WithBroadcast5DPaddedGridTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(WithBroadcast5DPaddedGridTest(*exec_env, data_type(), storage()));
}

TEST_P(Elementwise5DTest, TwoInputWithBroadcast5DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(TwoInputWithBroadcast5DTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    Elementwise5DTestSuite, Elementwise5DTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<Elementwise5DTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
