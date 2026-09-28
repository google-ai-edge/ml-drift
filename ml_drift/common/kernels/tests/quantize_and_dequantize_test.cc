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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/quantize_and_dequantize_test_util.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class QuantizeAndDequantizeFloatTest : public FloatTest {};

TEST_P(QuantizeAndDequantizeFloatTest, QuantAndDequant_Dim2Bits8Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(QuantAndDequant_Dim2Bits8Test(*exec_env, precision(), storage()));
}

TEST_P(QuantizeAndDequantizeFloatTest,
       QuantAndDequant_Dim3Bits8_NegativeRangeTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(QuantAndDequant_Dim3Bits8_NegativeRangeTest(*exec_env, precision(),
                                                        storage()));
}

TEST_P(QuantizeAndDequantizeFloatTest, QuantAndDequant_Dim3Bits16Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(QuantAndDequant_Dim3Bits16Test(*exec_env, precision(), storage()));
}

TEST_P(QuantizeAndDequantizeFloatTest,
       QuantAndDequant_Dim2Bits16_NegativeRangeTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(QuantAndDequant_Dim2Bits16_NegativeRangeTest(*exec_env, precision(),
                                                         storage()));
}

INSTANTIATE_TEST_SUITE_P(
    QuantizeAndDequantizeFloatTestSuite, QuantizeAndDequantizeFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<QuantizeAndDequantizeFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class QuantizationTest
    : public testing::Test,
      public testing::WithParamInterface<
          std::tuple<DataType, TensorStorageType, TensorStorageType>> {
  // Convenience accessor methods
 protected:
  DataType data_type() const { return std::get<0>(GetParam()); }
  TensorStorageType float_storage() const { return std::get<1>(GetParam()); }
  TensorStorageType quant_storage() const { return std::get<2>(GetParam()); }
};

TEST_P(QuantizationTest, QuantizationUint8) {
  if (!exec_env->IsStorageSupported(float_storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(float_storage());
  }
  if (!exec_env->IsStorageSupported(quant_storage(), DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(DataType::kUint8)
                 << " storage type: " << ToString(quant_storage());
  }
  ABSL_ASSERT_OK(QuantizationUint8Test(*exec_env, float_storage(), data_type(),
                                  quant_storage()));
}

TEST_P(QuantizationTest, QuantizationInt8) {
  if (!exec_env->IsStorageSupported(float_storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(float_storage());
  }
  if (!exec_env->IsStorageSupported(quant_storage(), DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(DataType::kInt8)
                 << " storage type: " << ToString(quant_storage());
  }
  ABSL_ASSERT_OK(QuantizationUint8Test(*exec_env, float_storage(), data_type(),
                                  quant_storage()));
}

INSTANTIATE_TEST_SUITE_P(
    Suite, QuantizationTest,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<QuantizationTest::ParamType>& info) {
      const std::string name =
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param)));
      return absl::StrReplaceAll(name, {{":", ""}});
    });

}  // namespace ml_drift
