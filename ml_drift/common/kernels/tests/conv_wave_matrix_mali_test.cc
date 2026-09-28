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

#include "ml_drift/common/kernels/conv_wave_matrix_mali.h"

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/conv_wave_matrix_mali_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class BaseTest : public Test, public WithParamInterface<TensorStorageType> {
 public:
  void SetUp() override {
    if (!exec_env->GetGpuInfo().IsMali()) {
      GTEST_SKIP() << "Gpu is not Mali.";
    }
  }
};

TEST_P(BaseTest, ConvWaveMatrixMaliInt8BigTest) {
  const BHWC src_shape(1, 2, 32, 128);
  if (!SupportsConvWaveMatrixMaliInt8(exec_env->GetGpuInfo(), src_shape)) {
    GTEST_SKIP() << "ConvWaveMatrixMaliInt8 not supported on this device.";
  }
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliInt8BigTest(*exec_env, GetParam(), src_shape));
}

TEST_P(BaseTest, ConvWaveMatrixMaliInt8ExternalWeightsBigTest) {
  const BHWC src_shape(1, 3, 40, 128);
  if (!SupportsConvWaveMatrixMaliInt8(exec_env->GetGpuInfo(), src_shape)) {
    GTEST_SKIP() << "ConvWaveMatrixMaliInt8 not supported on this device.";
  }
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliInt8ExternalWeightsBigTest(*exec_env, GetParam(),
                                                         src_shape));
}

INSTANTIATE_TEST_SUITE_P(
    ConvWaveMatrixMaliInt8BigTestSuite, BaseTest,
    ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
    [](const TestParamInfo<BaseTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class SrcQuantizationTest
    : public Test,
      public WithParamInterface<
          std::tuple<DataType, TensorStorageType, TensorStorageType>> {
 public:
  void SetUp() override {
    if (!exec_env->GetGpuInfo().IsMali()) {
      GTEST_SKIP() << "Gpu is not Mali.";
    }
  }
};

TEST_P(SrcQuantizationTest, ConvWaveMatrixMaliInt8WithSrcQuantizationBig) {
  const BHWC src_shape(1, 3, 39, 128);
  if (!SupportsConvWaveMatrixMaliInt8(exec_env->GetGpuInfo(), src_shape)) {
    GTEST_SKIP() << "ConvWaveMatrixMaliInt8 not supported on this device.";
  }
  const auto& [float_type, int_storage, float_storage] = GetParam();
  if (!exec_env->IsStorageSupported(int_storage, DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported int storage: " << ToString(int_storage);
  }
  if (!exec_env->IsStorageSupported(float_storage, float_type)) {
    GTEST_SKIP() << "Unsupported float storage: " << ToString(float_storage);
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliInt8WithSrcQuantizationBigTest(
      *exec_env, int_storage, float_storage, float_type, src_shape));
}

TEST_P(SrcQuantizationTest,
       ConvWaveMatrixMaliInt8WithSrcQuantizationBigBatched) {
  const BHWC src_shape(3, 5, 17, 32);
  if (!SupportsConvWaveMatrixMaliInt8(exec_env->GetGpuInfo(), src_shape)) {
    GTEST_SKIP() << "ConvWaveMatrixMaliInt8 not supported on this device.";
  }
  const auto& [float_type, int_storage, float_storage] = GetParam();
  if (!exec_env->IsStorageSupported(int_storage, DataType::kInt32)) {
    GTEST_SKIP() << "Unsupported int storage: " << ToString(int_storage);
  }
  if (!exec_env->IsStorageSupported(float_storage, float_type)) {
    GTEST_SKIP() << "Unsupported float storage: " << ToString(float_storage);
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliInt8WithSrcQuantizationBigTest(
      *exec_env, int_storage, float_storage, float_type, src_shape));
}

INSTANTIATE_TEST_SUITE_P(
    ConvWaveMatrixMaliSrcQuantizationTestSuite, SrcQuantizationTest,
    Combine(ValuesIn({DataType::kFloat16, DataType::kFloat32}),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<SrcQuantizationTest::ParamType>& info) {
      return absl::StrCat(
          ToString(std::get<0>(info.param)), "_int",
          absl::StrReplaceAll(ToString(std::get<1>(info.param)), {{":", ""}}),
          "_float",
          absl::StrReplaceAll(ToString(std::get<2>(info.param)), {{":", ""}}));
    });

}  // namespace ml_drift
