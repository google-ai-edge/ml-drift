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
#include "ml_drift/common/kernels/tests/conv_wave_matrix_mali_f16_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

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
    if (!exec_env->GetGpuInfo().mali_info.IsMaliG2()) {
      GTEST_SKIP() << "Gpu is not Mali G2.";
    }
    if (!exec_env->GetGpuInfo().SupportsExtension("cl_arm_matrix_multiply")) {
      GTEST_SKIP() << "cl_arm_matrix_multiply not supported on this device.";
    }
  }
};

TEST_P(BaseTest, ConvWaveMatrixMaliF161x1BigTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliF161x1BigTest(*exec_env, GetParam()));
}

TEST_P(BaseTest, ConvWaveMatrixMaliF16BigTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliF16BigTest(*exec_env, GetParam()));
}

TEST_P(BaseTest, ConvWaveMatrixMaliF16ExternalWeightsBigTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliF16ExternalWeightsBigTest(*exec_env, GetParam()));
}

TEST_P(BaseTest, ConvWaveMatrixMaliF16ExternalBatchedWeightsBigTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliF16ExternalBatchedWeightsBigTest(*exec_env,
                                                               GetParam()));
}

TEST_P(BaseTest, ConvWaveMatrixMaliF16PackedGroupsTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(ConvWaveMatrixMaliF16PackedGroupsTest(*exec_env, GetParam()));
}

TEST_P(BaseTest, ConvWaveMatrixMaliF16RuntimeSrcEndChannelsTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(
      ConvWaveMatrixMaliF16RuntimeSrcEndChannelsTest(*exec_env, GetParam()));
}

TEST_P(BaseTest, ConvWaveMatrixMaliF16RuntimeDstEndChannelsTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  ABSL_ASSERT_OK(
      ConvWaveMatrixMaliF16RuntimeDstEndChannelsTest(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvWaveMatrixMaliF16BigTestSuite, BaseTest,
    ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
    [](const TestParamInfo<BaseTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

}  // namespace ml_drift
