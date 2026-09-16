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

#include "ml_drift/common/kernels/conv_apple_mpp.h"

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/conv_apple_mpp_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class ConvAppleMPPFloat16Test : public Test,
                                public WithParamInterface<TensorStorageType> {
 public:
  void SetUp() override {
    if (!SupportsConvAppleMPP(exec_env->GetGpuInfo())) {
      GTEST_SKIP() << "Not supported on this device.";
    }
  }
};

TEST_P(ConvAppleMPPFloat16Test, BigTest) {
  const BHWC src_shape(1, 1, 128, 128);
  const int dst_channels = 128;
  ABSL_ASSERT_OK(
      ConvAppleMPPBigTest(*exec_env, GetParam(), src_shape, dst_channels));
}

TEST_P(ConvAppleMPPFloat16Test, ExternalWeightsBigTest) {
  const BHWC src_shape(1, 1, 128, 128);
  const int dst_channels = 128;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWeightsTest(*exec_env, GetParam(), src_shape,
                                            dst_channels));
}

TEST_P(ConvAppleMPPFloat16Test, PackedGroupsTest) {
  const BHWC src_shape(1, 1, 383, 128);
  const int dst_channels = 64;
  ABSL_ASSERT_OK(ConvAppleMPPPackedGroupsTest(*exec_env, GetParam(), src_shape,
                                         dst_channels));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalWfloatTest) {
  const BHWC src_shape(1, 1, 32, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWfloatTest(*exec_env, GetParam(), src_shape,
                                           dst_channels));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalBatchedWfloatTest) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWfloatTest(*exec_env, GetParam(), src_shape,
                                           dst_channels,
                                           /*batched_weights=*/true));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalWi8Test) {
  const BHWC src_shape(1, 1, 32, 128);
  const int dst_channels = 64;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi8Test(*exec_env, GetParam(), src_shape,
                                        dst_channels));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalBatchedWi8Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi8Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/true));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalGroupedWi8Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi8Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/false,
                                        /*group_size=*/12));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalBatchedGroupedWi8Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi8Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/true,
                                        /*group_size=*/12));
}

TEST_P(ConvAppleMPPFloat16Test,
       ConvAppleMPPExternalBatchedWi8NonBatchedScalesTest) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi8Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/true,
                                        /*group_size=*/-1,
                                        /*scale_zp_batch=*/1));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalWi4Test) {
  const BHWC src_shape(1, 1, 32, 128);
  const int dst_channels = 64;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi4Test(*exec_env, GetParam(), src_shape,
                                        dst_channels));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalBatchedWi4Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi4Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/true));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalGroupedWi4Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi4Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/false,
                                        /*group_size=*/12));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalBatchedGroupedWi4Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi4Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/true,
                                        /*group_size=*/12));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalWi2Test) {
  const BHWC src_shape(1, 1, 32, 128);
  const int dst_channels = 64;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi2Test(*exec_env, GetParam(), src_shape,
                                        dst_channels));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalBatchedWi2Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi2Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/true));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalGroupedWi2Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi2Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/false,
                                        /*group_size=*/12));
}

TEST_P(ConvAppleMPPFloat16Test, ConvAppleMPPExternalBatchedGroupedWi2Test) {
  const BHWC src_shape(1, 6, 12, 32 * 3);
  const int dst_channels = 68;
  ABSL_ASSERT_OK(ConvAppleMPPExternalWi2Test(*exec_env, GetParam(), src_shape,
                                        dst_channels,
                                        /*batched_weights=*/true,
                                        /*group_size=*/12));
}

TEST_P(ConvAppleMPPFloat16Test, BatchedMatMulTest) {
  const BHWC left_shape(1, 12, 128, 32);
  const BHWC right_shape(1, 12, 32, 64);
  ABSL_ASSERT_OK(ConvAppleMPPBatchedMatMulTest(*exec_env, GetParam(), left_shape,
                                          right_shape));
}

TEST_P(ConvAppleMPPFloat16Test, RuntimeSrcEndChannelsTest) {
  ABSL_ASSERT_OK(ConvAppleMPPRuntimeSrcEndChannelsTest(*exec_env, GetParam()));
}

TEST_P(ConvAppleMPPFloat16Test, RuntimeDstEndChannelsTest) {
  ABSL_ASSERT_OK(ConvAppleMPPRuntimeDstEndChannelsTest(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ConvAppleMPPFloat16TestSuite, ConvAppleMPPFloat16Test,
    ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
    [](const TestParamInfo<ConvAppleMPPFloat16Test::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class BaseTest : public Test, public WithParamInterface<TensorStorageType> {
 public:
  void SetUp() override {
    if (!SupportsConvAppleMPP(exec_env->GetGpuInfo())) {
      GTEST_SKIP() << "Not supported on this device.";
    }
  }
};

TEST_P(BaseTest, ConvAppleMPPInt8BigTest) {
  const BHWC src_shape(1, 1, 128, 128);
  const int dst_channels = 128;
  ABSL_ASSERT_OK(
      ConvAppleMPPInt8BigTest(*exec_env, GetParam(), src_shape, dst_channels));
}

TEST_P(BaseTest, ConvAppleMPPInt8ExternalWeightsBigTest) {
  const BHWC src_shape(1, 1, 128, 128);
  const int dst_channels = 128;
  ABSL_ASSERT_OK(ConvAppleMPPInt8ExternalWeightsBigTest(*exec_env, GetParam(),
                                                   src_shape, dst_channels));
}

TEST_P(BaseTest, ConvAppleMPPInt8ExternalBatchedWi4Test) {
  const BHWC src_shape(1, 6, 128, 128);
  const int dst_channels = 128;
  ABSL_ASSERT_OK(ConvAppleMPPInt8ExternalBatchedWi4Test(*exec_env, GetParam(),
                                                   src_shape, dst_channels));
}

INSTANTIATE_TEST_SUITE_P(
    ConvAppleMPPInt8BigTestSuite, BaseTest,
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
    if (!SupportsConvAppleMPP(exec_env->GetGpuInfo())) {
      GTEST_SKIP() << "Not supported on this device.";
    }
  }
};

TEST_P(SrcQuantizationTest, ConvAppleMPPInt8WithSrcQuantizationBigTest) {
  const BHWC src_shape(1, 3, 39, 128);
  const int dst_channels = 128;
  const auto& [float_type, int_storage, float_storage] = GetParam();
  ABSL_ASSERT_OK(ConvAppleMPPInt8WithSrcQuantizationBigTest(
      *exec_env, int_storage, float_storage, float_type, src_shape,
      dst_channels));
}

INSTANTIATE_TEST_SUITE_P(
    ConvAppleMPPInt8WithSrcQuantizationTestSuite, SrcQuantizationTest,
    ::testing::Combine(
        ValuesIn({DataType::FLOAT16, DataType::FLOAT32}),
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
