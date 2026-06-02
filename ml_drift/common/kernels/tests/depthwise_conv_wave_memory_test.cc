// Copyright 2025 The ML Drift Authors.
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

#include "ml_drift/common/kernels/depthwise_conv_wave_memory.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/depthwise_conv_wave_memory_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class DepthwiseConvWaveMemoryFloatTest : public FloatTest {
 public:
  void SetUp() override {
    if (!HardwareSupportsDepthwiseConvWaveMemory(exec_env->GetGpuInfo())) {
      GTEST_SKIP() << "Conv wave not supported on this device.";
    }
  }
};

TEST_P(DepthwiseConvWaveMemoryFloatTest, DepthwiseConvWaveMemory3x3Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvWaveMemory3x3Test(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvWaveMemoryFloatTest,
       DepthwiseConvWaveMemory3x3BatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      DepthwiseConvWaveMemory3x3BatchedTest(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvWaveMemoryFloatTest, DepthwiseConvWaveMemory5x5Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvWaveMemory5x5Test(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvWaveMemoryFloatTest, DepthwiseConvWaveMemory2x4Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvWaveMemory2x4Test(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvWaveMemoryFloatTest, DepthwiseConvWaveMemory7x3Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvWaveMemory7x3Test(*exec_env, precision(), storage()));
}

TEST_P(DepthwiseConvWaveMemoryFloatTest, DepthwiseConvWaveMemory7x7Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(DepthwiseConvWaveMemory7x7Test(*exec_env, precision(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    FloatTestSuite, DepthwiseConvWaveMemoryFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<DepthwiseConvWaveMemoryFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
