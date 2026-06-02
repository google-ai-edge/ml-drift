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

#include "ml_drift/common/kernels/conv_wave_memory.h"

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/conv_wave_memory_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class ConvWaveMemoryFloatTest : public FloatTest {
 public:
  void SetUp() override {
    if (!IsConvWaveMemorySupported(exec_env->GetGpuInfo())) {
      GTEST_SKIP() << "Conv wave not supported on this device.";
    }
  }
};

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst4SlicesTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(1, 7, 4, 7), 13));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst4SlicesBatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(5, 7, 4, 7), 13));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst9SlicesTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(1, 7, 4, 7), 33));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst9SlicesBatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(7, 7, 4, 7), 33));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst6SlicesTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(1, 5, 3, 7), 23));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst6SlicesBatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(11, 5, 3, 7), 23));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst3SlicesTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(1, 5, 3, 15), 9));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryDst3SlicesBatchedTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryTest(*exec_env, precision(), storage(),
                               BHWC(2, 5, 3, 15), 9));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryGroupedX3Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryGroupedX3Test(*exec_env, precision(), storage()));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryGroupedX7Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryGroupedX7Test(*exec_env, precision(), storage()));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryExternalWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      ConvWaveMemoryExternalWeightsTest(*exec_env, precision(), storage()));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryExternalBatchedWeightsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryExternalBatchedWeightsTest(*exec_env, precision(),
                                                     storage()));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryWinograd4x4To6x6Test) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(
      ConvWaveMemoryWinograd4x4To6x6Test(*exec_env, precision(), storage()));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryBatchedMatMulTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(ConvWaveMemoryBatchedMatMulTest(*exec_env, precision(), storage()));
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryRuntimeSrcEndChannelsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMemoryRuntimeSrcEndChannelsTest(*exec_env, precision(),
                                                        storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
}

TEST_P(ConvWaveMemoryFloatTest, ConvWaveMemoryRuntimeDstEndChannelsTest) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto status = ConvWaveMemoryRuntimeDstEndChannelsTest(*exec_env, precision(),
                                                        storage());
  if (!status.ok() &&
      absl::StrContains(status.message(), exec_env->SkipTestMessage())) {
    GTEST_SKIP() << status.message();
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConvWaveMemoryFloatTestSuite, ConvWaveMemoryFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ConvWaveMemoryFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

class IntTest : public Test,
                public WithParamInterface<
                    std::tuple<TensorStorageType, TensorStorageType>> {
 public:
  void SetUp() override {
    if (!IsConvWaveMemorySupported(exec_env->GetGpuInfo())) {
      GTEST_SKIP() << "Conv wave not supported on this device.";
    }
  }

 protected:
  TensorStorageType src_storage() const { return std::get<0>(GetParam()); }
  TensorStorageType dst_storage() const { return std::get<1>(GetParam()); }
};

TEST_P(IntTest, ConvWaveMemoryInt8Test) {
  // Check dst storage. Check src storage w/i helper.
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported dst storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  MLD_ASSERT_OK(ConvWaveMemoryInt8Test(*exec_env, src_storage(), dst_storage()));
}

TEST_P(IntTest, ConvWaveMemoryInt8ExternalWeightsTest) {
  // Check dst storage. Check src storage w/i helper.
  if (!exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported dst storage type: " << ToString(dst_storage())
                 << " data type: " << ToString(DataType::INT32);
  }
  MLD_ASSERT_OK(ConvWaveMemoryInt8ExternalWeightsTest(*exec_env, src_storage(),
                                                  dst_storage()));
}

INSTANTIATE_TEST_SUITE_P(IntTestSuite, IntTest,
                         Combine(ValuesIn(GetTensorStoragesTypes()),
                                 ValuesIn(GetTensorStoragesTypes())),
                         [](const TestParamInfo<IntTest::ParamType>& info) {
                           return absl::StrReplaceAll(
                               absl::StrCat(ToString(std::get<0>(info.param)),
                                            "_",
                                            ToString(std::get<1>(info.param))),
                               {{":", "_"}});
                         });

class SrcQuantizationTest
    : public Test,
      public WithParamInterface<
          std::tuple<DataType, TensorStorageType, TensorStorageType>> {
 public:
  void SetUp() override {
    if (!IsConvWaveMemorySupported(exec_env->GetGpuInfo())) {
      GTEST_SKIP() << "Conv wave not supported on this device.";
    }
  }
};

TEST_P(SrcQuantizationTest, ConvWaveMemoryInt8WithSrcQuantizationBig) {
  const auto& [float_type, int_storage, float_storage] = GetParam();
  if (!exec_env->IsStorageSupported(int_storage, DataType::INT32)) {
    GTEST_SKIP() << "Unsupported int storage: " << ToString(int_storage);
  }
  if (!exec_env->IsStorageSupported(float_storage, float_type)) {
    GTEST_SKIP() << "Unsupported float storage: " << ToString(float_storage);
  }
  MLD_ASSERT_OK(ConvWaveMemoryInt8WithSrcQuantizationTest(
      *exec_env, int_storage, float_storage, float_type));
}

INSTANTIATE_TEST_SUITE_P(
    ConvWaveMemorySrcQuantizationTestSuite, SrcQuantizationTest,
    Combine(ValuesIn({DataType::FLOAT16, DataType::FLOAT32}),
            ValuesIn(GetTensorStoragesTypes()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<SrcQuantizationTest::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_int",
                       ToString(std::get<1>(info.param)), "_float",
                       ToString(std::get<2>(info.param))),
          {{":", ""}});
    });

}  // namespace ml_drift
