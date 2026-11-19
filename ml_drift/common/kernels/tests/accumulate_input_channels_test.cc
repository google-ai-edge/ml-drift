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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/accumulate_input_channels_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class AccumulateInputChannelsTests
    : public testing::Test,
      public testing::WithParamInterface<std::tuple<TensorStorageType, OHWI>> {
};

TEST_P(AccumulateInputChannelsTests, AccumulateInputChannelsInt8ToInt32Test) {
  auto& [storage, weights_shape] = GetParam();
  if (!exec_env->IsStorageSupported(storage, DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(AccumulateInputChannelsInt8ToInt32Test(*exec_env, storage,
                                                   weights_shape));
}

TEST_P(AccumulateInputChannelsTests, AccumulateInputChannelsInt4ToInt32Test) {
  auto& [storage, weights_shape] = GetParam();
  // The input int4 data is stored in int8 (two int4 elements per int8 element).
  if (!exec_env->IsStorageSupported(storage, DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(AccumulateInputChannelsInt4ToInt32Test(*exec_env, storage,
                                                   weights_shape));
}

TEST_P(AccumulateInputChannelsTests, AccumulateInputChannelsInt2ToInt32Test) {
  auto& [storage, weights_shape] = GetParam();
  // The input int2 data is stored in int8 (four int2 elements per int8
  // element).
  if (!exec_env->IsStorageSupported(storage, DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(AccumulateInputChannelsInt2ToInt32Test(*exec_env, storage,
                                                   weights_shape));
}

INSTANTIATE_TEST_SUITE_P(
    AccumulateInputChannelsTestSuite, AccumulateInputChannelsTests,
    Combine(ValuesIn({TensorStorageType::kTexture2D,
                      TensorStorageType::kBuffer}),
            ValuesIn({OHWI(4, 1, 1, 8), OHWI(16, 1, 1, 16)})),
    [](const TestParamInfo<AccumulateInputChannelsTests::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat("storage_", ToString(std::get<0>(info.param)),
                       "_weights_shape_",
                       GetShapeName(std::get<1>(info.param))),
          {{":", ""}});
    });

}  // namespace ml_drift
