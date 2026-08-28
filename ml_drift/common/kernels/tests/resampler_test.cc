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
#include "ml_drift/common/kernels/tests/resampler_test_util.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class ResamplerTest
    : public Test,
      public WithParamInterface<std::tuple<DataType, TensorStorageType, BHWC>> {
};

TEST_P(ResamplerTest, ResamplerIdentityTest) {
  auto [data_type, storage, group_size] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ResamplerIdentityTest(*exec_env, data_type, storage, group_size));
}

INSTANTIATE_TEST_SUITE_P(
    ResamplerTestSuite, ResamplerTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes()),
            ValuesIn({BHWC(1, 2, 2, 1), BHWC(1, 3, 5, 3), BHWC(1, 6, 1, 7)})),
    [](const TestParamInfo<ResamplerTest::ParamType>& info) {
      // Test names can only contain alphanumeric
      // characters and underscores. ToString(storage_type)
      // = "TensorStorageType::x" We get rid of the colons
      // with StrReplaceAll.
      const BHWC shape = std::get<2>(info.param);
      const std::string shape_str =
          absl::StrCat(shape.b, "_", shape.h, "_", shape.w, "_", shape.c);
      const std::string name =
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_",
                       shape_str);
      return absl::StrReplaceAll(name, {{":", ""}});
    });

}  // namespace ml_drift
