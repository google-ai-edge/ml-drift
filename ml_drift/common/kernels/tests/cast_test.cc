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

#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/cast_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class CastTest : public Test,
                 public WithParamInterface<
                     std::tuple<TensorStorageType, TensorStorageType>> {
  // Convenience accessor methods
 protected:
  TensorStorageType src_storage() const { return std::get<0>(GetParam()); }
  TensorStorageType dst_storage() const { return std::get<1>(GetParam()); }
};

TEST_P(CastTest, Base) {
  if (!exec_env->IsStorageSupported(src_storage(), DataType::FLOAT32) ||
      !exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(dst_storage());
  }
  MLD_ASSERT_OK(CastBaseTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(CastTest, ToBool) {
  if (!exec_env->IsStorageSupported(src_storage(), DataType::FLOAT32) ||
      !exec_env->IsStorageSupported(dst_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(dst_storage());
  }
  MLD_ASSERT_OK(CastToBoolTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(CastTest, FromBool) {
  if (!exec_env->IsStorageSupported(src_storage(), DataType::BOOL) ||
      !exec_env->IsStorageSupported(dst_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(dst_storage());
  }
  MLD_ASSERT_OK(CastFromBoolTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(CastTest, ToBfloat) {
  if (!exec_env->IsStorageSupported(src_storage(), DataType::INT32) ||
      !exec_env->IsStorageSupported(dst_storage(), DataType::BFLOAT16)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(dst_storage());
  }
  MLD_ASSERT_OK(CastToBfloatTest(*exec_env, src_storage(), dst_storage()));
}

TEST_P(CastTest, FromBfloat) {
  if (!exec_env->IsStorageSupported(src_storage(), DataType::BFLOAT16) ||
      !exec_env->IsStorageSupported(dst_storage(), DataType::INT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(src_storage())
                 << " to " << ToString(dst_storage());
  }
  MLD_ASSERT_OK(CastFromBfloatTest(*exec_env, src_storage(), dst_storage()));
}

INSTANTIATE_TEST_SUITE_P(CastTestSuite, CastTest,
                         Combine(ValuesIn(GetTensorStoragesTypes()),
                                 ValuesIn(GetTensorStoragesTypes())),
                         [](const TestParamInfo<CastTest::ParamType>& info) {
                           // Test names can only contain alphanumeric
                           // characters and underscores. ToString(storage_type)
                           // = "TensorStorageType::x" We get rid of the colons
                           // with StrReplaceAll.
                           const std::string name = absl::StrCat(
                               ToString(std::get<0>(info.param)), "_",
                               ToString(std::get<1>(info.param)));
                           return absl::StrReplaceAll(name, {{":", ""}});
                         });

}  // namespace ml_drift
