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
#include "ml_drift/common/kernels/tests/select_v2_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

class SelectTest
    : public Test,
      public WithParamInterface<
          std::tuple<DataType, TensorStorageType, TensorStorageType>> {
  // Convenience accessor methods
 protected:
  DataType data_type() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
  TensorStorageType cond_storage() const { return std::get<2>(GetParam()); }
};

TEST_P(SelectTest, IfF32) {
  if (!exec_env->IsStorageSupported(storage(), DataType::kFloat32) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(IfTest<DataType::kFloat32>(*exec_env, data_type(), storage(),
                                       cond_storage()));
}

TEST_P(SelectTest, IfBool) {
  if (!exec_env->IsStorageSupported(storage(), DataType::kBool) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(IfTest<DataType::kBool>(*exec_env, data_type(), storage(),
                                    cond_storage()));
}

TEST_P(SelectTest, IfInt8) {
  if (!exec_env->IsStorageSupported(storage(), DataType::kInt8) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(IfTest<DataType::kInt8>(*exec_env, data_type(), storage(),
                                    cond_storage()));
}

TEST_P(SelectTest, IfInt16) {
  if (!exec_env->IsStorageSupported(storage(), DataType::kInt16) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(IfTest<DataType::kInt16>(*exec_env, data_type(), storage(),
                                     cond_storage()));
}

TEST_P(SelectTest, IfInt32) {
  if (!exec_env->IsStorageSupported(storage(), DataType::kInt32) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(IfTest<DataType::kInt32>(*exec_env, data_type(), storage(),
                                     cond_storage()));
}

TEST_P(SelectTest, SelectV2F32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2Test<DataType::kFloat32>(*exec_env, data_type(), storage(),
                                             cond_storage()));
}

TEST_P(SelectTest, SelectV2Bool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2Test<DataType::kBool>(*exec_env, data_type(), storage(),
                                          cond_storage()));
}

TEST_P(SelectTest, SelectV2Scalar4DF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2Scalar4DTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2Scalar4DBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2Scalar4DTest<DataType::kBool>(*exec_env, data_type(),
                                                  storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2TrueValueF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2TrueValueTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2TrueValueBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2TrueValueTest<DataType::kBool>(*exec_env, data_type(),
                                                   storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2FalseValueF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2FalseValueTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2FalseValueBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2FalseValueTest<DataType::kBool>(*exec_env, data_type(),
                                                    storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BatchF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BatchTest<DataType::kFloat32>(*exec_env, data_type(),
                                                  storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BatchBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BatchTest<DataType::kBool>(*exec_env, data_type(),
                                               storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2ChannelsTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2ChannelsTest<DataType::kBool>(*exec_env, data_type(),
                                                  storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBatchF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2ChannelsBatchTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBatchBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2ChannelsBatchTest<DataType::kBool>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastTrueF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BroadcastTrueTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastTrueBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BroadcastTrueTest<DataType::kBool>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastFalseF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BroadcastFalseTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastFalseBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BroadcastFalseTest<DataType::kBool>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastBothF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BroadcastBothTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastBothBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2BroadcastBothTest<DataType::kBool>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBroadcastFalseF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2ChannelsBroadcastFalseTest<DataType::kFloat32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBroadcastFalseBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::kBool)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  ABSL_ASSERT_OK(SelectV2ChannelsBroadcastFalseTest<DataType::kBool>(
      *exec_env, data_type(), storage(), cond_storage()));
}

INSTANTIATE_TEST_SUITE_P(SelectTestSuite, SelectTest,
                         Combine(ValuesIn(GetFloatTypes()),
                                 ValuesIn(GetTensorStoragesTypes()),
                                 ValuesIn(GetTensorStoragesTypes())),
                         [](const TestParamInfo<SelectTest::ParamType>& info) {
                           // Test names can only contain alphanumeric
                           // characters and underscores. ToString(storage_type)
                           // = "TensorStorageType::x" We get rid of the colons
                           // with StrReplaceAll.
                           const std::string name = absl::StrCat(
                               ToString(std::get<0>(info.param)), "_",
                               ToString(std::get<1>(info.param)), "_",
                               ToString(std::get<2>(info.param)));
                           return absl::StrReplaceAll(name, {{":", ""}});
                         });

}  // namespace ml_drift
