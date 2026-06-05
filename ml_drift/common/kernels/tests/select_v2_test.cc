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
#include "ml_drift/common/default/status_matchers.h"
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
  if (!exec_env->IsStorageSupported(storage(), DataType::FLOAT32) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(IfTest<DataType::FLOAT32>(*exec_env, data_type(), storage(),
                                      cond_storage()));
}

TEST_P(SelectTest, IfBool) {
  if (!exec_env->IsStorageSupported(storage(), DataType::BOOL) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(IfTest<DataType::BOOL>(*exec_env, data_type(), storage(),
                                   cond_storage()));
}

TEST_P(SelectTest, IfInt8) {
  if (!exec_env->IsStorageSupported(storage(), DataType::INT8) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(IfTest<DataType::INT8>(*exec_env, data_type(), storage(),
                                   cond_storage()));
}

TEST_P(SelectTest, IfInt16) {
  if (!exec_env->IsStorageSupported(storage(), DataType::INT16) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(IfTest<DataType::INT16>(*exec_env, data_type(), storage(),
                                    cond_storage()));
}

TEST_P(SelectTest, IfInt32) {
  if (!exec_env->IsStorageSupported(storage(), DataType::INT32) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(IfTest<DataType::INT32>(*exec_env, data_type(), storage(),
                                    cond_storage()));
}

TEST_P(SelectTest, SelectV2F32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2Test<DataType::FLOAT32>(*exec_env, data_type(), storage(),
                                            cond_storage()));
}

TEST_P(SelectTest, SelectV2Bool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2Test<DataType::BOOL>(*exec_env, data_type(), storage(),
                                         cond_storage()));
}

TEST_P(SelectTest, SelectV2Scalar4DF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2Scalar4DTest<DataType::FLOAT32>(*exec_env, data_type(),
                                                    storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2Scalar4DBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2Scalar4DTest<DataType::BOOL>(*exec_env, data_type(),
                                                 storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2TrueValueF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2TrueValueTest<DataType::FLOAT32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2TrueValueBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2TrueValueTest<DataType::BOOL>(*exec_env, data_type(),
                                                  storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2FalseValueF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2FalseValueTest<DataType::FLOAT32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2FalseValueBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2FalseValueTest<DataType::BOOL>(*exec_env, data_type(),
                                                   storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BatchF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BatchTest<DataType::FLOAT32>(*exec_env, data_type(),
                                                 storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BatchBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BatchTest<DataType::BOOL>(*exec_env, data_type(), storage(),
                                              cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2ChannelsTest<DataType::FLOAT32>(*exec_env, data_type(),
                                                    storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2ChannelsTest<DataType::BOOL>(*exec_env, data_type(),
                                                 storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBatchF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2ChannelsBatchTest<DataType::FLOAT32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBatchBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2ChannelsBatchTest<DataType::BOOL>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastTrueF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BroadcastTrueTest<DataType::FLOAT32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastTrueBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BroadcastTrueTest<DataType::BOOL>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastFalseF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BroadcastFalseTest<DataType::FLOAT32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastFalseBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BroadcastFalseTest<DataType::BOOL>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastBothF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BroadcastBothTest<DataType::FLOAT32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2BroadcastBothBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2BroadcastBothTest<DataType::BOOL>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBroadcastFalseF32) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2ChannelsBroadcastFalseTest<DataType::FLOAT32>(
      *exec_env, data_type(), storage(), cond_storage()));
}

TEST_P(SelectTest, SelectV2ChannelsBroadcastFalseBool) {
  if (!exec_env->IsStorageSupported(storage(), data_type()) ||
      !exec_env->IsStorageSupported(cond_storage(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage())
                 << " to " << ToString(cond_storage());
  }
  MLD_ASSERT_OK(SelectV2ChannelsBroadcastFalseTest<DataType::BOOL>(
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
