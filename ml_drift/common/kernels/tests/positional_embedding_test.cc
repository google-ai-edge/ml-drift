// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/positional_embedding.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

namespace ml_drift {
namespace {

absl::Status PositionalEmbeddingTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 4, 4);
  src_tensor.data = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                     -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};

  TensorFloat32 pos_tensor;
  pos_tensor.shape = BHWC(1, 1, 4, 1);
  pos_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePositionalEmbedding(env.GetGpuInfo(), op_def);
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      {src_tensor, pos_tensor},
      std::make_unique<GPUOperation>(std::move(operation)), BHWC(1, 1, 4, 4),
      &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {0.540302f, 2.0f, 3.0f, 4.0f, 3.583853f, 6.0f,
                                 7.0f, 8.0f, -1.989992f, -1.0f, -2.0f, -3.0f,
                                 -5.653644f, -5.0f, -6.0f, -7.0f}));
  return absl::OkStatus();
}
}  // namespace

class PositionalEmbeddingFloatTest : public DataTypeTest {};

TEST_P(PositionalEmbeddingFloatTest, PositionalEmbeddingTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_EXPECT_OK(PositionalEmbeddingTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    PositionalEmbeddingTestSuite, PositionalEmbeddingFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<PositionalEmbeddingFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
