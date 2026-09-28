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
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/tensor_matcher.h"

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

namespace ml_drift {
namespace {
absl::Status GroupNorm(TestExecutionEnvironment* env, int num_batch,
                       TensorStorageType storage) {
  GraphFloat32 model;
  Node* group_norm = model.NewNode();
  group_norm->operation.type = ToString(ml_drift::OperationType::kGroupNorm);
  ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::kFloat32> beta_tensor;
  beta_tensor.shape = Linear(32);
  beta_tensor.data.resize(beta_tensor.shape.DimensionsProduct());
  for (int i = 0; i < beta_tensor.shape.DimensionsProduct(); ++i) {
    beta_tensor.data[i] = i % 10;
  }
  ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::kFloat32> gamma_tensor;
  gamma_tensor.shape = Linear(32);
  gamma_tensor.data.resize(gamma_tensor.shape.DimensionsProduct());
  for (int i = 0; i < gamma_tensor.shape.DimensionsProduct(); ++i) {
    gamma_tensor.data[i] = i % 10;
  }
  GroupNormAttributes attr;
  attr.beta = std::move(beta_tensor);
  attr.gamma = std::move(gamma_tensor);
  attr.groups = 32;
  attr.epsilon = 1e-6;
  group_norm->operation.attributes = std::move(attr);

  Value* input = model.NewValue();
  Value* output = model.NewValue();

  TensorRef<BHWC> input_ref;
  input_ref.type = DataType::kFloat32;
  input_ref.ref = 1;
  input_ref.shape = BHWC(num_batch, 1, 1, 32);

  TensorRef<BHWC> output_ref;
  output_ref.type = DataType::kFloat32;
  output_ref.ref = 2;
  output_ref.shape = BHWC(num_batch, 1, 1, 32);

  input->tensor = input_ref;
  output->tensor = output_ref;

  model.AddConsumer(group_norm->id, input->id);
  model.SetProducer(group_norm->id, output->id);

  ml_drift::TensorFloat32 input_tensor;
  input_tensor.shape = ml_drift::BHWC(num_batch, 1, 1, 32);
  input_tensor.data.resize(input_tensor.shape.DimensionsProduct());
  int batch_size = input_tensor.shape.DimensionsProduct() / num_batch;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < num_batch; ++j) {
      input_tensor.data[i + batch_size * j] = i % 10;
    }
  }

  TensorFloat32 inference_result;
  {
    CreateGpuModelInfo create_info;
    create_info.precision = ml_drift::CalculationsPrecision::kF32;
    create_info.storage_type = storage;
    create_info.hints.Add(ModelHints::kAllowSpecialKernels);
    create_info.hints.Add(ml_drift::ModelHints::kFastTuning);
    GpuModel gpu_model;
    ABSL_EXPECT_OK(GraphToGpuModel(model, create_info, exec_env->GetGpuInfo(),
                              &gpu_model));
    ABSL_EXPECT_OK(exec_env->ExecuteGpuModel(
        {input_tensor}, std::vector<TensorFloat32*>{&inference_result},
        &gpu_model));
  }

  TensorFloat32 expected;
  expected.shape = ml_drift::BHWC(num_batch, 1, 1, 32);
  expected.data.resize(expected.shape.DimensionsProduct());
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_batch; ++j) {
      expected.data[i + batch_size * j] = i % 10;
    }
  }
  EXPECT_THAT(inference_result, TensorEq(FloatNear(1e-5), expected));
  return absl::OkStatus();
}

}  // namespace

class GroupNormTest
    : public Test,
      public WithParamInterface<std::tuple<int, TensorStorageType>> {};

TEST_P(GroupNormTest, GroupNormMultiBatches) {
  auto [num_batch, storage] = GetParam();
  if (!exec_env->IsStorageSupported(storage, DataType::kFloat32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ToString(std::get<1>(GetParam()));
  }
  ABSL_EXPECT_OK(GroupNorm(exec_env, num_batch, storage));
}

INSTANTIATE_TEST_SUITE_P(
    GroupNormTestSuite, GroupNormTest,
    Combine(ValuesIn({1, 2}), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<GroupNormTest::ParamType>& info) {
      const int batch_size = std::get<0>(info.param);
      const TensorStorageType storage = std::get<1>(info.param);
      return absl::StrReplaceAll(
          absl::StrCat("BatchSize", batch_size, "_", ToString(storage)),
          {{":", ""}});
    });

}  // namespace ml_drift
