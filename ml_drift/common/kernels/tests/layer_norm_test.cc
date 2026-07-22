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
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/status.h"
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
  Node* layer_norm = model.NewNode();
  layer_norm->operation.type = ToString(ml_drift::OperationType::LAYER_NORM);
  ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::FLOAT32> bias_tensor;
  bias_tensor.shape = Linear(4);
  bias_tensor.data.resize(bias_tensor.shape.DimensionsProduct());
  for (int i = 0; i < bias_tensor.shape.DimensionsProduct(); ++i) {
    bias_tensor.data[i] = 1;
  }
  ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::FLOAT32>
      scale_tensor;
  scale_tensor.shape = Linear(4);
  scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
  for (int i = 0; i < scale_tensor.shape.DimensionsProduct(); ++i) {
    scale_tensor.data[i] = 0.1;
  }
  LayerNormAttributes attr;
  attr.bias = std::move(bias_tensor);
  attr.scale = std::move(scale_tensor);
  attr.epsilon = 1e-6;
  layer_norm->operation.attributes = std::move(attr);

  Value* input = model.NewValue();
  Value* output = model.NewValue();
  TensorRef<BHWC> input_ref;
  input_ref.type = DataType::FLOAT32;
  input_ref.ref = 1;
  input_ref.shape = BHWC(num_batch, 1, 1, 4);

  TensorRef<BHWC> output_ref;
  output_ref.type = DataType::FLOAT32;
  output_ref.ref = 2;
  output_ref.shape = BHWC(num_batch, 1, 1, 4);

  input->tensor = input_ref;
  output->tensor = output_ref;

  model.AddConsumer(layer_norm->id, input->id);
  model.SetProducer(layer_norm->id, output->id);

  ml_drift::TensorFloat32 input_tensor;
  input_tensor.shape = ml_drift::BHWC(num_batch, 1, 1, 4);
  for (int i = 0; i < num_batch; i++) {
    input_tensor.data.insert(input_tensor.data.end(), {1, 2, 3, 4});
  }
  TensorFloat32 inference_result;
  {
    CreateGpuModelInfo create_info;
    create_info.precision = ml_drift::CalculationsPrecision::F32;
    create_info.storage_type = storage;
    create_info.hints.Add(ModelHints::kAllowSpecialKernels);
    create_info.hints.Add(ml_drift::ModelHints::kFastTuning);
    GpuModel gpu_model;
    MLD_EXPECT_OK(GraphToGpuModel(model, create_info, exec_env->GetGpuInfo(),
                              &gpu_model));
    MLD_EXPECT_OK(exec_env->ExecuteGpuModel(
        {input_tensor}, std::vector<TensorFloat32*>{&inference_result},
        &gpu_model));
  }

  TensorFloat32 expected;
  expected.shape = ml_drift::BHWC(num_batch, 1, 1, 4);
  for (int i = 0; i < num_batch; i++) {
    expected.data.insert(expected.data.end(),
                          {0.865836, 0.955279, 1.04472, 1.13416});
  }
  EXPECT_THAT(inference_result, TensorEq(FloatNear(1e-5), expected));
  return absl::OkStatus();
}
}  // namespace


class LayerNormTest
    : public Test,
      public WithParamInterface<std::tuple<int, TensorStorageType>> {};

TEST_P(LayerNormTest, GroupNormMultiBatches) {
  auto [num_batch, storage] = GetParam();
  if (!exec_env->IsStorageSupported(storage, DataType::FLOAT32)) {
    GTEST_SKIP() << "Unsupported storage type: "
                 << ToString(std::get<1>(GetParam()));
  }
  MLD_EXPECT_OK(GroupNorm(exec_env, num_batch, storage));
}

INSTANTIATE_TEST_SUITE_P(
    LayerNormTestSuite, LayerNormTest,
    Combine(ValuesIn({1, 2}), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<LayerNormTest::ParamType>& info) {
      const int batch_size = std::get<0>(info.param);
      const TensorStorageType storage = std::get<1>(info.param);
      return absl::StrReplaceAll(
          absl::StrCat("BatchSize", batch_size, "_", ToString(storage)),
          {{":", ""}});
    });

}  // namespace ml_drift
