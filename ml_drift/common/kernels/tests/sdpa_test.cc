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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
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
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/tensor_matcher.h"

namespace ml_drift {

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

namespace {

absl::StatusOr<TensorFloat32> RunModel(
    GraphFloat32& model, const ml_drift::TensorFloat32& input_tensor_q,
    const ml_drift::TensorFloat32& input_tensor_k,
    const ml_drift::TensorFloat32& input_tensor_v,
    std::optional<ml_drift::TensorFloat32> input_tensor_mask,
    TensorStorageType storage_type, TestExecutionEnvironment& exec_env) {
  TensorFloat32 inference_result;
  CreateGpuModelInfo create_info;
  create_info.precision = ml_drift::CalculationsPrecision::F32;
  create_info.storage_type = storage_type;
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  create_info.hints.Add(ml_drift::ModelHints::kFastTuning);
  GpuModel gpu_model;
  RETURN_IF_ERROR(
      GraphToGpuModel(model, create_info, exec_env.GetGpuInfo(), &gpu_model));
  std::vector<TensorFloat32> src_cpu_tensors = {input_tensor_q, input_tensor_k,
                                                input_tensor_v};
  if (input_tensor_mask.has_value()) {
    src_cpu_tensors.push_back(input_tensor_mask.value());
  }
  RETURN_IF_ERROR(exec_env.ExecuteGpuModel(
      src_cpu_tensors, std::vector<TensorFloat32*>{&inference_result},
      &gpu_model));
  return inference_result;
}

}  // namespace

using SdpaTest = TestWithParam<TensorStorageType>;

TEST_P(SdpaTest, SdpaWithoutMask) {
  GraphFloat32 model;
  Node* sdpa = model.NewNode();
  sdpa->operation.type =
      ToString(ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION);
  ScaledDotProductAttentionAttributes attr;
  attr.scale = 1;
  sdpa->operation.attributes = std::move(attr);
  Value* query = model.NewValue();
  Value* key = model.NewValue();
  Value* value = model.NewValue();
  Value* output = model.NewValue();

  TensorRef<BHWC> query_ref;
  query_ref.type = DataType::FLOAT32;
  query_ref.ref = 0;
  query_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> key_ref;
  key_ref.type = DataType::FLOAT32;
  key_ref.ref = 1;
  key_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> value_ref;
  value_ref.type = DataType::FLOAT32;
  value_ref.ref = 2;
  value_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> output_ref;
  output_ref.type = DataType::FLOAT32;
  output_ref.ref = 3;
  output_ref.shape = BHWC(1, 4, 1, 4);

  query->tensor = query_ref;
  key->tensor = key_ref;
  value->tensor = value_ref;
  output->tensor = output_ref;

  model.AddConsumer(sdpa->id, query->id);
  model.AddConsumer(sdpa->id, key->id);
  model.AddConsumer(sdpa->id, value->id);
  model.SetProducer(sdpa->id, output->id);

  ml_drift::TensorFloat32 input_tensor_q;
  input_tensor_q.shape = query_ref.shape;
  input_tensor_q.data.resize(input_tensor_q.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_q.shape.DimensionsProduct(); ++i) {
    input_tensor_q.data[i] = i;
  }

  ml_drift::TensorFloat32 input_tensor_k;
  input_tensor_k.shape = key_ref.shape;
  input_tensor_k.data.resize(input_tensor_k.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_k.shape.DimensionsProduct(); ++i) {
    input_tensor_k.data[i] = i % 10;
  }

  ml_drift::TensorFloat32 input_tensor_v;
  input_tensor_v.shape = value_ref.shape;
  input_tensor_v.data.resize(input_tensor_v.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_v.shape.DimensionsProduct(); ++i) {
    input_tensor_v.data[i] = i % 10;
  }

  TensorFloat32 inference_result;
  MLD_ASSERT_OK_AND_ASSIGN(
      inference_result,
      RunModel(model, input_tensor_q, input_tensor_k, input_tensor_v,
               /*input_tensor_mask=*/{}, GetParam(), *exec_env));

  TensorFloat32 expected;
  expected.shape = ml_drift::BHWC(1, 4, 1, 4);
  expected.data.resize(expected.shape.DimensionsProduct());
  for (int i = 0; i < expected.shape.DimensionsProduct() / 4; i++) {
    expected.data[i * 4] = 4;
    expected.data[i * 4 + 1] = 5;
    expected.data[i * 4 + 2] = 6;
    expected.data[i * 4 + 3] = 7;
  }
  EXPECT_THAT(inference_result, TensorEq(testing::FloatNear(2e-5), expected));
}

TEST_P(SdpaTest, SdpaWithMaskNoScale) {
  GraphFloat32 model;
  Node* sdpa = model.NewNode();
  sdpa->operation.type =
      ToString(ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION);
  ScaledDotProductAttentionAttributes attr;
  sdpa->operation.attributes = std::move(attr);
  Value* query = model.NewValue();
  Value* key = model.NewValue();
  Value* value = model.NewValue();
  Value* mask = model.NewValue();
  Value* output = model.NewValue();

  TensorRef<BHWC> query_ref;
  query_ref.type = DataType::FLOAT32;
  query_ref.ref = 0;
  query_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> key_ref;
  key_ref.type = DataType::FLOAT32;
  key_ref.ref = 1;
  key_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> value_ref;
  value_ref.type = DataType::FLOAT32;
  value_ref.ref = 2;
  value_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> mask_ref;
  mask_ref.type = DataType::FLOAT32;
  mask_ref.ref = 3;
  mask_ref.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output_ref;
  output_ref.type = DataType::FLOAT32;
  output_ref.ref = 4;
  output_ref.shape = BHWC(1, 4, 1, 4);

  query->tensor = query_ref;
  key->tensor = key_ref;
  value->tensor = value_ref;
  mask->tensor = mask_ref;
  output->tensor = output_ref;

  model.AddConsumer(sdpa->id, query->id);
  model.AddConsumer(sdpa->id, key->id);
  model.AddConsumer(sdpa->id, value->id);
  model.AddConsumer(sdpa->id, mask->id);
  model.SetProducer(sdpa->id, output->id);

  ml_drift::TensorFloat32 input_tensor_q;
  input_tensor_q.shape = query_ref.shape;
  input_tensor_q.data.resize(input_tensor_q.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_q.shape.DimensionsProduct(); ++i) {
    input_tensor_q.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_k;
  input_tensor_k.shape = key_ref.shape;
  input_tensor_k.data.resize(input_tensor_k.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_k.shape.DimensionsProduct(); ++i) {
    input_tensor_k.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_v;
  input_tensor_v.shape = value_ref.shape;
  input_tensor_v.data.resize(input_tensor_v.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_v.shape.DimensionsProduct(); ++i) {
    input_tensor_v.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_mask;
  input_tensor_mask.shape = ml_drift::BHWC(1, 1, 1, 4);
  input_tensor_mask.data.resize(input_tensor_mask.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_mask.shape.DimensionsProduct(); ++i) {
    input_tensor_mask.data[i] = 0.0f;
  }

  TensorFloat32 inference_result;
  MLD_ASSERT_OK_AND_ASSIGN(
      inference_result,
      RunModel(model, input_tensor_q, input_tensor_k, input_tensor_v,
               input_tensor_mask, GetParam(), *exec_env));

  TensorFloat32 expected;
  expected.shape = ml_drift::BHWC(1, 4, 1, 4);
  expected.data.resize(expected.shape.DimensionsProduct());
  for (int i = 0; i < expected.shape.DimensionsProduct() / 4; i++) {
    expected.data[i * 4] = 12;
    expected.data[i * 4 + 1] = 13;
    expected.data[i * 4 + 2] = 14;
    expected.data[i * 4 + 3] = 15;
  }
  EXPECT_THAT(inference_result, TensorEq(testing::FloatNear(2e-5), expected));
}

TEST_P(SdpaTest, SdpaWithMaskZeroScale) {
  GraphFloat32 model;
  Node* sdpa = model.NewNode();
  sdpa->operation.type =
      ToString(ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION);
  ScaledDotProductAttentionAttributes attr;
  attr.scale = 0;
  sdpa->operation.attributes = std::move(attr);
  Value* query = model.NewValue();
  Value* key = model.NewValue();
  Value* value = model.NewValue();
  Value* mask = model.NewValue();
  Value* output = model.NewValue();

  TensorRef<BHWC> query_ref;
  query_ref.type = DataType::FLOAT32;
  query_ref.ref = 0;
  query_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> key_ref;
  key_ref.type = DataType::FLOAT32;
  key_ref.ref = 1;
  key_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> value_ref;
  value_ref.type = DataType::FLOAT32;
  value_ref.ref = 2;
  value_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> mask_ref;
  mask_ref.type = DataType::FLOAT32;
  mask_ref.ref = 3;
  mask_ref.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output_ref;
  output_ref.type = DataType::FLOAT32;
  output_ref.ref = 4;
  output_ref.shape = BHWC(1, 4, 1, 4);

  query->tensor = query_ref;
  key->tensor = key_ref;
  value->tensor = value_ref;
  mask->tensor = mask_ref;
  output->tensor = output_ref;

  model.AddConsumer(sdpa->id, query->id);
  model.AddConsumer(sdpa->id, key->id);
  model.AddConsumer(sdpa->id, value->id);
  model.AddConsumer(sdpa->id, mask->id);
  model.SetProducer(sdpa->id, output->id);

  ml_drift::TensorFloat32 input_tensor_q;
  input_tensor_q.shape = query_ref.shape;
  input_tensor_q.data.resize(input_tensor_q.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_q.shape.DimensionsProduct(); ++i) {
    input_tensor_q.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_k;
  input_tensor_k.shape = key_ref.shape;
  input_tensor_k.data.resize(input_tensor_k.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_k.shape.DimensionsProduct(); ++i) {
    input_tensor_k.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_v;
  input_tensor_v.shape = value_ref.shape;
  input_tensor_v.data.resize(input_tensor_v.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_v.shape.DimensionsProduct(); ++i) {
    input_tensor_v.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_mask;
  input_tensor_mask.shape = ml_drift::BHWC(1, 1, 1, 4);
  input_tensor_mask.data.resize(input_tensor_mask.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_mask.shape.DimensionsProduct(); ++i) {
    input_tensor_mask.data[i] = 0.0f;
  }

  TensorFloat32 inference_result;
  MLD_ASSERT_OK_AND_ASSIGN(
      inference_result,
      RunModel(model, input_tensor_q, input_tensor_k, input_tensor_v,
               input_tensor_mask, GetParam(), *exec_env));

  TensorFloat32 expected;
  expected.shape = ml_drift::BHWC(1, 4, 1, 4);
  expected.data.resize(expected.shape.DimensionsProduct());
  for (int i = 0; i < expected.shape.DimensionsProduct() / 4; i++) {
    expected.data[i * 4] = 6;
    expected.data[i * 4 + 1] = 7;
    expected.data[i * 4 + 2] = 8;
    expected.data[i * 4 + 3] = 9;
  }
  EXPECT_THAT(inference_result, TensorEq(testing::FloatNear(2e-5), expected));
}

TEST_P(SdpaTest, SdpaWithMaskNegativeScale) {
  GraphFloat32 model;
  Node* sdpa = model.NewNode();
  sdpa->operation.type =
      ToString(ml_drift::OperationType::SCALED_DOT_PRODUCT_ATTENTION);
  ScaledDotProductAttentionAttributes attr;
  attr.scale = -5;
  sdpa->operation.attributes = std::move(attr);
  Value* query = model.NewValue();
  Value* key = model.NewValue();
  Value* value = model.NewValue();
  Value* mask = model.NewValue();
  Value* output = model.NewValue();

  TensorRef<BHWC> query_ref;
  query_ref.type = DataType::FLOAT32;
  query_ref.ref = 0;
  query_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> key_ref;
  key_ref.type = DataType::FLOAT32;
  key_ref.ref = 1;
  key_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> value_ref;
  value_ref.type = DataType::FLOAT32;
  value_ref.ref = 2;
  value_ref.shape = BHWC(1, 4, 1, 4);

  TensorRef<BHWC> mask_ref;
  mask_ref.type = DataType::FLOAT32;
  mask_ref.ref = 3;
  mask_ref.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output_ref;
  output_ref.type = DataType::FLOAT32;
  output_ref.ref = 4;
  output_ref.shape = BHWC(1, 4, 1, 4);

  query->tensor = query_ref;
  key->tensor = key_ref;
  value->tensor = value_ref;
  mask->tensor = mask_ref;
  output->tensor = output_ref;

  model.AddConsumer(sdpa->id, query->id);
  model.AddConsumer(sdpa->id, key->id);
  model.AddConsumer(sdpa->id, value->id);
  model.AddConsumer(sdpa->id, mask->id);
  model.SetProducer(sdpa->id, output->id);

  ml_drift::TensorFloat32 input_tensor_q;
  input_tensor_q.shape = query_ref.shape;
  input_tensor_q.data.resize(input_tensor_q.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_q.shape.DimensionsProduct(); ++i) {
    input_tensor_q.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_k;
  input_tensor_k.shape = key_ref.shape;
  input_tensor_k.data.resize(input_tensor_k.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_k.shape.DimensionsProduct(); ++i) {
    input_tensor_k.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_v;
  input_tensor_v.shape = value_ref.shape;
  input_tensor_v.data.resize(input_tensor_v.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_v.shape.DimensionsProduct(); ++i) {
    input_tensor_v.data[i] = i % 20;
  }

  ml_drift::TensorFloat32 input_tensor_mask;
  input_tensor_mask.shape = ml_drift::BHWC(1, 1, 1, 4);
  input_tensor_mask.data.resize(input_tensor_mask.shape.DimensionsProduct());
  for (int i = 0; i < input_tensor_mask.shape.DimensionsProduct(); ++i) {
    input_tensor_mask.data[i] = 0.0f;
  }

  TensorFloat32 inference_result;
  MLD_ASSERT_OK_AND_ASSIGN(
      inference_result,
      RunModel(model, input_tensor_q, input_tensor_k, input_tensor_v,
               input_tensor_mask, GetParam(), *exec_env));

  TensorFloat32 expected;
  expected.shape = ml_drift::BHWC(1, 4, 1, 4);
  expected.data.resize(expected.shape.DimensionsProduct());
  for (int i = 0; i < expected.shape.DimensionsProduct() / 4; i++) {
    expected.data[i * 4] = 0;
    expected.data[i * 4 + 1] = 1;
    expected.data[i * 4 + 2] = 2;
    expected.data[i * 4 + 3] = 3;
  }
  EXPECT_THAT(inference_result, TensorEq(testing::FloatNear(2e-5), expected));
}

INSTANTIATE_TEST_SUITE_P(SdpaTestSuite, SdpaTest,
                         ValuesIn(GetTensorStoragesTypes()),
                         [](const TestParamInfo<SdpaTest::ParamType>& info) {
                           return absl::StrReplaceAll(ToString(info.param),
                                                      {{":", ""}});
                         });

}  // namespace ml_drift
