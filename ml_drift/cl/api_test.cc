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

#include "ml_drift/cl/api.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/cl/cl_test.h"
#include "ml_drift/common/api_cl_gl_vk.h"
#include "ml_drift/common/api_common.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"

namespace ml_drift::cl {
namespace {

using ApiTest = OpenCLTest;

TEST_F(ApiTest, NewInferenceBuilderPopulatesSerializedModel) {
  GraphFloat32 graph;
  Value* input = graph.NewValue();
  input->tensor.type = DataType::kFloat32;
  input->tensor.shape = BHWC(1, 2, 2, 1);
  Value* output = graph.NewValue();
  output->tensor.type = DataType::kFloat32;
  output->tensor.shape = BHWC(1, 2, 2, 1);
  Node* add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::kAdd);
  add_node->operation.attributes = ElementwiseAttributes{.param = 1.0f};
  graph.AddConsumer(add_node->id, input->id);
  graph.SetProducer(add_node->id, output->id);

  const InferenceEnvironmentOptions env_options{
      .device = env_.device().id(),
      .context = env_.context().context(),
      .command_queue = env_.queue()->queue(),
  };
  std::unique_ptr<InferenceEnvironment> inf_env;
  ABSL_ASSERT_OK(
      NewInferenceEnvironment(env_options, &inf_env, /*properties=*/nullptr));

  const InferenceOptions options{{.usage = InferenceUsage::kFastSingleAnswer}};
  std::unique_ptr<InferenceBuilder> builder;
  std::vector<uint8_t> serialized_model;
  ABSL_ASSERT_OK(inf_env->NewInferenceBuilder(options, std::move(graph), &builder,
                                         &serialized_model));
  EXPECT_NE(builder, nullptr);
  EXPECT_FALSE(serialized_model.empty());

  std::unique_ptr<InferenceBuilder> deserialized_builder;
  ABSL_EXPECT_OK(
      inf_env->NewInferenceBuilder(serialized_model, &deserialized_builder));
  EXPECT_NE(deserialized_builder, nullptr);
}

}  // namespace
}  // namespace ml_drift::cl
