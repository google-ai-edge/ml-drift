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

#include "ml_drift/common/transformations/make_fully_connected.h"

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_transformer.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

TEST(MakeFullyConnected, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 4, 4, 8);

  Convolution2DAttributes attr0;
  attr0.padding.prepended = HW(0, 0);
  attr0.padding.appended = HW(0, 0);
  attr0.strides = HW(1, 1);
  attr0.dilations = HW(1, 1);
  auto& attr0_weights =
      attr0.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr0_weights.shape = OHWI(16, 1, 1, 8);
  attr0.bias.shape = Linear(16);

  Convolution2DAttributes attr1;
  attr1.padding.prepended = HW(0, 0);
  attr1.padding.appended = HW(0, 0);
  attr1.strides = HW(4, 4);
  attr1.dilations = HW(1, 1);
  auto& attr1_weights =
      attr1.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr1_weights.shape = OHWI(16, 4, 4, 16);
  attr1.bias.shape = Linear(16);

  Convolution2DAttributes attr2;
  attr2.padding.prepended = HW(0, 0);
  attr2.padding.appended = HW(0, 0);
  attr2.strides = HW(1, 1);
  attr2.dilations = HW(1, 1);
  auto& attr2_weights =
      attr2.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr2_weights.shape = OHWI(32, 1, 1, 16);
  attr2.bias.shape = Linear(32);

  auto conv1x1_node0 = graph.NewNode();
  conv1x1_node0->operation.type = ToString(OperationType::kConvolution2D);
  conv1x1_node0->operation.attributes = attr0;
  auto conv4x4_node1 = graph.NewNode();
  conv4x4_node1->operation.type = ToString(OperationType::kConvolution2D);
  conv4x4_node1->operation.attributes = attr1;
  auto conv1x1_node2 = graph.NewNode();
  conv1x1_node2->operation.type = ToString(OperationType::kConvolution2D);
  conv1x1_node2->operation.attributes = attr2;

  graph.AddConsumer(conv1x1_node0->id, input->id);

  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, conv1x1_node2, &output).ok());
  output->tensor.shape = BHWC(1, 1, 1, 32);

  Value* link1 = nullptr;
  ASSERT_TRUE(
      ConnectTwoNodes(&graph, conv1x1_node0, conv4x4_node1, &link1).ok());
  link1->tensor.shape = BHWC(1, 4, 4, 16);

  Value* link2 = nullptr;
  ASSERT_TRUE(
      ConnectTwoNodes(&graph, conv4x4_node1, conv1x1_node2, &link2).ok());
  link2->tensor.shape = BHWC(1, 1, 1, 16);

  ASSERT_EQ(3, graph.nodes().size());
  ASSERT_EQ(4, graph.values().size());

  auto transformation = NewMakeFullyConnectedFromConvolution();
  ModelTransformer transformer(&graph);
  transformer.Apply("make_fully_connected", transformation.get());

  ASSERT_EQ(3, graph.nodes().size());
  ASSERT_EQ(4, graph.values().size());
  ASSERT_EQ(ToString(OperationType::kConvolution2D),
            graph.nodes()[0]->operation.type);
  ASSERT_EQ(ToString(OperationType::kConvolution2D),
            graph.nodes()[1]->operation.type);
  ASSERT_EQ(ToString(OperationType::kFullyConnected),
            graph.nodes()[2]->operation.type);
  auto fc_attr = std::any_cast<FullyConnectedAttributes>(
      graph.nodes()[2]->operation.attributes);
  EXPECT_EQ(OHWI(32, 1, 1, 16), fc_attr.weights.shape);
  EXPECT_EQ(Linear(32), fc_attr.bias.shape);
}

}  // namespace
}  // namespace ml_drift
