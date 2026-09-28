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

#include "ml_drift/common/transformations/remove_noop.h"

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
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

using ::testing::UnorderedElementsAre;

TEST(RemoveSingleInputAdd, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  graph.AddConsumer(first_node->id, input->id);

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::kAdd);
  add_node->operation.attributes = ElementwiseAttributes();

  Value* temp = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, add_node, &temp).ok());
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  EXPECT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());
  ASSERT_EQ(first_node, graph.nodes()[0]);
  ASSERT_EQ(input, graph.values()[0]);
  ASSERT_EQ(output, graph.values()[1]);
}

TEST(RemoveSingleInputAdd, DoNotTrigger_LinearTensor) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  graph.AddConsumer(first_node->id, input->id);

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::kAdd);
  ElementwiseAttributes attr;
  attr.param = Tensor<Linear, DataType::kFloat32>();
  add_node->operation.attributes = attr;

  Value* temp = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, add_node, &temp).ok());
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  EXPECT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());
}

TEST(RemoveSingleInputAdd, DoNotTrigger_Scalar) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  graph.AddConsumer(first_node->id, input->id);

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::kAdd);
  ElementwiseAttributes attr;
  attr.param = 0.5f;
  add_node->operation.attributes = attr;

  Value* temp = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, add_node, &temp).ok());
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  EXPECT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());
}

TEST(RemoveSingleInputAdd, DoNotTrigger_Multiple) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto node_a = graph.NewNode();
  auto node_b = graph.NewNode();
  graph.AddConsumer(node_a->id, input->id);
  graph.AddConsumer(node_b->id, input->id);

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::kAdd);

  Value* temp_a = nullptr;
  Value* temp_b = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, node_a, add_node, &temp_a).ok());
  ASSERT_TRUE(ConnectTwoNodes(&graph, node_b, add_node, &temp_b).ok());
  ASSERT_EQ(3, graph.nodes().size());
  ASSERT_EQ(4, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  ASSERT_EQ(3, graph.nodes().size());
  ASSERT_EQ(4, graph.values().size());
}

TEST(RemoveDegenerateUpsampling, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  graph.AddConsumer(first_node->id, input->id);

  auto node_to_remove = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, node_to_remove, &output).ok());
  output->tensor.shape = BHWC(1, 5, 5, 1);
  node_to_remove->operation.type = ToString(OperationType::kResize);
  Resize2DAttributes attr;
  attr.new_shape = HW(5, 5);
  attr.type = SamplingType::kBilinear;
  node_to_remove->operation.attributes = attr;

  Value* link = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, node_to_remove, &link).ok());
  link->tensor.shape = output->tensor.shape;
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveDegenerateUpsampling();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  ASSERT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());
  EXPECT_EQ(first_node, graph.nodes()[0]);
  EXPECT_EQ(input, graph.values()[0]);
  EXPECT_EQ(output, graph.values()[1]);
}

TEST(RemoveIdentityReshape, Smoke) {
  GraphFloat32 graph;
  Node* simple_node = graph.NewNode();
  Node* producer_node = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 1, 1, 11);
  simple_node->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr;
  attr.new_shape = BHWC(1, 1, 1, 11);
  simple_node->operation.attributes = attr;

  graph.AddConsumer(producer_node->id, graph_input->id);
  graph.SetProducer(producer_node->id, value0->id);
  graph.AddConsumer(simple_node->id, value0->id);
  graph.SetProducer(simple_node->id, value1->id);
  graph.AddConsumer(consumer_node->id, value1->id);
  graph.SetProducer(consumer_node->id, graph_output->id);
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(simple_node, producer_node, consumer_node));

  auto transformation = NewRemoveIdentityReshape();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(producer_node, consumer_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value0));
}


TEST(RemoveIdentityReshape, SkipWhenProducerAlreadyFedToReshapeConsumer) {
  //      ||
  //      \/
  // (producer node)=======
  //      ||              ||
  //      \/              ||
  //   [value0]           ||
  //      ||              ||
  //      \/              ||
  // (reshape node)       ||
  //      ||              ||
  //      \/              ||
  //   [value1]           ||
  //      ||              ||
  //      \/              \/
  // (consumer node) <======
  //      ||
  //      \/

  GraphFloat32 graph;
  Node* simple_node = graph.NewNode();
  Node* producer_node = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 1, 1, 11);
  simple_node->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr;
  attr.new_shape = BHWC(1, 1, 1, 11);
  simple_node->operation.attributes = attr;

  graph.AddConsumer(producer_node->id, graph_input->id);
  graph.SetProducer(producer_node->id, value0->id);
  graph.AddConsumer(simple_node->id, value0->id);
  graph.SetProducer(simple_node->id, value1->id);
  graph.AddConsumer(consumer_node->id, value0->id);
  graph.AddConsumer(consumer_node->id, value1->id);
  graph.SetProducer(consumer_node->id, graph_output->id);
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(simple_node, producer_node, consumer_node));

  auto transformation = NewRemoveIdentityReshape();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(simple_node, producer_node, consumer_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value0, value1));
}

TEST(MergeConsecutiveReshapes, Smoke) {
  GraphFloat32 graph;
  Node* producer_node = graph.NewNode();
  Node* reshape0 = graph.NewNode();
  Node* reshape1 = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();
  Value* value2 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 1, 1, 12);
  value1->tensor.shape = BHWC(1, 2, 2, 3);
  value2->tensor.shape = BHWC(1, 3, 2, 2);

  reshape0->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr0;
  attr0.new_shape = BHWC(1, 2, 2, 3);
  reshape0->operation.attributes = attr0;

  reshape1->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr1;
  attr1.new_shape = BHWC(1, 3, 2, 2);
  reshape1->operation.attributes = attr1;

  graph.AddConsumer(producer_node->id, graph_input->id);
  graph.SetProducer(producer_node->id, value0->id);
  graph.AddConsumer(reshape0->id, value0->id);
  graph.SetProducer(reshape0->id, value1->id);
  graph.AddConsumer(reshape1->id, value1->id);
  graph.SetProducer(reshape1->id, value2->id);
  graph.AddConsumer(consumer_node->id, value2->id);
  graph.SetProducer(consumer_node->id, graph_output->id);

  auto transformation = NewMergeConsecutiveReshapes();
  ModelTransformer transformer(&graph);
  ASSERT_TRUE(transformer.Apply("merge_reshapes", transformation.get()));

  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(producer_node, reshape0, consumer_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value0, value2));

  const auto& merged_attr =
      std::any_cast<const ReshapeAttributes&>(reshape0->operation.attributes);
  EXPECT_EQ(merged_attr.new_shape, BHWC(1, 3, 2, 2));
}

TEST(MergeConsecutiveReshapes, CancellingReshapesWithRemoveIdentity) {
  GraphFloat32 graph;
  Node* producer_node = graph.NewNode();
  Node* reshape0 = graph.NewNode();
  Node* reshape1 = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();
  Value* value2 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 12, 630, 630);
  value1->tensor.shape = BHWC(12, 1, 630, 630);
  value2->tensor.shape = BHWC(1, 12, 630, 630);

  reshape0->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr0;
  attr0.new_shape = BHWC(12, 1, 630, 630);
  reshape0->operation.attributes = attr0;

  reshape1->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr1;
  attr1.new_shape = BHWC(1, 12, 630, 630);
  reshape1->operation.attributes = attr1;

  graph.AddConsumer(producer_node->id, graph_input->id);
  graph.SetProducer(producer_node->id, value0->id);
  graph.AddConsumer(reshape0->id, value0->id);
  graph.SetProducer(reshape0->id, value1->id);
  graph.AddConsumer(reshape1->id, value1->id);
  graph.SetProducer(reshape1->id, value2->id);
  graph.AddConsumer(consumer_node->id, value2->id);
  graph.SetProducer(consumer_node->id, graph_output->id);

  auto merge_transform = NewMergeConsecutiveReshapes();
  auto remove_identity_transform = NewRemoveIdentityReshape();
  ModelTransformer transformer(&graph);
  ASSERT_TRUE(transformer.Apply("merge_reshapes", merge_transform.get()));
  ASSERT_TRUE(
      transformer.Apply("remove_identity", remove_identity_transform.get()));

  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(producer_node, consumer_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value0));
}

TEST(MergeConsecutiveReshapes, SkipWhenIntermediateHasMultipleConsumers) {
  GraphFloat32 graph;
  Node* producer_node = graph.NewNode();
  Node* reshape0 = graph.NewNode();
  Node* reshape1 = graph.NewNode();
  Node* other_consumer = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* other_output = graph.NewValue();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();
  Value* value2 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 1, 1, 12);
  value1->tensor.shape = BHWC(1, 2, 2, 3);
  value2->tensor.shape = BHWC(1, 3, 2, 2);

  reshape0->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr0;
  attr0.new_shape = BHWC(1, 2, 2, 3);
  reshape0->operation.attributes = attr0;

  reshape1->operation.type = ToString(OperationType::kReshape);
  ReshapeAttributes attr1;
  attr1.new_shape = BHWC(1, 3, 2, 2);
  reshape1->operation.attributes = attr1;

  graph.AddConsumer(producer_node->id, graph_input->id);
  graph.SetProducer(producer_node->id, value0->id);
  graph.AddConsumer(reshape0->id, value0->id);
  graph.SetProducer(reshape0->id, value1->id);
  graph.AddConsumer(reshape1->id, value1->id);
  graph.SetProducer(reshape1->id, value2->id);
  graph.AddConsumer(other_consumer->id, value1->id);
  graph.SetProducer(other_consumer->id, other_output->id);
  graph.AddConsumer(consumer_node->id, value2->id);
  graph.SetProducer(consumer_node->id, graph_output->id);

  auto transformation = NewMergeConsecutiveReshapes();
  ModelTransformer transformer(&graph);
  ASSERT_TRUE(transformer.Apply("merge_reshapes", transformation.get()));

  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(producer_node, reshape0, reshape1,
                                   other_consumer, consumer_node));
}

TEST(RemoveIdentityStridedSlice, Smoke) {
  GraphFloat32 graph;
  Node* simple_node = graph.NewNode();
  Node* producer_node = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 1, 1, 11);
  value1->tensor.shape = BHWC(1, 1, 1, 11);
  simple_node->operation.type = ToString(OperationType::kSlice);
  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 0, 0);
  attr.strides = BHWC(1, 1, 1, 1);
  attr.ends = BHWC(1, 1, 1, 11);
  simple_node->operation.attributes = attr;

  graph.AddConsumer(producer_node->id, graph_input->id);
  graph.SetProducer(producer_node->id, value0->id);
  graph.AddConsumer(simple_node->id, value0->id);
  graph.SetProducer(simple_node->id, value1->id);
  graph.AddConsumer(consumer_node->id, value1->id);
  graph.SetProducer(consumer_node->id, graph_output->id);
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(simple_node, producer_node, consumer_node));

  auto transformation = NewRemoveIdentityStridedSlice();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(producer_node, consumer_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value0));
}

TEST(RemoveIdentityStridedSlice, OutputIsGraphOutputInputConsumedByFewNodes) {
  //   [value0]
  //      ||
  //      \/
  // (first node)
  //      ||
  //      \/
  //   [value1]============
  //      ||              ||
  //      \/              \/
  // (slice node)    (second node)
  //      ||              ||
  //      \/              \/
  //   [value2]        [value3]

  GraphFloat32 graph;
  Node* first_node = graph.NewNode();
  Node* slice_node = graph.NewNode();
  Node* second_node = graph.NewNode();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();
  Value* value2 = graph.NewValue();
  Value* value3 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 1, 1, 11);
  value1->tensor.shape = BHWC(1, 1, 1, 11);
  value2->tensor.shape = BHWC(1, 1, 1, 11);
  value3->tensor.shape = BHWC(1, 1, 1, 11);
  slice_node->operation.type = ToString(OperationType::kSlice);
  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 0, 0);
  attr.strides = BHWC(1, 1, 1, 1);
  attr.ends = BHWC(1, 1, 1, 11);
  slice_node->operation.attributes = attr;

  graph.AddConsumer(first_node->id, value0->id);
  graph.SetProducer(first_node->id, value1->id);
  graph.AddConsumer(slice_node->id, value1->id);
  graph.AddConsumer(second_node->id, value1->id);
  graph.SetProducer(slice_node->id, value2->id);
  graph.SetProducer(second_node->id, value3->id);
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(value0));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(value2, value3));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(first_node, slice_node, second_node));

  auto transformation = NewRemoveIdentityStridedSlice();
  ModelTransformer transformer(&graph);
  transformer.Apply("noop", transformation.get());

  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(value0));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(value2, value3));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(first_node, slice_node, second_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(value0, value1, value2, value3));
}

}  // namespace
}  // namespace ml_drift
