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

#include "ml_drift/common/model.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(Model, SingleNode) {
  // graph_input -> node -> graph_output
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  graph.AddConsumer(node->id, graph_input->id);
  graph.SetProducer(node->id, graph_output->id);

  EXPECT_THAT(graph.nodes(), ElementsAre(node));
  EXPECT_THAT(graph.values(), UnorderedElementsAre(graph_input, graph_output));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindInputs(node->id), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.FindOutputs(node->id), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindConsumers(graph_input->id), UnorderedElementsAre(node));
  EXPECT_TRUE(graph.FindProducer(graph_output->id) == node);
  EXPECT_THAT(graph.FindConsumers(graph_output->id), UnorderedElementsAre());
  EXPECT_TRUE(graph.FindProducer(graph_input->id) == nullptr);
}

TEST(Model, SingleNodeMultipleOutputs) {
  // graph_input -> node -> (graph_output1, graph_output2)
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output1 = graph.NewValue();
  Value* graph_output2 = graph.NewValue();
  graph.AddConsumer(node->id, graph_input->id);
  graph.SetProducer(node->id, graph_output1->id);
  graph.SetProducer(node->id, graph_output2->id);
  EXPECT_THAT(graph.FindOutputs(node->id),
              UnorderedElementsAre(graph_output1, graph_output2));
  EXPECT_TRUE(graph.FindProducer(graph_output1->id) == node);
  EXPECT_TRUE(graph.FindProducer(graph_output2->id) == node);
}

TEST(Model, RemoveConsumer) {
  // (graph_input1, graph_input2) -> node
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_input1 = graph.NewValue();
  Value* graph_input2 = graph.NewValue();
  graph.AddConsumer(node->id, graph_input1->id);
  graph.AddConsumer(node->id, graph_input2->id);
  EXPECT_THAT(graph.FindConsumers(graph_input1->id),
              UnorderedElementsAre(node));
  EXPECT_THAT(graph.FindConsumers(graph_input2->id),
              UnorderedElementsAre(node));
  EXPECT_THAT(graph.FindInputs(node->id),
              UnorderedElementsAre(graph_input1, graph_input2));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre());

  // Now remove graph_input1
  MLD_ASSERT_OK(graph.RemoveConsumer(node->id, graph_input1->id));
  EXPECT_THAT(graph.FindConsumers(graph_input1->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindInputs(node->id), UnorderedElementsAre(graph_input2));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_input1));

  // Can not remove it twice
  EXPECT_FALSE(graph.RemoveConsumer(node->id, graph_input1->id).ok());
}

TEST(Model, ReplaceInput) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* v0 = graph.NewValue();
  Value* v1 = graph.NewValue();
  Value* v2 = graph.NewValue();
  Value* v3 = graph.NewValue();
  graph.AddConsumer(node->id, v0->id);
  graph.AddConsumer(node->id, v1->id);
  graph.AddConsumer(node->id, v2->id);
  EXPECT_THAT(graph.FindInputs(node->id), ElementsAre(v0, v1, v2));
  MLD_ASSERT_OK(graph.ReplaceInput(node->id, v1->id, v3->id));
  EXPECT_THAT(graph.FindInputs(node->id), ElementsAre(v0, v3, v2));
}

TEST(Model, RemoveProducer) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_output = graph.NewValue();

  graph.SetProducer(node->id, graph_output->id);
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre());
  EXPECT_TRUE(graph.FindProducer(graph_output->id) == node);

  MLD_ASSERT_OK(graph.RemoveProducer(graph_output->id));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_output));
  EXPECT_TRUE(graph.FindProducer(graph_output->id) == nullptr);

  // Can not remove producer twice
  EXPECT_FALSE(graph.RemoveProducer(graph_output->id).ok());
}

class OneNodeModel : public testing::Test {
 protected:
  void SetUp() override {
    node_ = graph_.NewNode();
    Value* graph_input = graph_.NewValue();
    Value* graph_output = graph_.NewValue();
    graph_.AddConsumer(node_->id, graph_input->id);
    graph_.SetProducer(node_->id, graph_output->id);
    EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input));
    EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output));
    EXPECT_THAT(graph_.nodes(), ElementsAre(node_));
  }
  GraphFloat32 graph_;
  Node* node_;
};

TEST_F(OneNodeModel, DeleteNodeKeepInput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepInput(&graph_, node_));
  EXPECT_TRUE(graph_.inputs().empty());
  EXPECT_TRUE(graph_.outputs().empty());
  EXPECT_TRUE(graph_.nodes().empty());
}

TEST_F(OneNodeModel, DeleteNodeKeepOutput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepOutput(&graph_, node_));
  EXPECT_TRUE(graph_.inputs().empty());
  EXPECT_TRUE(graph_.outputs().empty());
  EXPECT_TRUE(graph_.nodes().empty());
}

class TwoNodesModel : public testing::Test {
 protected:
  void SetUp() override {
    graph_input_ = graph_.NewValue();
    first_node_ = graph_.NewNode();
    value_ = graph_.NewValue();
    second_node_ = graph_.NewNode();
    graph_output_ = graph_.NewValue();

    graph_.AddConsumer(first_node_->id, graph_input_->id);
    graph_.SetProducer(first_node_->id, value_->id);
    graph_.AddConsumer(second_node_->id, value_->id);
    graph_.SetProducer(second_node_->id, graph_output_->id);
    EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
    EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
    EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_, second_node_));
  }
  GraphFloat32 graph_;
  Node* first_node_;
  Node* second_node_;
  Value* graph_input_;
  Value* value_;
  Value* graph_output_;
};

TEST_F(TwoNodesModel, DeleteFirstNodeKeepInput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepInput(&graph_, first_node_));
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(second_node_));
}

TEST_F(TwoNodesModel, DeleteFirstNodeKeepOutput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepOutput(&graph_, first_node_));
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(value_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(second_node_));
}

TEST_F(TwoNodesModel, DeleteSecondNodeKeepInput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepInput(&graph_, second_node_));
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(value_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_));
}

TEST_F(TwoNodesModel, DeleteSecondNodeKeepOutput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepOutput(&graph_, second_node_));
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_));
}

class ThreeNodesModel : public testing::Test {
 protected:
  void SetUp() override {
    first_node_ = graph_.NewNode();
    second_node_ = graph_.NewNode();
    third_node_ = graph_.NewNode();
    graph_input_ = graph_.NewValue();
    value0_ = graph_.NewValue();
    value1_ = graph_.NewValue();
    graph_output_ = graph_.NewValue();

    graph_.AddConsumer(first_node_->id, graph_input_->id);
    graph_.SetProducer(first_node_->id, value0_->id);
    graph_.AddConsumer(second_node_->id, value0_->id);
    graph_.SetProducer(second_node_->id, value1_->id);
    graph_.AddConsumer(third_node_->id, value1_->id);
    graph_.SetProducer(third_node_->id, graph_output_->id);
    EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
    EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
    EXPECT_THAT(graph_.nodes(),
                ElementsAre(first_node_, second_node_, third_node_));
  }
  GraphFloat32 graph_;
  Node* first_node_;
  Node* second_node_;
  Node* third_node_;
  Value* graph_input_;
  Value* value0_;
  Value* value1_;
  Value* graph_output_;
};

TEST_F(ThreeNodesModel, DeleteMiddleNodeKeepInput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepInput(&graph_, second_node_));
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_, third_node_));
  EXPECT_THAT(graph_.values(),
              UnorderedElementsAre(graph_input_, value0_, graph_output_));
}

TEST_F(ThreeNodesModel, DeleteMiddleNodeKeepOutput) {
  MLD_ASSERT_OK(RemoveSimpleNodeKeepOutput(&graph_, second_node_));
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_, third_node_));
  EXPECT_THAT(graph_.values(),
              UnorderedElementsAre(graph_input_, value1_, graph_output_));
}

TEST(Model, RemoveSimpleNodeKeepInputComplexCase) {
  // We have this graph and we are going to delete n1 and preserve order of
  // v0, v1 for n0 node and v2, v3 for n2 node
  //  v0   v1
  //   \  /  \
  //    n0    n1
  //    |      \
  //    o1      v2   v3
  //             \  /
  //              n2
  //              |
  //              o2
  //
  // And we are going to receive this:
  //  v0   v1
  //   \  /  \
  //    n0    \
  //    |      \
  //    o1      \   v3
  //             \  /
  //              n2
  //              |
  //              o2
  GraphFloat32 graph;
  Node* n0 = graph.NewNode();
  Node* n1 = graph.NewNode();  // node to remove
  Node* n2 = graph.NewNode();
  Value* v0 = graph.NewValue();
  Value* v1 = graph.NewValue();
  Value* v2 = graph.NewValue();  // value to be removed
  Value* v3 = graph.NewValue();
  Value* o1 = graph.NewValue();
  Value* o2 = graph.NewValue();

  graph.AddConsumer(n0->id, v0->id);
  graph.AddConsumer(n0->id, v1->id);
  graph.SetProducer(n0->id, o1->id);
  graph.AddConsumer(n1->id, v1->id);
  graph.SetProducer(n1->id, v2->id);
  graph.AddConsumer(n2->id, v2->id);
  graph.AddConsumer(n2->id, v3->id);
  graph.SetProducer(n2->id, o2->id);
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(v0, v1, v3));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(o1, o2));
  EXPECT_THAT(graph.nodes(), ElementsAre(n0, n1, n2));

  // Node should be the only consumer of the input value to be able to be
  // deleted with this function.
  ASSERT_FALSE(RemoveSimpleNodeKeepOutput(&graph, n1).ok());

  MLD_ASSERT_OK(RemoveSimpleNodeKeepInput(&graph, n1));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(v0, v1, v3));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(o1, o2));
  EXPECT_THAT(graph.nodes(), ElementsAre(n0, n2));
  EXPECT_THAT(graph.values(), UnorderedElementsAre(v0, v1, v3, o1, o2));
  EXPECT_THAT(graph.FindInputs(n0->id), ElementsAre(v0, v1));
  EXPECT_THAT(graph.FindInputs(n2->id), ElementsAre(v1, v3));
}

TEST(Model, ReassignValue) {
  // Before:
  //   graph_input  -> node1 -> graph_output
  //              \ -> node2
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  graph.AddConsumer(node1->id, graph_input->id);
  graph.SetProducer(node1->id, graph_output->id);
  graph.AddConsumer(node2->id, graph_input->id);

  // After:
  //   graph_input  -> node1
  //              \ -> node2 -> graph_output
  graph.SetProducer(node2->id, graph_output->id);

  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2));
  EXPECT_THAT(graph.FindInputs(node1->id), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.FindOutputs(node1->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindOutputs(node2->id), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindConsumers(graph_input->id),
              UnorderedElementsAre(node1, node2));
  EXPECT_TRUE(graph.FindProducer(graph_output->id) == node2);
  EXPECT_THAT(graph.FindConsumers(graph_output->id), UnorderedElementsAre());
}

TEST(Model, DeleteValue) {
  // graph_input  -> node1 -> value -> node2 -> graph_output
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value = graph.NewValue();
  graph.AddConsumer(node1->id, graph_input->id);
  graph.SetProducer(node1->id, value->id);
  graph.AddConsumer(node2->id, value->id);
  graph.SetProducer(node2->id, graph_output->id);

  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));
  EXPECT_TRUE(graph.FindProducer(value->id) == node1);
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(value));
  EXPECT_THAT(graph.FindOutputs(node1->id), UnorderedElementsAre(value));

  MLD_ASSERT_OK(graph.DeleteValue(value->id));
  value = nullptr;
  EXPECT_THAT(graph.values(), UnorderedElementsAre(graph_input, graph_output));
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindOutputs(node1->id), UnorderedElementsAre());

  MLD_ASSERT_OK(graph.DeleteValue(graph_input->id));
  graph_input = nullptr;
  EXPECT_THAT(graph.values(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.FindInputs(node1->id), UnorderedElementsAre());

  MLD_ASSERT_OK(graph.DeleteValue(graph_output->id));
  graph_output = nullptr;
  EXPECT_THAT(graph.values(), UnorderedElementsAre());
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.FindOutputs(node2->id), UnorderedElementsAre());
}

TEST(Model, DeleteNode) {
  // graph_input -> node1 -> value  -> node2 -> graph_output
  //                               \-> node3 -> graph_output2
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Node* node3 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* graph_output2 = graph.NewValue();
  Value* value = graph.NewValue();
  graph.AddConsumer(node1->id, graph_input->id);
  graph.SetProducer(node1->id, value->id);
  graph.AddConsumer(node2->id, value->id);
  graph.AddConsumer(node3->id, value->id);
  graph.SetProducer(node2->id, graph_output->id);
  graph.SetProducer(node3->id, graph_output2->id);

  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2, node3));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id),
              UnorderedElementsAre(node2, node3));
  EXPECT_TRUE(graph.FindProducer(value->id) == node1);
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(value));
  EXPECT_THAT(graph.FindInputs(node3->id), UnorderedElementsAre(value));

  // graph_input  -> node1 -> value -> node2 -> graph_output
  // graph_output2
  MLD_ASSERT_OK(graph.DeleteNode(node3->id));
  node3 = nullptr;
  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input, graph_output2));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));

  // value -> node2 -> graph_output
  // graph_input
  // graph_output2
  MLD_ASSERT_OK(graph.DeleteNode(node1->id));
  node1 = nullptr;
  EXPECT_THAT(graph.nodes(), ElementsAre(node2));
  EXPECT_THAT(graph.inputs(),
              UnorderedElementsAre(value, graph_output2, graph_input));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_input, graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));
  EXPECT_TRUE(graph.FindProducer(value->id) == nullptr);

  MLD_ASSERT_OK(graph.DeleteNode(node2->id));
  node2 = nullptr;
  EXPECT_THAT(graph.nodes(), ElementsAre());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_output, graph_output2,
                                                   graph_input, value));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output, graph_output2,
                                                    graph_input, value));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre());
  EXPECT_TRUE(graph.FindProducer(value->id) == nullptr);
}

TEST(Model, InsertNodeAfter) {
  // graph_input -> node1 -> value -> node2 -> graph_output
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value = graph.NewValue();
  graph.AddConsumer(node1->id, graph_input->id);
  graph.SetProducer(node1->id, value->id);
  graph.AddConsumer(node2->id, value->id);
  graph.SetProducer(node2->id, graph_output->id);

  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));
  EXPECT_TRUE(graph.FindProducer(value->id) == node1);
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(value));

  Node* new_node1;
  MLD_ASSERT_OK(graph.InsertNodeAfter(node1->id, &new_node1));
  EXPECT_THAT(graph.nodes(), ElementsAre(node1, new_node1, node2));

  Node* new_node2;
  EXPECT_EQ(graph.InsertNodeAfter(/*id=*/100, &new_node2).code(),
            absl::StatusCode::kOutOfRange);

  MLD_ASSERT_OK(graph.InsertNodeAfter(node2->id, &new_node2));
  EXPECT_THAT(graph.nodes(), ElementsAre(node1, new_node1, node2, new_node2));
}

}  // namespace
}  // namespace ml_drift
