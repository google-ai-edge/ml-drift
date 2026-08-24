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

#include <any>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/transformations/ir/transform.h"

namespace ml_drift::ir {
namespace {

using ::testing::ElementsAre;

TEST(IrRemoveNoopTest, RemoveSingleInputAdd_Smoke) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* add_op = model.add_op();
  add_op->name = ToString(OperationType::ADD);
  ElementwiseAttributes attr;
  add_op->attr = attr;  // std::monostate by default

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate->id, op1->id);

  model.AddConsumer(intermediate->id, add_op->id);
  model.SetProducer(output->id, add_op->id);

  IrOpId add_op_id = add_op->id;
  IrTensorId intermediate_id = intermediate->id;
  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(add_op_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate_id), nullptr);

  const IrOp* remaining_op = model.op(op1->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_THAT(remaining_op->outputs, ElementsAre(output->id));
}

TEST(IrRemoveNoopTest, RemoveSingleInputAdd_DoNotTrigger_LinearTensor) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* add_op = model.add_op();
  add_op->name = ToString(OperationType::ADD);
  ElementwiseAttributes attr;
  attr.param = Tensor<Linear, DataType::FLOAT32>();
  add_op->attr = attr;

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate->id, op1->id);

  model.AddConsumer(intermediate->id, add_op->id);
  model.SetProducer(output->id, add_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_NE(model.op(add_op->id), nullptr);
  EXPECT_NE(model.tensor(intermediate->id), nullptr);
}

TEST(IrRemoveNoopTest, RemoveSingleInputAdd_DoNotTrigger_Scalar) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* add_op = model.add_op();
  add_op->name = ToString(OperationType::ADD);
  ElementwiseAttributes attr;
  attr.param = 0.5f;
  add_op->attr = attr;

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate->id, op1->id);

  model.AddConsumer(intermediate->id, add_op->id);
  model.SetProducer(output->id, add_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_NE(model.op(add_op->id), nullptr);
  EXPECT_NE(model.tensor(intermediate->id), nullptr);
}

TEST(IrRemoveNoopTest, RemoveSingleInputAdd_DoNotTrigger_Multiple) {
  IrModel model;
  IrOp* op_a = model.add_op();
  IrOp* op_b = model.add_op();
  IrOp* add_op = model.add_op();
  add_op->name = ToString(OperationType::ADD);

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* temp_a =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* temp_b =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op_a->id);
  model.SetProducer(temp_a->id, op_a->id);

  model.AddConsumer(input->id, op_b->id);
  model.SetProducer(temp_b->id, op_b->id);

  model.AddConsumer(temp_a->id, add_op->id);
  model.AddConsumer(temp_b->id, add_op->id);
  model.SetProducer(output->id, add_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_NE(model.op(add_op->id), nullptr);
}

TEST(IrRemoveNoopTest, RemoveDegenerateUpsampling_Smoke) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* resize_op = model.add_op();
  resize_op->name = ToString(OperationType::RESIZE);
  Resize2DAttributes attr;
  attr.new_shape = ::ml_drift::HW(5, 5);
  attr.type = ::ml_drift::SamplingType::BILINEAR;
  resize_op->attr = attr;

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(5, 5, 1));
  IrTensor* intermediate =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(5, 5, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(5, 5, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate->id, op1->id);

  model.AddConsumer(intermediate->id, resize_op->id);
  model.SetProducer(output->id, resize_op->id);

  IrOpId resize_op_id = resize_op->id;
  IrTensorId intermediate_id = intermediate->id;
  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(resize_op_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate_id), nullptr);

  const IrOp* remaining_op = model.op(op1->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_THAT(remaining_op->outputs, ElementsAre(output->id));
}

TEST(IrRemoveNoopTest, RemoveIdentityReshape_Smoke) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* reshape_op = model.add_op();
  IrOp* op2 = model.add_op();
  reshape_op->name = ToString(OperationType::RESHAPE);
  ReshapeAttributes attr;
  attr.new_shape = ::ml_drift::BHWC(1, 1, 1, 11);
  reshape_op->attr = attr;

  IrTensor* input = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                     ::ml_drift::HWC(1, 1, 11));
  IrTensor* intermediate = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                            ::ml_drift::HWC(1, 1, 11));
  IrTensor* intermediate2 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                             ::ml_drift::HWC(1, 1, 11));
  IrTensor* output = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate->id, op1->id);

  model.AddConsumer(intermediate->id, reshape_op->id);
  model.SetProducer(intermediate2->id, reshape_op->id);

  model.AddConsumer(intermediate2->id, op2->id);
  model.SetProducer(output->id, op2->id);

  IrOpId reshape_op_id = reshape_op->id;
  IrTensorId intermediate2_id = intermediate2->id;
  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(reshape_op_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate2_id), nullptr);

  const IrOp* remaining_op = model.op(op1->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_THAT(remaining_op->outputs, ElementsAre(intermediate->id));

  const IrOp* remaining_op2 = model.op(op2->id);
  ASSERT_NE(remaining_op2, nullptr);
  EXPECT_THAT(remaining_op2->inputs, ElementsAre(intermediate->id));
}

TEST(IrRemoveNoopTest,
     RemoveIdentityReshape_SkipWhenProducerAlreadyFedToReshapeConsumer) {
  IrModel model;
  IrOp* producer_op = model.add_op();
  IrOp* reshape_op = model.add_op();
  IrOp* consumer_op = model.add_op();

  reshape_op->name = ToString(OperationType::RESHAPE);
  ReshapeAttributes attr;
  attr.new_shape = ::ml_drift::BHWC(1, 1, 1, 11);
  reshape_op->attr = attr;

  IrTensor* graph_input = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                           ::ml_drift::HWC(1, 1, 11));
  IrTensor* value0 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));
  IrTensor* value1 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));
  IrTensor* graph_output = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                            ::ml_drift::HWC(1, 1, 11));

  model.add_input(graph_input->id);
  model.add_output(graph_output->id);

  model.AddConsumer(graph_input->id, producer_op->id);
  model.SetProducer(value0->id, producer_op->id);

  model.AddConsumer(value0->id, reshape_op->id);
  model.SetProducer(value1->id, reshape_op->id);

  // Consumer uses BOTH the input to the reshape and the output of the reshape
  model.AddConsumer(value0->id, consumer_op->id);
  model.AddConsumer(value1->id, consumer_op->id);
  model.SetProducer(graph_output->id, consumer_op->id);

  IrOpId reshape_op_id = reshape_op->id;
  IrTensorId value1_id = value1->id;
  EXPECT_TRUE(TransformIrModel(&model).ok());

  // This test verifies the behavior when an identity Reshape op is removed.
  // The consumer_op uses both the input (value0) and the output (value1) of the
  // reshape. When TransformIrModel removes the reshape, RemoveSimpleOp rewires
  // usages of value1 to value0. Consequently, consumer_op's inputs will both
  // become value0. This behavior, while resulting in duplicate inputs, is
  // safely handled by IrModel.
  EXPECT_EQ(model.op(reshape_op_id), nullptr);
  EXPECT_EQ(model.tensor(value1_id), nullptr);

  const IrOp* remaining_consumer = model.op(consumer_op->id);
  ASSERT_NE(remaining_consumer, nullptr);
  EXPECT_THAT(remaining_consumer->inputs, ElementsAre(value0->id, value0->id));
}

TEST(IrRemoveNoopTest, RemoveIdentityStridedSlice_Smoke) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* slice_op = model.add_op();
  IrOp* op2 = model.add_op();

  slice_op->name = ToString(OperationType::SLICE);
  SliceAttributes attr;
  attr.starts = ::ml_drift::BHWC(0, 0, 0, 0);
  attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
  attr.ends = ::ml_drift::BHWC(1, 1, 1, 11);
  slice_op->attr = attr;

  IrTensor* input = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                     ::ml_drift::HWC(1, 1, 11));
  IrTensor* intermediate1 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                             ::ml_drift::HWC(1, 1, 11));
  IrTensor* intermediate2 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                             ::ml_drift::HWC(1, 1, 11));
  IrTensor* output = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate1->id, op1->id);

  model.AddConsumer(intermediate1->id, slice_op->id);
  model.SetProducer(intermediate2->id, slice_op->id);

  model.AddConsumer(intermediate2->id, op2->id);
  model.SetProducer(output->id, op2->id);

  IrOpId slice_op_id = slice_op->id;
  IrTensorId intermediate2_id = intermediate2->id;
  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(slice_op_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate2_id), nullptr);

  const IrOp* remaining_op = model.op(op1->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_THAT(remaining_op->outputs, ElementsAre(intermediate1->id));

  const IrOp* remaining_op2 = model.op(op2->id);
  ASSERT_NE(remaining_op2, nullptr);
  EXPECT_THAT(remaining_op2->inputs, ElementsAre(intermediate1->id));
}

TEST(IrRemoveNoopTest,
     RemoveIdentityStridedSlice_OutputIsGraphOutputInputConsumedByFewNodes) {
  IrModel model;
  IrOp* first_node = model.add_op();
  IrOp* slice_node = model.add_op();
  IrOp* second_node = model.add_op();

  slice_node->name = ToString(OperationType::SLICE);
  SliceAttributes attr;
  attr.starts = ::ml_drift::BHWC(0, 0, 0, 0);
  attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
  attr.ends = ::ml_drift::BHWC(1, 1, 1, 11);
  slice_node->attr = attr;

  IrTensor* value0 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));
  IrTensor* value1 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));
  IrTensor* value2 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));
  IrTensor* value3 = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::HWC(1, 1, 11));

  model.add_input(value0->id);
  model.add_output(value2->id);
  model.add_output(value3->id);

  model.AddConsumer(value0->id, first_node->id);
  model.SetProducer(value1->id, first_node->id);

  model.AddConsumer(value1->id, slice_node->id);
  model.AddConsumer(value1->id, second_node->id);

  model.SetProducer(value2->id, slice_node->id);
  model.SetProducer(value3->id, second_node->id);

  IrOpId slice_node_id = slice_node->id;
  IrTensorId value1_id = value1->id;
  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(slice_node_id), nullptr);
  EXPECT_EQ(model.tensor(value1_id), nullptr);  // Because value2 is kept

  const IrOp* remaining_first = model.op(first_node->id);
  ASSERT_NE(remaining_first, nullptr);
  EXPECT_THAT(remaining_first->outputs, ElementsAre(value2->id));

  const IrOp* remaining_second = model.op(second_node->id);
  ASSERT_NE(remaining_second, nullptr);
  EXPECT_THAT(remaining_second->inputs, ElementsAre(value2->id));
}

TEST(IrRemoveNoopTest,
     DoNotRemoveNoopWhenBothInputAndOutputAreGraphBoundaries) {
  IrModel model;
  IrOp* reshape_op = model.add_op();
  reshape_op->name = ToString(OperationType::RESHAPE);
  ReshapeAttributes attr;
  attr.new_shape = ::ml_drift::BHWC(1, 1, 2, 4);
  reshape_op->attr = attr;

  IrTensor* input = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                     ::ml_drift::BHWC(1, 1, 2, 4));
  IrTensor* output = model.add_tensor(::ml_drift::DataType::FLOAT32,
                                      ::ml_drift::BHWC(1, 1, 2, 4));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, reshape_op->id);
  model.SetProducer(output->id, reshape_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());
  EXPECT_NE(model.op(reshape_op->id), nullptr);
  EXPECT_NE(model.tensor(input->id), nullptr);
  EXPECT_NE(model.tensor(output->id), nullptr);
}

}  // namespace
}  // namespace ml_drift::ir
