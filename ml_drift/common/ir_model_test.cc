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

#include "ml_drift/common/ir_model.h"

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"

namespace ml_drift::ir {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::PrintToString;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

MATCHER_P(UniquePtr, expected_ptr,
          std::string(negation ? "isn't" : "is") +
              " a unique_ptr pointing to " + PrintToString(expected_ptr)) {
  return arg.get() == expected_ptr;
}

TEST(IrModelTest, SingleNode1Input1Output) {
  // input -> node -> output
  IrModel model;
  const IrOp* op = model.add_op();
  const IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  model.add_input(input->id);
  const IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  model.add_output(output->id);
  model.AddConsumer(input->id, op->id);
  model.SetProducer(output->id, op->id);
  // Check model state.
  EXPECT_THAT(model.ops(), ElementsAre(UniquePtr(op)));
  EXPECT_THAT(model.tensors(),
              UnorderedElementsAre(UniquePtr(input), UniquePtr(output)));
  EXPECT_THAT(model.inputs(), ElementsAre(input->id));
  EXPECT_THAT(model.outputs(), ElementsAre(output->id));
  // Check op's inputs & outputs.
  {
    const auto* model_op = model.op(op->id);
    ASSERT_NE(model_op, nullptr);
    EXPECT_THAT(model_op->inputs, ElementsAre(input->id));
    EXPECT_THAT(model_op->outputs, ElementsAre(output->id));
  }
  // Check tensors' producers & consumers.
  {
    const auto* model_tensor = model.tensor(input->id);
    ASSERT_NE(model_tensor, nullptr);
    EXPECT_EQ(model_tensor->producer, std::nullopt);
    EXPECT_THAT(model_tensor->consumers, UnorderedElementsAre(op->id));
  }
  {
    const auto* model_tensor = model.tensor(output->id);
    ASSERT_NE(model_tensor, nullptr);
    EXPECT_EQ(model_tensor->producer, op->id);
    EXPECT_THAT(model_tensor->consumers, IsEmpty());
  }
}

TEST(IrModelTest, SingleNode1Input2Outputs) {
  // input -> node -> (output1, output2)
  IrModel model;
  const IrOp* op = model.add_op();
  const IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  model.add_input(input->id);
  const IrTensor* output1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  model.add_output(output1->id);
  const IrTensor* output2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  model.add_output(output2->id);
  model.AddConsumer(input->id, op->id);
  model.SetProducer(output1->id, op->id);
  model.SetProducer(output2->id, op->id);
  // Check model state.
  EXPECT_THAT(model.ops(), ElementsAre(UniquePtr(op)));
  EXPECT_THAT(model.tensors(),
              UnorderedElementsAre(UniquePtr(input), UniquePtr(output1),
                                   UniquePtr(output2)));
  EXPECT_THAT(model.inputs(), UnorderedElementsAre(input->id));
  EXPECT_THAT(model.outputs(), UnorderedElementsAre(output1->id, output2->id));
  // Check op's inputs & outputs.
  {
    const auto* model_op = model.op(op->id);
    ASSERT_NE(model_op, nullptr);
    EXPECT_THAT(model_op->inputs, ElementsAre(input->id));
    EXPECT_THAT(model_op->outputs, ElementsAre(output1->id, output2->id));
  }
  // Check tensors' producers & consumers.
  {
    const auto* model_tensor = model.tensor(input->id);
    ASSERT_NE(model_tensor, nullptr);
    EXPECT_EQ(model_tensor->producer, std::nullopt);
    EXPECT_THAT(model_tensor->consumers, UnorderedElementsAre(op->id));
  }
  {
    const auto* model_tensor = model.tensor(output1->id);
    ASSERT_NE(model_tensor, nullptr);
    EXPECT_EQ(model_tensor->producer, op->id);
    EXPECT_THAT(model_tensor->consumers, IsEmpty());
  }
  {
    const auto* model_tensor = model.tensor(output2->id);
    ASSERT_NE(model_tensor, nullptr);
    EXPECT_EQ(model_tensor->producer, op->id);
    EXPECT_THAT(model_tensor->consumers, IsEmpty());
  }
}

TEST(IrModelTest, RemoveSimpleOp_KeepInput) {
  // input -> op1 -> intermediate -> op2 -> output
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* op2 = model.add_op();

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

  model.AddConsumer(intermediate->id, op2->id);
  model.SetProducer(output->id, op2->id);

  // We remove op1, which is not producing a graph output.
  // We expect intermediate tensor to be removed, and op2 should consume input
  // instead.
  IrOpId op1_id = op1->id;
  IrTensorId intermediate_id = intermediate->id;
  EXPECT_TRUE(model.RemoveSimpleOp(op1_id).ok());

  EXPECT_EQ(model.op(op1_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate_id), nullptr);

  const IrOp* remaining_op = model.op(op2->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_THAT(remaining_op->inputs, ElementsAre(input->id));

  const IrTensor* remaining_input = model.tensor(input->id);
  ASSERT_NE(remaining_input, nullptr);
  EXPECT_THAT(remaining_input->consumers, UnorderedElementsAre(op2->id));
}

TEST(IrModelTest, RemoveSimpleOp_KeepOutput) {
  // input -> op1 -> intermediate -> op2 -> output
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* op2 = model.add_op();

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

  model.AddConsumer(intermediate->id, op2->id);
  model.SetProducer(output->id, op2->id);

  // We remove op2, which produces a graph output.
  // We expect intermediate tensor to be removed, and op1 should produce output
  // instead.
  IrOpId op2_id = op2->id;
  IrTensorId intermediate_id = intermediate->id;
  EXPECT_TRUE(model.RemoveSimpleOp(op2_id).ok());

  EXPECT_EQ(model.op(op2_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate_id), nullptr);

  const IrOp* remaining_op = model.op(op1->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_THAT(remaining_op->outputs, ElementsAre(output->id));

  const IrTensor* remaining_output = model.tensor(output->id);
  ASSERT_NE(remaining_output, nullptr);
  EXPECT_EQ(remaining_output->producer, op1->id);
}

TEST(IrModelTest, RemoveSimpleOp_FailsWhenBothAreGraphBoundaries) {
  // input -> op -> output
  IrModel model;
  IrOp* op = model.add_op();

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op->id);
  model.SetProducer(output->id, op->id);

  // We try to remove op, which consumes a graph input and produces a graph
  // output.
  EXPECT_FALSE(model.RemoveSimpleOp(op->id).ok());
}

TEST(IrModelTest,
     RemoveSimpleOp_KeepOutput_MultipleConsumers_RewiresConsumers) {
  // input -> op1 -> intermediate -> op2 -> output
  //                      |
  //                      -> op3 -> output2
  // When we remove op2, this turns into:
  // input -> op1 -> output -> op3 -> output2
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* op2 = model.add_op();
  IrOp* op3 = model.add_op();

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);
  model.add_output(output2->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate->id, op1->id);

  model.AddConsumer(intermediate->id, op2->id);
  model.SetProducer(output->id, op2->id);

  model.AddConsumer(intermediate->id, op3->id);
  model.SetProducer(output2->id, op3->id);

  // We remove op2, which produces a graph output.
  // This deletes 'intermediate', rewires op1 to produce 'output', and rewires
  // op3 to consume 'output' instead of 'intermediate'.
  IrOpId op2_id = op2->id;
  IrTensorId intermediate_id = intermediate->id;
  EXPECT_TRUE(model.RemoveSimpleOp(op2_id).ok());

  EXPECT_EQ(model.op(op2_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate_id), nullptr);

  const IrOp* remaining_op1 = model.op(op1->id);
  ASSERT_NE(remaining_op1, nullptr);
  EXPECT_THAT(remaining_op1->outputs, ElementsAre(output->id));

  const IrTensor* remaining_output = model.tensor(output->id);
  ASSERT_NE(remaining_output, nullptr);
  EXPECT_EQ(remaining_output->producer, op1->id);
  EXPECT_THAT(remaining_output->consumers, UnorderedElementsAre(op3->id));

  const IrOp* remaining_op3 = model.op(op3->id);
  ASSERT_NE(remaining_op3, nullptr);
  EXPECT_THAT(remaining_op3->inputs, ElementsAre(output->id));
}

TEST(IrModelTest, RemoveSimpleOp_FailsInvalidOpId) {
  IrModel model;
  EXPECT_FALSE(model.RemoveSimpleOp(0).ok());
  EXPECT_FALSE(model.RemoveSimpleOp(999).ok());
}

TEST(IrModelTest, RemoveSimpleOp_FailsInvalidNumInputsOutputs) {
  IrModel model;
  IrOp* op_no_io = model.add_op();

  IrOp* op_2_in = model.add_op();
  IrTensor* in1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* in2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* out1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  model.AddConsumer(in1->id, op_2_in->id);
  model.AddConsumer(in2->id, op_2_in->id);
  model.SetProducer(out1->id, op_2_in->id);

  EXPECT_FALSE(model.RemoveSimpleOp(op_no_io->id).ok());
  EXPECT_FALSE(model.RemoveSimpleOp(op_2_in->id).ok());
}

TEST(IrModelTest, RemoveSimpleOp_FailsInputHasNoProducerWhenKeepingOutput) {
  // dangling_input -> op -> output
  IrModel model;
  IrOp* op = model.add_op();

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_output(output->id);

  model.AddConsumer(input->id, op->id);
  model.SetProducer(output->id, op->id);

  // We try to remove op, which produces a graph output, but its input has no
  // producer. keep_input is false, keep_output is true.
  EXPECT_FALSE(model.RemoveSimpleOp(op->id).ok());
}

TEST(IrModelTest, RemoveSimpleOp_IntermediateNode) {
  // input -> op1 -> intermediate1 -> op2 -> intermediate2 -> op3 -> output
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* op2 = model.add_op();
  IrOp* op3 = model.add_op();

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate1->id, op1->id);

  model.AddConsumer(intermediate1->id, op2->id);
  model.SetProducer(intermediate2->id, op2->id);

  model.AddConsumer(intermediate2->id, op3->id);
  model.SetProducer(output->id, op3->id);

  // We remove op2, which is an intermediate node (keep_input=false,
  // keep_output=false). We expect intermediate2 to be removed, and op3 should
  // consume intermediate1.
  IrOpId op2_id = op2->id;
  IrTensorId intermediate2_id = intermediate2->id;
  EXPECT_TRUE(model.RemoveSimpleOp(op2_id).ok());

  EXPECT_EQ(model.op(op2_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate2_id), nullptr);

  const IrOp* remaining_op3 = model.op(op3->id);
  ASSERT_NE(remaining_op3, nullptr);
  EXPECT_THAT(remaining_op3->inputs, ElementsAre(intermediate1->id));

  const IrTensor* remaining_intermediate1 = model.tensor(intermediate1->id);
  ASSERT_NE(remaining_intermediate1, nullptr);
  EXPECT_THAT(remaining_intermediate1->consumers,
              UnorderedElementsAre(op3->id));
}

TEST(IrModelTest, RemoveSimpleOp_IntermediateNode_MultipleConsumers) {
  // input -> op1 -> intermediate1 -> op2 -> intermediate2 -> op3 -> output1
  //                                                      \-> op4 -> output2
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* op2 = model.add_op();
  IrOp* op3 = model.add_op();
  IrOp* op4 = model.add_op();

  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* intermediate2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output1->id);
  model.add_output(output2->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(intermediate1->id, op1->id);

  model.AddConsumer(intermediate1->id, op2->id);
  model.SetProducer(intermediate2->id, op2->id);

  model.AddConsumer(intermediate2->id, op3->id);
  model.SetProducer(output1->id, op3->id);

  model.AddConsumer(intermediate2->id, op4->id);
  model.SetProducer(output2->id, op4->id);

  // We remove op2, which is an intermediate node (keep_input=false,
  // keep_output=false). We expect intermediate2 to be removed, and both op3 and
  // op4 should consume intermediate1.
  IrOpId op2_id = op2->id;
  IrTensorId intermediate2_id = intermediate2->id;
  EXPECT_TRUE(model.RemoveSimpleOp(op2_id).ok());

  EXPECT_EQ(model.op(op2_id), nullptr);
  EXPECT_EQ(model.tensor(intermediate2_id), nullptr);

  const IrOp* remaining_op3 = model.op(op3->id);
  ASSERT_NE(remaining_op3, nullptr);
  EXPECT_THAT(remaining_op3->inputs, ElementsAre(intermediate1->id));

  const IrOp* remaining_op4 = model.op(op4->id);
  ASSERT_NE(remaining_op4, nullptr);
  EXPECT_THAT(remaining_op4->inputs, ElementsAre(intermediate1->id));

  const IrTensor* remaining_intermediate1 = model.tensor(intermediate1->id);
  ASSERT_NE(remaining_intermediate1, nullptr);
  EXPECT_THAT(remaining_intermediate1->consumers,
              UnorderedElementsAre(op3->id, op4->id));
}

TEST(IrModelTest, ReplaceInput_Success) {
  IrModel model;
  IrOp* op = model.add_op();
  IrTensor* old_tensor =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* new_tensor =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.AddConsumer(old_tensor->id, op->id);

  EXPECT_THAT(op->inputs, ElementsAre(old_tensor->id));
  EXPECT_THAT(old_tensor->consumers, UnorderedElementsAre(op->id));
  EXPECT_THAT(new_tensor->consumers, IsEmpty());

  EXPECT_TRUE(model.ReplaceInput(op->id, old_tensor->id, new_tensor->id).ok());

  EXPECT_THAT(op->inputs, ElementsAre(new_tensor->id));
  EXPECT_THAT(old_tensor->consumers, IsEmpty());
  EXPECT_THAT(new_tensor->consumers, UnorderedElementsAre(op->id));
}

TEST(IrModelTest, ReplaceInput_Errors) {
  IrModel model;
  IrOp* op = model.add_op();
  IrTensor* t1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* t2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.AddConsumer(t1->id, op->id);

  // Invalid op ID
  EXPECT_FALSE(model.ReplaceInput(999, t1->id, t2->id).ok());

  // Invalid old tensor ID
  EXPECT_FALSE(model.ReplaceInput(op->id, 999, t2->id).ok());

  // Invalid new tensor ID
  EXPECT_FALSE(model.ReplaceInput(op->id, t1->id, 999).ok());

  // Old tensor not in op's inputs
  EXPECT_FALSE(model.ReplaceInput(op->id, t2->id, t1->id).ok());
}

TEST(IrModelTest, RemoveOp_Success) {
  IrModel model;
  IrOp* op = model.add_op();
  IrTensor* in_tensor =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* out_tensor =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.AddConsumer(in_tensor->id, op->id);
  model.SetProducer(out_tensor->id, op->id);

  EXPECT_THAT(in_tensor->consumers, UnorderedElementsAre(op->id));
  EXPECT_EQ(out_tensor->producer, op->id);
  EXPECT_NE(model.op(op->id), nullptr);

  IrOpId op_id = op->id;
  IrTensorId in_tensor_id = in_tensor->id;
  IrTensorId out_tensor_id = out_tensor->id;

  // When op is removed, since both in_tensor and out_tensor have no other
  // producer/consumers, they are automatically orphaned and deleted.
  EXPECT_TRUE(model.RemoveOp(op_id).ok());
  EXPECT_EQ(model.op(op_id), nullptr);
  EXPECT_EQ(model.tensor(in_tensor_id), nullptr);
  EXPECT_EQ(model.tensor(out_tensor_id), nullptr);
}

TEST(IrModelTest, RemoveOp_BoundaryPromotion) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* op2 = model.add_op();
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
  model.AddConsumer(intermediate->id, op2->id);
  model.SetProducer(output->id, op2->id);

  IrOpId op2_id = op2->id;
  IrTensorId output_id = output->id;
  IrTensorId intermediate_id = intermediate->id;

  // Removing op2: intermediate loses its consumer but keeps producer op1,
  // so it should be promoted to a graph output.
  // output loses its producer and has no consumers, so it should be deleted.
  EXPECT_TRUE(model.RemoveOp(op2_id).ok());
  EXPECT_EQ(model.op(op2_id), nullptr);
  EXPECT_EQ(model.tensor(output_id), nullptr);
  EXPECT_FALSE(model.IsGraphOutput(output_id));
  EXPECT_TRUE(model.IsGraphOutput(intermediate_id));
}

TEST(IrModelTest, RemoveOp_SequentialWithOrphanPruning) {
  IrModel model;
  IrOp* op1 = model.add_op();
  IrOp* op2 = model.add_op();
  IrOp* op3 = model.add_op();
  IrTensor* input =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* t1 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* t2 =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));
  IrTensor* output =
      model.add_tensor(::ml_drift::DataType::FLOAT32, ::ml_drift::HWC(1, 1, 1));

  model.add_input(input->id);
  model.add_output(output->id);

  model.AddConsumer(input->id, op1->id);
  model.SetProducer(t1->id, op1->id);
  model.AddConsumer(t1->id, op2->id);
  model.SetProducer(t2->id, op2->id);
  model.AddConsumer(t2->id, op3->id);
  model.SetProducer(output->id, op3->id);

  IrOpId op2_id = op2->id;
  IrOpId op3_id = op3->id;
  IrTensorId t1_id = t1->id;
  IrTensorId t2_id = t2->id;
  IrTensorId output_id = output->id;

  // Sequentially remove op2 then op3:
  // t1 is kept and promoted to output because it's produced by op1.
  // t2 was between op2 and op3; it is orphaned and automatically deleted.
  // output is orphaned and deleted.
  EXPECT_TRUE(model.RemoveOp(op2_id).ok());
  EXPECT_TRUE(model.RemoveOp(op3_id).ok());

  EXPECT_EQ(model.op(op2_id), nullptr);
  EXPECT_EQ(model.op(op3_id), nullptr);
  EXPECT_EQ(model.tensor(t2_id), nullptr);
  EXPECT_EQ(model.tensor(output_id), nullptr);
  EXPECT_TRUE(model.IsGraphOutput(t1_id));
}

TEST(IrModelTest, RemoveOp_InvalidOpId) {
  IrModel model;
  EXPECT_FALSE(model.RemoveOp(999).ok());
}

}  // namespace
}  // namespace ml_drift::ir
