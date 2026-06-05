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
#include "ml_drift/common/default/status_matchers.h"
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

}  // namespace
}  // namespace ml_drift::ir
