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

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/transformations/ir/transform.h"

namespace ml_drift::ir {
namespace {

TEST(AddQuantAdjustmentsTest, OneNode) {
  IrModel model;
  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  input->quant_params = {.min = 0.0, .max = 1.0, .scale = 0.004};

  IrOp* add_node = model.add_op();
  add_node->name = ToString(OperationType::ADD);
  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(8);
  add_tensor.data.resize(8);
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;
  add_node->attr = add_attr;

  model.AddConsumer(input->id, add_node->id);

  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  output->quant_params = {.min = 0.0, .max = 2.0, .scale = 0.004};
  model.SetProducer(output->id, add_node->id);
  model.add_output(output->id);

  EXPECT_EQ(1, model.ops().size());
  EXPECT_EQ(2, model.tensors().size());

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Since output has no consumers, no new node should be added
  EXPECT_EQ(1, model.ops().size());
  EXPECT_EQ(2, model.tensors().size());
}

TEST(AddQuantAdjustmentsTest, GeneralCase) {
  IrModel model;
  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  input->quant_params = {.min = 0.0, .max = 1.0, .scale = 0.004};
  model.add_input(input->id);

  // First Add.
  IrOp* add1_node = model.add_op();
  add1_node->name = ToString(OperationType::ADD);
  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(8);
  add_tensor.data.resize(8);
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;
  add1_node->attr = add_attr;

  // QuantizeAndDequantize.
  IrOp* quant_node = model.add_op();
  quant_node->name = ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
  QuantizeAndDequantizeAttributes quant_attr;
  quant_attr.min = -1.0;
  quant_attr.max = 1.0;
  quant_attr.scale = 0.008;
  quant_node->attr = quant_attr;

  // Second Add.
  IrOp* add2_node = model.add_op();
  add2_node->name = ToString(OperationType::ADD);
  add2_node->attr = add_attr;

  // Connections.
  model.AddConsumer(input->id, add1_node->id);

  IrTensor* link1 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  link1->quant_params = {.min = 0.0, .max = 2.0, .scale = 0.008};
  model.SetProducer(link1->id, add1_node->id);
  model.AddConsumer(link1->id, quant_node->id);
  model.AddConsumer(link1->id, add2_node->id);

  IrTensor* link2 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  link2->quant_params = {.min = -1.0, .max = 1.0, .scale = 0.008};
  model.SetProducer(link2->id, quant_node->id);
  model.AddConsumer(link2->id, add2_node->id);

  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  output->quant_params = {.min = -1.0, .max = 1.0, .scale = 0.008};
  model.SetProducer(output->id, add2_node->id);
  model.add_output(output->id);

  EXPECT_EQ(3, model.ops().size());
  EXPECT_EQ(4, model.tensors().size());

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // 1 new qdq op, 1 new tensor
  EXPECT_EQ(4, model.ops().size());
  EXPECT_EQ(5, model.tensors().size());

  // The new node should be appended at the end (index 3).
  const IrOp* new_qdq_op = model.op(3);
  ASSERT_NE(new_qdq_op, nullptr);
  EXPECT_EQ(new_qdq_op->name, ToString(OperationType::QUANTIZE_AND_DEQUANTIZE));

  auto new_quant_attr =
      std::any_cast<QuantizeAndDequantizeAttributes>(new_qdq_op->attr);
  EXPECT_EQ(0.0, new_quant_attr.min);
  EXPECT_EQ(2.0, new_quant_attr.max);
  EXPECT_EQ(0.008f, new_quant_attr.scale);

  // The newly created tensor should be the 5th tensor (index 4).
  const IrTensor* adjusted_tensor = model.tensor(4);
  ASSERT_NE(adjusted_tensor, nullptr);
  EXPECT_EQ(adjusted_tensor->producer, new_qdq_op->id);

  ASSERT_TRUE(adjusted_tensor->quant_params.has_value());
  EXPECT_EQ(0.0, adjusted_tensor->quant_params->min);
  EXPECT_EQ(2.0, adjusted_tensor->quant_params->max);
  EXPECT_EQ(0.008f, adjusted_tensor->quant_params->scale);

  // It should have taken over consumers of link1 (which were quant_node and
  // add2_node).
  EXPECT_EQ(2, adjusted_tensor->consumers.size());
  EXPECT_TRUE(adjusted_tensor->consumers.contains(quant_node->id));
  EXPECT_TRUE(adjusted_tensor->consumers.contains(add2_node->id));

  // Transformation should be idempotent.
  EXPECT_TRUE(TransformIrModel(&model).ok());
  EXPECT_EQ(4, model.ops().size());
  EXPECT_EQ(5, model.tensors().size());
}

}  // namespace
}  // namespace ml_drift::ir
