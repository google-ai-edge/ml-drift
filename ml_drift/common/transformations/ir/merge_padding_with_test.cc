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

#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/transformations/ir/transform.h"

namespace ml_drift::ir {
namespace {

std::any GetSpatialAttr(OperationType op_type) {
  if (op_type == OperationType::CONVOLUTION_2D) {
    Convolution2DAttributes attr;
    attr.padding.appended = HW(0, 0);
    attr.padding.prepended = HW(0, 0);
    return attr;
  } else if (op_type == OperationType::DEPTHWISE_CONVOLUTION) {
    DepthwiseConvolution2DAttributes attr;
    attr.padding.appended = HW(0, 0);
    attr.padding.prepended = HW(0, 0);
    return attr;
  } else if (op_type == OperationType::POOLING_2D) {
    Pooling2DAttributes attr;
    attr.padding.appended = HW(0, 0);
    attr.padding.prepended = HW(0, 0);
    return attr;
  }
  return {};
}

Padding2D GetPadding(OperationType op_type, const std::any& attr) {
  if (op_type == OperationType::CONVOLUTION_2D) {
    return std::any_cast<Convolution2DAttributes>(attr).padding;
  } else if (op_type == OperationType::DEPTHWISE_CONVOLUTION) {
    return std::any_cast<DepthwiseConvolution2DAttributes>(attr).padding;
  } else if (op_type == OperationType::POOLING_2D) {
    return std::any_cast<Pooling2DAttributes>(attr).padding;
  }
  return Padding2D();
}

using MergePaddingSpatialParamTest = ::testing::TestWithParam<OperationType>;

TEST_P(MergePaddingSpatialParamTest, Smoke) {
  // Topology: input -> PAD -> temp -> SpatialOp -> output
  IrModel model;
  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* temp = model.add_tensor(DataType::FLOAT32, BHWC(1, 7, 7, 8));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 7, 7, 16));

  model.add_input(input->id);
  model.add_output(output->id);

  IrOp* pad_node = model.add_op();
  pad_node->name = ToString(OperationType::PAD);
  PadAttributes attr;
  attr.prepended = BHWC(0, 1, 1, 0);
  attr.appended = BHWC(0, 2, 2, 0);
  attr.type = PaddingContentType::ZEROS;
  pad_node->attr = attr;
  model.AddConsumer(input->id, pad_node->id);
  model.SetProducer(temp->id, pad_node->id);

  IrOp* spatial_node = model.add_op();
  spatial_node->name = ToString(GetParam());
  spatial_node->attr = GetSpatialAttr(GetParam());
  model.AddConsumer(temp->id, spatial_node->id);
  model.SetProducer(output->id, spatial_node->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(pad_node->id), nullptr);
  EXPECT_EQ(model.tensor(temp->id), nullptr);

  const IrOp* remaining_op = model.op(spatial_node->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_EQ(remaining_op->name, ToString(GetParam()));

  EXPECT_EQ(HW(1, 1), GetPadding(GetParam(), remaining_op->attr).prepended);
  EXPECT_EQ(HW(2, 2), GetPadding(GetParam(), remaining_op->attr).appended);
}

TEST_P(MergePaddingSpatialParamTest, MergeTwo) {
  // Topology: input -> PAD1 -> temp1 -> PAD2 -> temp2 -> SpatialOp -> output
  IrModel model;
  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* temp1 = model.add_tensor(DataType::FLOAT32, BHWC(1, 5, 5, 8));
  IrTensor* temp2 = model.add_tensor(DataType::FLOAT32, BHWC(1, 7, 7, 8));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 7, 7, 16));

  model.add_input(input->id);
  model.add_output(output->id);

  IrOp* pad_node1 = model.add_op();
  pad_node1->name = ToString(OperationType::PAD);
  PadAttributes attr;
  attr.prepended = BHWC(0, 1, 1, 0);
  attr.appended = BHWC(0, 0, 0, 0);
  attr.type = PaddingContentType::ZEROS;
  pad_node1->attr = attr;
  model.AddConsumer(input->id, pad_node1->id);
  model.SetProducer(temp1->id, pad_node1->id);

  IrOp* pad_node2 = model.add_op();
  pad_node2->name = ToString(OperationType::PAD);
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 2, 2, 0);
  pad_node2->attr = attr;
  model.AddConsumer(temp1->id, pad_node2->id);
  model.SetProducer(temp2->id, pad_node2->id);

  IrOp* spatial_node = model.add_op();
  spatial_node->name = ToString(GetParam());
  spatial_node->attr = GetSpatialAttr(GetParam());
  model.AddConsumer(temp2->id, spatial_node->id);
  model.SetProducer(output->id, spatial_node->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(pad_node1->id), nullptr);
  EXPECT_EQ(model.op(pad_node2->id), nullptr);
  EXPECT_EQ(model.tensor(temp1->id), nullptr);
  EXPECT_EQ(model.tensor(temp2->id), nullptr);

  const IrOp* remaining_op = model.op(spatial_node->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_EQ(remaining_op->name, ToString(GetParam()));

  EXPECT_EQ(HW(1, 1), GetPadding(GetParam(), remaining_op->attr).prepended);
  EXPECT_EQ(HW(2, 2), GetPadding(GetParam(), remaining_op->attr).appended);
}

TEST_P(MergePaddingSpatialParamTest, DoNotTrigger_ChannelPadding) {
  // Topology: input -> PAD (channel padding) -> temp -> SpatialOp -> output
  IrModel model;
  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* temp = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 12));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));

  model.add_input(input->id);
  model.add_output(output->id);

  IrOp* pad_node = model.add_op();
  pad_node->name = ToString(OperationType::PAD);
  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 0, 4);  // Channel padding
  attr.type = PaddingContentType::ZEROS;
  pad_node->attr = attr;
  model.AddConsumer(input->id, pad_node->id);
  model.SetProducer(temp->id, pad_node->id);

  IrOp* spatial_node = model.add_op();
  spatial_node->name = ToString(GetParam());
  spatial_node->attr = GetSpatialAttr(GetParam());
  model.AddConsumer(temp->id, spatial_node->id);
  model.SetProducer(output->id, spatial_node->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Should decline fusion because spatial ops only support HW padding
  EXPECT_NE(model.op(pad_node->id), nullptr);
  EXPECT_NE(model.tensor(temp->id), nullptr);
}

INSTANTIATE_TEST_SUITE_P(MergePaddingWithSpatial, MergePaddingSpatialParamTest,
                         ::testing::Values(OperationType::CONVOLUTION_2D,
                                           OperationType::DEPTHWISE_CONVOLUTION,
                                           OperationType::POOLING_2D));

TEST(MergePaddingWithAdd, MergeAlignedPadding) {
  // Topology: input0 -> PAD -> padded
  //           [padded, input1] -> ADD -> output
  IrModel model;
  IrTensor* input0 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* input1 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 40));
  IrTensor* padded = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 40));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 40));

  model.add_input(input0->id);
  model.add_input(input1->id);
  model.add_output(output->id);

  IrOp* pad_node = model.add_op();
  pad_node->name = ToString(OperationType::PAD);
  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 0, 0);
  pad_attr.appended = BHWC(0, 0, 0, 32);
  pad_attr.type = PaddingContentType::ZEROS;
  pad_node->attr = pad_attr;
  model.AddConsumer(input0->id, pad_node->id);
  model.SetProducer(padded->id, pad_node->id);

  IrOp* add_node = model.add_op();
  add_node->name = ToString(OperationType::ADD);
  ElementwiseAttributes add_attr;
  add_node->attr = add_attr;
  model.AddConsumer(padded->id, add_node->id);
  model.AddConsumer(input1->id, add_node->id);
  model.SetProducer(output->id, add_node->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_EQ(model.op(pad_node->id), nullptr);
  EXPECT_EQ(model.tensor(padded->id), nullptr);

  const IrOp* remaining_op = model.op(add_node->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_EQ(remaining_op->name, ToString(OperationType::ADD));
}

TEST(MergePaddingWithAdd, DoNotTrigger_AddWithAttributes) {
  // Topology: input0 -> PAD -> padded
  //           padded -> ADD (with constant attribute) -> output
  IrModel model;
  IrTensor* input0 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* padded = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 40));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 40));

  model.add_input(input0->id);
  model.add_output(output->id);

  IrOp* pad_node = model.add_op();
  pad_node->name = ToString(OperationType::PAD);
  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 0, 0);
  pad_attr.appended = BHWC(0, 0, 0, 32);
  pad_attr.type = PaddingContentType::ZEROS;
  pad_node->attr = pad_attr;
  model.AddConsumer(input0->id, pad_node->id);
  model.SetProducer(padded->id, pad_node->id);

  IrOp* add_node = model.add_op();
  add_node->name = ToString(OperationType::ADD);
  ElementwiseAttributes add_attr;
  add_attr.param = Tensor<Linear, DataType::FLOAT32>();
  add_node->attr = add_attr;
  model.AddConsumer(padded->id, add_node->id);
  // Missing the second input for ADD, so it implies constant param (which will
  // decline)
  model.SetProducer(output->id, add_node->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  EXPECT_NE(model.op(pad_node->id), nullptr);
  EXPECT_NE(model.tensor(padded->id), nullptr);
  EXPECT_NE(model.op(add_node->id), nullptr);
}

TEST(MergePaddingWithAdd, DoNotTrigger_SpatialPadding) {
  // Topology: input0 -> PAD (spatial padding) -> padded
  //           [padded, input1] -> ADD -> output
  IrModel model;
  IrTensor* input0 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* input1 = model.add_tensor(DataType::FLOAT32, BHWC(1, 5, 6, 8));
  IrTensor* padded = model.add_tensor(DataType::FLOAT32, BHWC(1, 5, 6, 8));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 5, 6, 8));

  model.add_input(input0->id);
  model.add_input(input1->id);
  model.add_output(output->id);

  IrOp* pad_node = model.add_op();
  pad_node->name = ToString(OperationType::PAD);
  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 0, 0);
  pad_attr.appended = BHWC(0, 1, 2, 0);  // HW padding instead of channel
  pad_attr.type = PaddingContentType::ZEROS;
  pad_node->attr = pad_attr;
  model.AddConsumer(input0->id, pad_node->id);
  model.SetProducer(padded->id, pad_node->id);

  IrOp* add_node = model.add_op();
  add_node->name = ToString(OperationType::ADD);
  ElementwiseAttributes add_attr;
  add_node->attr = add_attr;
  model.AddConsumer(padded->id, add_node->id);
  model.AddConsumer(input1->id, add_node->id);
  model.SetProducer(output->id, add_node->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Should decline because ADD padding only works for channels
  EXPECT_NE(model.op(pad_node->id), nullptr);
  EXPECT_NE(model.tensor(padded->id), nullptr);
}

}  // namespace
}  // namespace ml_drift::ir
