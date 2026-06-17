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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/transformations/fuse_add_to_conv.h"
#include "ml_drift/common/transformations/ir/transform.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace ml_drift::ir {
namespace {

std::any GetDefaultAttr(OperationType op_type) {
  if (op_type == OperationType::CONVOLUTION_2D) {
    Convolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    weights.shape = OHWI(16, 3, 2, 8);
    weights.data.resize(weights.shape.DimensionsProduct(), 0.1f);
    attr.bias.shape = Linear(16);
    attr.bias.data.resize(16, 1.1f);
    return attr;
  } else if (op_type == OperationType::DEPTHWISE_CONVOLUTION) {
    DepthwiseConvolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    weights.shape = OHWI(2, 3, 2, 8);
    weights.data.resize(weights.shape.DimensionsProduct(), 0.1f);
    attr.bias.shape = Linear(16);
    attr.bias.data.resize(16, 1.1f);
    return attr;
  } else if (op_type == OperationType::CONVOLUTION_TRANSPOSED) {
    ConvolutionTransposedAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.stride = HW(1, 1);
    attr.weights.shape = OHWI(16, 3, 2, 8);
    attr.weights.data.resize(attr.weights.shape.DimensionsProduct(), 0.1f);
    attr.bias.shape = Linear(16);
    attr.bias.data.resize(16, 1.1f);
    return attr;
  } else if (op_type == OperationType::FULLY_CONNECTED) {
    FullyConnectedAttributes attr;
    attr.weights.shape = OHWI(16, 1, 1, 8);
    attr.weights.data.resize(attr.weights.shape.DimensionsProduct(), 0.1f);
    attr.bias.shape = Linear(16);
    attr.bias.data.resize(16, 1.1f);
    return attr;
  }
  return {};
}

ElementwiseAttributes GetAddAttr() {
  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(16);
  add_tensor.data.resize(16, 0.3f);
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;
  return add_attr;
}

using FuseAddTopologyParamTest = ::testing::TestWithParam<OperationType>;

TEST_P(FuseAddTopologyParamTest, GemmThenAdd) {
  IrModel model;

  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* intermediate =
      model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));
  model.add_input(input->id);
  model.add_output(output->id);

  IrOp* main_op = model.add_op();
  main_op->name = ToString(GetParam());
  main_op->attr = GetDefaultAttr(GetParam());
  model.AddConsumer(input->id, main_op->id);
  model.SetProducer(intermediate->id, main_op->id);

  IrOp* add_op = model.add_op();
  add_op->name = ToString(OperationType::ADD);
  add_op->attr = GetAddAttr();
  model.AddConsumer(intermediate->id, add_op->id);
  model.SetProducer(output->id, add_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Add should be removed
  EXPECT_EQ(model.op(add_op->id), nullptr);
  EXPECT_EQ(model.tensor(intermediate->id), nullptr);

  // Main op should remain and point to output
  const IrOp* remaining_op = model.op(main_op->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_EQ(remaining_op->name, ToString(GetParam()));
}

TEST_P(FuseAddTopologyParamTest, AbsorbProducer) {
  IrModel model;

  ElementwiseAttributes add_attr;
  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(8);  // Input channels is 8
  add_tensor.data.resize(8, 0.3f);
  add_attr.param = add_tensor;

  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* intermediate =
      model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));
  model.add_input(input->id);
  model.add_output(output->id);

  IrOp* add_op = model.add_op();
  add_op->name = ToString(OperationType::ADD);
  add_op->attr = add_attr;
  model.AddConsumer(input->id, add_op->id);
  model.SetProducer(intermediate->id, add_op->id);

  IrOp* main_op = model.add_op();
  main_op->name = ToString(GetParam());
  main_op->attr = GetDefaultAttr(GetParam());
  model.AddConsumer(intermediate->id, main_op->id);
  model.SetProducer(output->id, main_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  if (GetParam() == OperationType::CONVOLUTION_2D) {
    // Add should be removed
    EXPECT_EQ(model.op(add_op->id), nullptr);
    EXPECT_EQ(model.tensor(intermediate->id), nullptr);
  } else {
    // Not supported for other operations, should remain
    EXPECT_NE(model.op(add_op->id), nullptr);
    EXPECT_NE(model.tensor(intermediate->id), nullptr);
  }

  // Main op should remain
  const IrOp* remaining_op = model.op(main_op->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_EQ(remaining_op->name, ToString(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    FuseAddToConvTest, FuseAddTopologyParamTest,
    ::testing::Values(OperationType::CONVOLUTION_2D,
                      OperationType::DEPTHWISE_CONVOLUTION,
                      OperationType::CONVOLUTION_TRANSPOSED,
                      OperationType::FULLY_CONNECTED));

TEST(IrFuseAddToGemmTest, ComplexChainedTopology) {
  // input -> Conv2D -> Add -> Add -> Output
  IrModel model;

  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* inter1 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));
  IrTensor* inter2 = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));
  model.add_input(input->id);
  model.add_output(output->id);

  IrOp* conv = model.add_op();
  conv->name = ToString(OperationType::CONVOLUTION_2D);
  conv->attr = GetDefaultAttr(OperationType::CONVOLUTION_2D);
  model.AddConsumer(input->id, conv->id);
  model.SetProducer(inter1->id, conv->id);

  IrOp* add1 = model.add_op();
  add1->name = ToString(OperationType::ADD);
  add1->attr = GetAddAttr();
  model.AddConsumer(inter1->id, add1->id);
  model.SetProducer(inter2->id, add1->id);

  IrOp* add2 = model.add_op();
  add2->name = ToString(OperationType::ADD);
  add2->attr = GetAddAttr();
  model.AddConsumer(inter2->id, add2->id);
  model.SetProducer(output->id, add2->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Both Adds should be fused away
  EXPECT_EQ(model.op(add1->id), nullptr);
  EXPECT_EQ(model.op(add2->id), nullptr);
  EXPECT_EQ(model.tensor(inter1->id), nullptr);
  EXPECT_EQ(model.tensor(inter2->id), nullptr);

  const IrOp* remaining_op = model.op(conv->id);
  ASSERT_NE(remaining_op, nullptr);

  // The bias should be updated by both adds: 1.1 + 0.3 + 0.3 = 1.7
  auto conv_attr = std::any_cast<Convolution2DAttributes>(remaining_op->attr);
  EXPECT_THAT(conv_attr.bias.data,
              Pointwise(FloatNear(1e-5), std::vector<float>(16, 1.7f)));
}

// Retain the specific math-validation tests (they test the underlying
// Elementwise helpers directly)
TEST(FuseAddAfterConvolution2DTest, MathVerification) {
  Convolution2DAttributes attr;
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 2, 2);
  attr_weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {0.3f, 0.7f};
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;

  FuseConvolution2DWithAdd(add_attr, &attr);

  EXPECT_THAT(attr_weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.4f, 1.9f}));
}

TEST(FuseAddAfterDepthwiseConvolution2DTest, MathVerification) {
  DepthwiseConvolution2DAttributes attr;
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 2, 2);
  attr_weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1.1f, 1.2f, 1.3f, 1.4f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(4);
  add_tensor.data = {0.3f, 0.7f, 0.5f, 0.1f};
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;

  FuseDepthwiseConvolution2DWithAdd(add_attr, &attr);

  EXPECT_THAT(attr_weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data,
              Pointwise(FloatNear(1e-6), {1.4f, 1.9f, 1.8f, 1.5f}));
}

TEST(FuseAddAfterConvolutionTransposedTest, MathVerification) {
  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {0.3f, 0.7f};
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;

  FuseConvolutionTransposedWithAdd(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.4f, 1.9f}));
}

TEST(FuseAddAfterFullyConnectedTest, MathVerification) {
  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {0.3f, 0.7f};
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;

  FuseFullyConnectedWithAdd(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6), {0.1f, 0.2f, 0.3f, 0.4f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.4f, 1.9f}));
}

}  // namespace
}  // namespace ml_drift::ir
