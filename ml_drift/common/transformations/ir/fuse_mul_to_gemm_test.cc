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
#include "ml_drift/common/transformations/fuse_mul_to_conv.h"
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
    attr.bias.data.resize(16, 1.5f);
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
    attr.bias.data.resize(16, 1.5f);
    return attr;
  } else if (op_type == OperationType::CONVOLUTION_TRANSPOSED) {
    ConvolutionTransposedAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.stride = HW(1, 1);
    attr.weights.shape = OHWI(16, 3, 2, 8);
    attr.weights.data.resize(attr.weights.shape.DimensionsProduct(), 0.1f);
    attr.bias.shape = Linear(16);
    attr.bias.data.resize(16, 1.5f);
    return attr;
  } else if (op_type == OperationType::FULLY_CONNECTED) {
    FullyConnectedAttributes attr;
    attr.weights.shape = OHWI(16, 1, 1, 8);
    attr.weights.data.resize(attr.weights.shape.DimensionsProduct(), 0.1f);
    attr.bias.shape = Linear(16);
    attr.bias.data.resize(16, 1.5f);
    return attr;
  }
  return {};
}

ElementwiseAttributes GetMulAttr() {
  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(16);
  mul_tensor.data.resize(16, 0.5f);
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;
  return mul_attr;
}

using FuseMulTopologyParamTest = ::testing::TestWithParam<OperationType>;

TEST_P(FuseMulTopologyParamTest, GemmThenMul) {
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

  IrOp* mul_op = model.add_op();
  mul_op->name = ToString(OperationType::MUL);
  mul_op->attr = GetMulAttr();
  model.AddConsumer(intermediate->id, mul_op->id);
  model.SetProducer(output->id, mul_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Mul should be removed
  EXPECT_EQ(model.op(mul_op->id), nullptr);
  EXPECT_EQ(model.tensor(intermediate->id), nullptr);

  // Main op should remain and point to output
  const IrOp* remaining_op = model.op(main_op->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_EQ(remaining_op->name, ToString(GetParam()));
}

TEST_P(FuseMulTopologyParamTest, AbsorbProducer) {
  IrModel model;

  ElementwiseAttributes mul_attr;
  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(8);  // Input channels is 8
  mul_tensor.data.resize(8, 0.5f);
  mul_attr.param = mul_tensor;

  IrTensor* input = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* intermediate =
      model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 8));
  IrTensor* output = model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 16));
  model.add_input(input->id);
  model.add_output(output->id);

  IrOp* mul_op = model.add_op();
  mul_op->name = ToString(OperationType::MUL);
  mul_op->attr = mul_attr;
  model.AddConsumer(input->id, mul_op->id);
  model.SetProducer(intermediate->id, mul_op->id);

  IrOp* main_op = model.add_op();
  main_op->name = ToString(GetParam());
  main_op->attr = GetDefaultAttr(GetParam());
  model.AddConsumer(intermediate->id, main_op->id);
  model.SetProducer(output->id, main_op->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Mul should be removed
  EXPECT_EQ(model.op(mul_op->id), nullptr);
  EXPECT_EQ(model.tensor(intermediate->id), nullptr);

  // Main op should remain
  const IrOp* remaining_op = model.op(main_op->id);
  ASSERT_NE(remaining_op, nullptr);
  EXPECT_EQ(remaining_op->name, ToString(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    FuseMulToConvTest, FuseMulTopologyParamTest,
    ::testing::Values(OperationType::CONVOLUTION_2D,
                      OperationType::DEPTHWISE_CONVOLUTION,
                      OperationType::CONVOLUTION_TRANSPOSED,
                      OperationType::FULLY_CONNECTED));

TEST(IrFuseMulToGemmTest, ComplexChainedTopology) {
  // input -> Conv2D -> Mul -> Mul -> Output
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

  IrOp* mul1 = model.add_op();
  mul1->name = ToString(OperationType::MUL);
  mul1->attr = GetMulAttr();
  model.AddConsumer(inter1->id, mul1->id);
  model.SetProducer(inter2->id, mul1->id);

  IrOp* mul2 = model.add_op();
  mul2->name = ToString(OperationType::MUL);
  mul2->attr = GetMulAttr();
  model.AddConsumer(inter2->id, mul2->id);
  model.SetProducer(output->id, mul2->id);

  EXPECT_TRUE(TransformIrModel(&model).ok());

  // Both Muls should be fused away
  EXPECT_EQ(model.op(mul1->id), nullptr);
  EXPECT_EQ(model.op(mul2->id), nullptr);
  EXPECT_EQ(model.tensor(inter1->id), nullptr);
  EXPECT_EQ(model.tensor(inter2->id), nullptr);

  const IrOp* remaining_op = model.op(conv->id);
  ASSERT_NE(remaining_op, nullptr);

  // The weights and bias should be updated by both muls: (0.1 * 0.5 * 0.5) and
  // (1.5 * 0.5 * 0.5)
  auto conv_attr = std::any_cast<Convolution2DAttributes>(remaining_op->attr);
  EXPECT_THAT(conv_attr.bias.data,
              Pointwise(FloatNear(1e-5), std::vector<float>(16, 0.375f)));
  auto& weights = std::get<Tensor<OHWI, DataType::FLOAT32>>(conv_attr.weights);
  EXPECT_THAT(
      weights.data,
      Pointwise(FloatNear(1e-5),
                std::vector<float>(weights.shape.DimensionsProduct(), 0.025f)));
}

// Retain the specific math-validation tests (they test the underlying
// Elementwise helpers directly)
TEST(FuseMulAfterConvolution2DTest, MathVerification) {
  Convolution2DAttributes attr;
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 2, 2);
  attr_weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseConvolution2DWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr_weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.1f, 0.15f, 0.2f, 1.0f, 1.2f, 1.4f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {0.75f, 5.0f}));
}

TEST(FuseMulAfterDepthwiseConvolution2DTest, MathVerification) {
  DepthwiseConvolution2DAttributes attr;
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 2, 2);
  attr_weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1.5f, 2.5f, 1.0f, 2.0f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(4);
  mul_tensor.data = {0.5f, 2.0f, 4.0f, 0.25f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseDepthwiseConvolution2DWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr_weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.8f, 0.15f, 1.6f, 1.0f, 0.15f, 1.4f, 0.2f}));
  EXPECT_THAT(attr.bias.data,
              Pointwise(FloatNear(1e-6), {0.75f, 5.0f, 4.0f, 0.5f}));
}

TEST(FuseMulAfterConvolutionTransposedTest, MathVerification) {
  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseConvolutionTransposedWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.1f, 0.15f, 0.2f, 1.0f, 1.2f, 1.4f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {0.75f, 5.0f}));
}

TEST(FuseMulAfterFullyConnectedTest, MathVerification) {
  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseFullyConnectedWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6), {0.05f, 0.1f, 0.6f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {0.75f, 5.0f}));
}

TEST(FuseMulBeforeConvolution2DTest, MathVerification) {
  Convolution2DAttributes attr;
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 2, 2);
  attr_weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithConvolution2D(mul_attr, &attr);

  EXPECT_THAT(attr_weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.4f, 0.15f, 0.8f, 0.25f, 1.2f, 0.35f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.5f, 2.5f}));
}

TEST(FuseMulBeforeDepthwiseConvolution2DTest, MathVerification) {
  DepthwiseConvolution2DAttributes attr;
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 2, 2);
  attr_weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1.5f, 2.5f, 1.0f, 2.0f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(4);
  mul_tensor.data = {0.5f, 2.0f, 4.0f, 0.25f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithDepthwiseConvolution2D(mul_attr, &attr);

  EXPECT_THAT(attr_weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.4f, 0.15f, 0.8f, 0.25f, 1.2f, 0.35f, 1.6f}));
  EXPECT_THAT(attr.bias.data,
              Pointwise(FloatNear(1e-6), {1.5f, 2.5f, 1.0f, 2.0f}));
}

TEST(FuseMulBeforeConvolutionTransposedTest, MathVerification) {
  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithConvolutionTransposed(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.4f, 0.15f, 0.8f, 0.25f, 1.2f, 0.35f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.5f, 2.5f}));
}

TEST(FuseMulBeforeFullyConnectedTest, MathVerification) {
  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithFullyConnected(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6), {0.05f, 0.4f, 0.15f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.5f, 2.5f}));
}

}  // namespace
}  // namespace ml_drift::ir
