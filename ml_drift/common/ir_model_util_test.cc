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

#include "ml_drift/common/ir_model_util.h"

#include <utility>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift::ir {
namespace {

GpuInfo GetTestGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.opencl_info.supports_fp16 = true;
  gpu_info.opencl_info.max_allocation_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.buffer_max_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.image2d_max_width = 16384;
  gpu_info.opencl_info.image2d_max_height = 16384;
  gpu_info.opencl_info.image_buffer_max_size = 65536;
  gpu_info.opencl_info.supported_images_2d.r_layout.insert(DataType::FLOAT32);
  gpu_info.opencl_info.supported_images_2d.rg_layout.insert(DataType::FLOAT32);
  gpu_info.opencl_info.supported_images_2d.rgb_layout.insert(DataType::FLOAT32);
  gpu_info.opencl_info.supported_images_2d.rgba_layout.insert(
      DataType::FLOAT32);
  gpu_info.opencl_info.supported_images_2d.r_layout.insert(DataType::FLOAT16);
  gpu_info.opencl_info.supported_images_2d.rg_layout.insert(DataType::FLOAT16);
  gpu_info.opencl_info.supported_images_2d.rgb_layout.insert(DataType::FLOAT16);
  gpu_info.opencl_info.supported_images_2d.rgba_layout.insert(
      DataType::FLOAT16);
  gpu_info.opencl_info.supports_images = true;
  gpu_info.vendor = GpuVendor::kQualcomm;
  gpu_info.adreno_info.generation = AdrenoInfo::Generation::kGen7;
  return gpu_info;
}

TEST(IrModelUtilTest, TryAddThenReduce) {
  IrModel ir_model;
  IrTensor* input1 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  IrTensor* input2 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.add_input(input1->id);
  ir_model.add_input(input2->id);

  IrOp* add_op = ir_model.add_op();
  add_op->name = ToString(OperationType::ADD);
  add_op->attr = ElementwiseAttributes{};
  ir_model.AddConsumer(input1->id, add_op->id);
  ir_model.AddConsumer(input2->id, add_op->id);

  IrTensor* add_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(add_out->id, add_op->id);

  IrOp* reduce_op = ir_model.add_op();
  reduce_op->name = ToString(OperationType::REDUCE_SUM);
  ReduceAttributes reduce_attr;
  reduce_attr.dims = {Axis::CHANNELS};
  reduce_op->attr = reduce_attr;
  ir_model.AddConsumer(add_out->id, reduce_op->id);

  IrTensor* reduce_out =
      ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 1));
  ir_model.SetProducer(reduce_out->id, reduce_op->id);
  ir_model.add_output(reduce_out->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::BUFFER;
  GpuInfo gpu_info = GetTestGpuInfo();

  GpuModel gpu_model;
  MLD_ASSERT_OK(IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
  // 1 fused node instead of 2.
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].inputs.size(), 2);
  EXPECT_THAT(gpu_model.nodes[0].name,
              ::testing::HasSubstr("reduce (2 input add)"));
}

TEST(IrModelUtilTest, TryMish) {
  IrModel ir_model;
  IrTensor* input = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.add_input(input->id);

  IrOp* exp_op = ir_model.add_op();
  exp_op->name = ToString(OperationType::EXP);
  exp_op->attr = ElementwiseAttributes{};
  ir_model.AddConsumer(input->id, exp_op->id);
  IrTensor* exp_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(exp_out->id, exp_op->id);

  IrOp* add_op = ir_model.add_op();
  add_op->name = ToString(OperationType::ADD);
  ElementwiseAttributes add_attr;
  add_attr.param = 1.0f;
  add_op->attr = add_attr;
  ir_model.AddConsumer(exp_out->id, add_op->id);
  IrTensor* add_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(add_out->id, add_op->id);

  IrOp* log_op = ir_model.add_op();
  log_op->name = ToString(OperationType::LOG);
  log_op->attr = ElementwiseAttributes{};
  ir_model.AddConsumer(add_out->id, log_op->id);
  IrTensor* log_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(log_out->id, log_op->id);

  IrOp* tanh_op = ir_model.add_op();
  tanh_op->name = ToString(OperationType::TANH);
  tanh_op->attr = ElementwiseAttributes{};
  ir_model.AddConsumer(log_out->id, tanh_op->id);
  IrTensor* tanh_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(tanh_out->id, tanh_op->id);

  IrOp* mul_op = ir_model.add_op();
  mul_op->name = ToString(OperationType::MUL);
  mul_op->attr = ElementwiseAttributes{};
  ir_model.AddConsumer(tanh_out->id, mul_op->id);
  ir_model.AddConsumer(input->id, mul_op->id);
  IrTensor* mul_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(mul_out->id, mul_op->id);

  ir_model.add_output(mul_out->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::BUFFER;
  GpuInfo gpu_info = GetTestGpuInfo();

  GpuModel gpu_model;
  MLD_ASSERT_OK(IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
  // 1 fused mish node instead of 5 individual elementwise ops.
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "mish");
}

TEST(IrModelUtilTest, TryConcatConv) {
  IrModel ir_model;
  IrTensor* input1 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 1));
  IrTensor* input2 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 1));
  IrTensor* input3 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 1));
  ir_model.add_input(input1->id);
  ir_model.add_input(input2->id);
  ir_model.add_input(input3->id);

  IrOp* concat_op = ir_model.add_op();
  concat_op->name = ToString(OperationType::CONCAT);
  ConcatAttributes concat_attr;
  concat_attr.axis = Axis::CHANNELS;
  concat_op->attr = concat_attr;
  ir_model.AddConsumer(input1->id, concat_op->id);
  ir_model.AddConsumer(input2->id, concat_op->id);
  ir_model.AddConsumer(input3->id, concat_op->id);
  IrTensor* concat_out =
      ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 3));
  ir_model.SetProducer(concat_out->id, concat_op->id);

  IrOp* conv_op = ir_model.add_op();
  conv_op->name = ToString(OperationType::CONVOLUTION_2D);
  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(1, 1);
  conv_attr.padding.appended = HW(1, 1);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(4, 3, 3, 3);
  // Add 16 elements for XNN_EXTRA_BYTES
  weights.data.resize(4 * 3 * 3 * 3 + 16, 1.0f);
  conv_attr.weights = std::move(weights);
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape = Linear(4);
  bias.data.resize(4, 0.0f);
  conv_attr.bias = std::move(bias);
  conv_op->attr = conv_attr;
  ir_model.AddConsumer(concat_out->id, conv_op->id);

  IrTensor* conv_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(conv_out->id, conv_op->id);
  ir_model.add_output(conv_out->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::TEXTURE_2D;
  // ConcatConv is a special kernel, needs to be enabled.
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  GpuInfo gpu_info = GetTestGpuInfo();
  // Ensure ThinLocalMemory doesn't intercept it to guarantee TryConcatConv
  // triggers.
  gpu_info.mali_info.generation = MaliInfo::Gen::kUnknown;

  GpuModel gpu_model;
  MLD_ASSERT_OK(IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
  // 1 fused node instead of 2.
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].inputs.size(), 3);
  EXPECT_THAT(gpu_model.nodes[0].name,
              ::testing::HasSubstr("convolution_2d (+ concat input)"));
}

TEST(IrModelUtilTest, TryResizeAddConvLocalMemoryFuser) {
  IrModel ir_model;
  IrTensor* input1 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  IrTensor* input2 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 4));
  ir_model.add_input(input1->id);
  ir_model.add_input(input2->id);

  IrOp* resize_op = ir_model.add_op();
  resize_op->name = ToString(OperationType::RESIZE);
  Resize2DAttributes resize_attr;
  resize_attr.new_shape = HW(4, 4);
  resize_attr.type = SamplingType::BILINEAR;
  resize_op->attr = resize_attr;
  ir_model.AddConsumer(input1->id, resize_op->id);
  IrTensor* resize_out =
      ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 4));
  ir_model.SetProducer(resize_out->id, resize_op->id);

  IrOp* add_op = ir_model.add_op();
  add_op->name = ToString(OperationType::ADD);
  add_op->attr = ElementwiseAttributes{};
  ir_model.AddConsumer(resize_out->id, add_op->id);
  ir_model.AddConsumer(input2->id, add_op->id);
  IrTensor* add_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 4));
  ir_model.SetProducer(add_out->id, add_op->id);

  IrOp* conv_op = ir_model.add_op();
  conv_op->name = ToString(OperationType::CONVOLUTION_2D);
  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(1, 1);
  conv_attr.padding.appended = HW(1, 1);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 3, 3, 4);
  // Add 16 elements for XNN_EXTRA_BYTES
  weights.data.resize(1 * 3 * 3 * 4 + 16, 1.0f);
  conv_attr.weights = std::move(weights);
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape = Linear(1);
  bias.data.resize(1, 0.0f);
  conv_attr.bias = std::move(bias);
  conv_op->attr = conv_attr;
  ir_model.AddConsumer(add_out->id, conv_op->id);

  IrTensor* conv_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 4, 4, 1));
  ir_model.SetProducer(conv_out->id, conv_op->id);
  ir_model.add_output(conv_out->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::TEXTURE_2D;
  // LocalMemory fusers require special kernel hints
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  GpuInfo gpu_info = GetTestGpuInfo();  // Faked Mali

  GpuModel gpu_model;
  MLD_ASSERT_OK(IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
  // 1 fused node instead of 3.
  EXPECT_EQ(gpu_model.nodes.size(), 1);
}

TEST(IrModelUtilTest, HandlesTombstonedOpsAndTensors) {
  IrModel ir_model;
  IrTensor* input1 = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.add_input(input1->id);

  // Intermediate identity op to be tombstoned
  IrOp* relu_op = ir_model.add_op();
  relu_op->name = ToString(OperationType::RELU);
  relu_op->attr = ReLUAttributes{};
  ir_model.AddConsumer(input1->id, relu_op->id);

  IrTensor* relu_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(relu_out->id, relu_op->id);

  IrOp* add_op = ir_model.add_op();
  add_op->name = ToString(OperationType::ADD);
  add_op->attr = ElementwiseAttributes{};
  ir_model.AddConsumer(relu_out->id, add_op->id);
  ir_model.AddConsumer(input1->id, add_op->id);

  IrTensor* add_out = ir_model.add_tensor(DataType::FLOAT32, BHWC(1, 2, 2, 4));
  ir_model.SetProducer(add_out->id, add_op->id);
  ir_model.add_output(add_out->id);

  // RemoveSimpleOp tombstones relu_op (null op entry in model_ops).
  MLD_ASSERT_OK(ir_model.RemoveSimpleOp(relu_op->id));

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::BUFFER;
  GpuInfo gpu_info = GetTestGpuInfo();

  GpuModel gpu_model;
  MLD_ASSERT_OK(IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
}

}  // namespace
}  // namespace ml_drift::ir
