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

#include "ml_drift/common/gpu_model_linking_test_util.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/kernels/cast.h"
#include "ml_drift/common/kernels/concat_z.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/padding.h"
#include "ml_drift/common/kernels/prelu.h"
#include "ml_drift/common/kernels/reshape.h"
#include "ml_drift/common/kernels/reshapex4.h"
#include "ml_drift/common/kernels/strided_slice.h"
#include "ml_drift/common/kernels/transpose.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/transformations/model_transformations.h"

namespace ml_drift {

absl::Status TestLinkingConvolutionAndCosOp(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 128);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input->id);

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, cos_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  Value* cos_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, cos_node, &cos_output));
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 16);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolution2InputMul2InputMul(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 16);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 32, 32, 16);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input0->id);

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  graph.AddConsumer(mul0_node->id, input1->id);
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  graph.AddConsumer(mul1_node->id, input2->id);
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  Value* mul1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, mul1_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src0_tensor, src1_tensor, src2_tensor},
          std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src1_tensor.shape,
          mul0_output->tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src2_tensor.shape,
          mul1_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolution2InputBroadcastMul2InputMul(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 1);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 32, 32, 16);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input0->id);

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  graph.AddConsumer(mul0_node->id, input1->id);
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  graph.AddConsumer(mul1_node->id, input2->id);
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  Value* mul1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, mul1_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src0_tensor, src1_tensor, src2_tensor},
          std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src1_tensor.shape,
          mul0_output->tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src2_tensor.shape,
          mul1_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolution2InputMul2InputBroadcastMul(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 16);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 1, 1, 16);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input0->id);

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  graph.AddConsumer(mul0_node->id, input1->id);
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  graph.AddConsumer(mul1_node->id, input2->id);
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  Value* mul1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, mul1_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src0_tensor, src1_tensor, src2_tensor},
          std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src1_tensor.shape,
          mul0_output->tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src2_tensor.shape,
          mul1_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolution2InputMul2InputMulCos(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  auto input2 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 128);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 16);
  input2->tensor.type = DataType::FLOAT32;
  input2->tensor.shape = BHWC(1, 32, 32, 16);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input0->id);

  auto mul0_node = graph.NewNode();
  mul0_node->operation.type = ToString(OperationType::MUL);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, mul0_node, &conv_output));
  graph.AddConsumer(mul0_node->id, input1->id);
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto mul1_node = graph.NewNode();
  mul1_node->operation.type = ToString(OperationType::MUL);
  Value* mul0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul0_node, mul1_node, &mul0_output));
  graph.AddConsumer(mul1_node->id, input2->id);
  mul0_output->tensor.type = DataType::FLOAT32;
  mul0_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  Value* mul1_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, mul1_node, cos_node, &mul1_output));
  mul1_output->tensor.type = DataType::FLOAT32;
  mul1_output->tensor.shape = BHWC(1, 32, 32, 16);

  Value* cos_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, cos_node, &cos_output));
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 16);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src0_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src1_tensor;
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src2_tensor;
      src2_tensor.shape = input2->tensor.shape;
      src2_tensor.data.resize(src2_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src2_tensor.data.size(); ++i) {
        src2_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src0_tensor, src1_tensor, src2_tensor},
          std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src0_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      OperationDef op_def_mul;
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_mul.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation mul0_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src1_tensor.shape,
          mul0_output->tensor.shape);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, src1_tensor},
          std::make_unique<GPUOperation>(std::move(mul0_operation)),
          mul0_output->tensor.shape, &intermediate1));

      GPUOperation mul1_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_mul, OperationType::MUL, src2_tensor.shape,
          mul1_output->tensor.shape);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, src2_tensor},
          std::make_unique<GPUOperation>(std::move(mul1_operation)),
          mul1_output->tensor.shape, &intermediate2));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate2,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolutionFirstTanh2InputDiff(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 128);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input->id);

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, tanh_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.SetProducer(tanh_node->id, tanh_output->id);
  graph.AddConsumer(sub_node->id, tanh_output->id);
  graph.AddConsumer(sub_node->id, conv_output->id);
  graph.SetProducer(sub_node->id, sub_output->id);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      OperationDef op_def_sub;
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.dst_tensors.push_back({data_type, storage, Layout::HWC});
      GPUOperation sub_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_sub, OperationType::SUB,
          conv_output->tensor.shape, sub_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, intermediate0},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConvolutionSecondTanh2InputDiff(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 128);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input->id);

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, tanh_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.SetProducer(tanh_node->id, tanh_output->id);
  graph.AddConsumer(sub_node->id, conv_output->id);
  graph.AddConsumer(sub_node->id, tanh_output->id);
  graph.SetProducer(sub_node->id, sub_output->id);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      OperationDef op_def_sub;
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.dst_tensors.push_back({data_type, storage, Layout::HWC});
      GPUOperation sub_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_sub, OperationType::SUB,
          conv_output->tensor.shape, sub_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate0, intermediate1},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

//      input
//        |
//   convolution
//     /     \
//   tanh    cos
//     \     /
//  subtraction
//        |
//     output
absl::Status TestLinkingConvolutionFirstTanhSecondCos2InputDiff(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 128);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input->id);

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, conv_node, tanh_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  auto cos_output = graph.NewValue();
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(cos_node->id, conv_output->id);
  graph.SetProducer(cos_node->id, cos_output->id);

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.SetProducer(tanh_node->id, tanh_output->id);
  graph.AddConsumer(sub_node->id, tanh_output->id);
  graph.AddConsumer(sub_node->id, cos_output->id);
  graph.SetProducer(sub_node->id, sub_output->id);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &intermediate2));

      OperationDef op_def_sub;
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_sub.dst_tensors.push_back({data_type, storage, Layout::HWC});
      GPUOperation sub_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_sub, OperationType::SUB,
          conv_output->tensor.shape, sub_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, intermediate2},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

//      input
//        |
//   convolution
//      /    \
//   tanh    cos
//    /     /   \
//   |    prelu  sin
//   |      |     |
//   |     abs   /
//   |      \   /
//   |       pow
//   |        |
//   |       exp
//    \       |
//   subtraction
//        |
//     output
absl::Status TestLinkingComplex0(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 128);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  auto conv_output = graph.NewValue();
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(conv_node->id, input->id);
  graph.SetProducer(conv_node->id, conv_output->id);

  auto tanh_node = graph.NewNode();
  tanh_node->operation.type = ToString(OperationType::TANH);
  auto tanh_output = graph.NewValue();
  tanh_output->tensor.type = DataType::FLOAT32;
  tanh_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(tanh_node->id, conv_output->id);
  graph.SetProducer(tanh_node->id, tanh_output->id);

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  auto cos_output = graph.NewValue();
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(cos_node->id, conv_output->id);
  graph.SetProducer(cos_node->id, cos_output->id);

  auto prelu_node = graph.NewNode();
  prelu_node->operation.type = ToString(OperationType::PRELU);
  PReLUAttributes prelu_attr;
  Tensor<Linear, DataType::FLOAT32> parameters;
  parameters.shape = Linear(16);
  parameters.data.resize(parameters.shape.DimensionsProduct());
  for (int i = 0; i < parameters.data.size(); ++i) {
    parameters.data[i] = std::sin(i * 0.5f);
  }
  prelu_attr.alpha = parameters;
  prelu_node->operation.attributes = prelu_attr;
  auto prelu_output = graph.NewValue();
  prelu_output->tensor.type = DataType::FLOAT32;
  prelu_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(prelu_node->id, cos_output->id);
  graph.SetProducer(prelu_node->id, prelu_output->id);

  auto abs_node = graph.NewNode();
  abs_node->operation.type = ToString(OperationType::ABS);
  auto abs_output = graph.NewValue();
  abs_output->tensor.type = DataType::FLOAT32;
  abs_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(abs_node->id, prelu_output->id);
  graph.SetProducer(abs_node->id, abs_output->id);

  auto sin_node = graph.NewNode();
  sin_node->operation.type = ToString(OperationType::SIN);
  auto sin_output = graph.NewValue();
  sin_output->tensor.type = DataType::FLOAT32;
  sin_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(sin_node->id, cos_output->id);
  graph.SetProducer(sin_node->id, sin_output->id);

  auto pow_node = graph.NewNode();
  pow_node->operation.type = ToString(OperationType::POW);
  auto pow_output = graph.NewValue();
  pow_output->tensor.type = DataType::FLOAT32;
  pow_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(pow_node->id, abs_output->id);
  graph.AddConsumer(pow_node->id, sin_output->id);
  graph.SetProducer(pow_node->id, pow_output->id);

  auto exp_node = graph.NewNode();
  exp_node->operation.type = ToString(OperationType::EXP);
  auto exp_output = graph.NewValue();
  exp_output->tensor.type = DataType::FLOAT32;
  exp_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(exp_node->id, pow_output->id);
  graph.SetProducer(exp_node->id, exp_output->id);

  auto sub_node = graph.NewNode();
  sub_node->operation.type = ToString(OperationType::SUB);
  auto sub_output = graph.NewValue();
  sub_output->tensor.type = DataType::FLOAT32;
  sub_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(sub_node->id, tanh_output->id);
  graph.AddConsumer(sub_node->id, exp_output->id);
  graph.SetProducer(sub_node->id, sub_output->id);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      OperationDef op_def_two_input;
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate0));

      GPUOperation tanh_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(tanh_operation)),
          tanh_output->tensor.shape, &intermediate1));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &intermediate2));

      GPUOperation prelu_operation =
          CreatePReLU(env->GetGpuInfo(), op_def, prelu_attr);
      TensorFloat32 intermediate3;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate2,
          std::make_unique<GPUOperation>(std::move(prelu_operation)),
          prelu_output->tensor.shape, &intermediate3));

      GPUOperation abs_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::ABS);
      TensorFloat32 intermediate4;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate3,
          std::make_unique<GPUOperation>(std::move(abs_operation)),
          prelu_output->tensor.shape, &intermediate4));

      GPUOperation sin_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIN);
      TensorFloat32 intermediate5;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate2,
          std::make_unique<GPUOperation>(std::move(sin_operation)),
          sin_output->tensor.shape, &intermediate5));

      GPUOperation pow_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_two_input, OperationType::POW,
          sin_output->tensor.shape, pow_output->tensor.shape);
      TensorFloat32 intermediate6;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate4, intermediate5},
          std::make_unique<GPUOperation>(std::move(pow_operation)),
          pow_output->tensor.shape, &intermediate6));

      GPUOperation exp_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::EXP);
      TensorFloat32 intermediate7;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate6,
          std::make_unique<GPUOperation>(std::move(exp_operation)),
          exp_output->tensor.shape, &intermediate7));

      GPUOperation sub_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_two_input, OperationType::SUB,
          conv_output->tensor.shape, sub_output->tensor.shape);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate1, intermediate7},
          std::make_unique<GPUOperation>(std::move(sub_operation)),
          sub_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

//                input1
//                  |
//              convolution
//                  |
//         input0  cos0
//             \   /
//              add
//               |
//              cos1
//               |
//              sin
//               |
//              abs
//               |
//             output
absl::Status TestLinkingConvElem2InputAddElemsOp(
    TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  auto input1 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 16);
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 8);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 8);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  conv_node->operation.attributes = conv_attr;
  auto conv_output = graph.NewValue();
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(conv_node->id, input1->id);
  graph.SetProducer(conv_node->id, conv_output->id);

  auto cos0_node = graph.NewNode();
  cos0_node->operation.type = ToString(OperationType::COS);
  auto cos0_output = graph.NewValue();
  cos0_output->tensor.type = DataType::FLOAT32;
  cos0_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(cos0_node->id, conv_output->id);
  graph.SetProducer(cos0_node->id, cos0_output->id);

  auto add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::ADD);
  auto add_output = graph.NewValue();
  add_output->tensor.type = DataType::FLOAT32;
  add_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(add_node->id, input0->id);
  graph.AddConsumer(add_node->id, cos0_output->id);
  graph.SetProducer(add_node->id, add_output->id);

  auto cos1_node = graph.NewNode();
  cos1_node->operation.type = ToString(OperationType::COS);
  auto cos1_output = graph.NewValue();
  cos1_output->tensor.type = DataType::FLOAT32;
  cos1_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(cos1_node->id, add_output->id);
  graph.SetProducer(cos1_node->id, cos1_output->id);

  auto sin_node = graph.NewNode();
  sin_node->operation.type = ToString(OperationType::SIN);
  auto sin_output = graph.NewValue();
  sin_output->tensor.type = DataType::FLOAT32;
  sin_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(sin_node->id, cos1_output->id);
  graph.SetProducer(sin_node->id, sin_output->id);

  auto abs_node = graph.NewNode();
  abs_node->operation.type = ToString(OperationType::ABS);
  auto abs_output = graph.NewValue();
  abs_output->tensor.type = DataType::FLOAT32;
  abs_output->tensor.shape = BHWC(1, 32, 32, 16);
  graph.AddConsumer(abs_node->id, sin_output->id);
  graph.SetProducer(abs_node->id, abs_output->id);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    if (precision != CalculationsPrecision::F16) {
      continue;
    }
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src0_tensor, src1_tensor;
      src0_tensor.shape = input0->tensor.shape;
      src0_tensor.data.resize(src0_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src0_tensor.data.size(); ++i) {
        src0_tensor.data[i] = std::sin(i * 0.12345f);
      }
      src1_tensor.shape = input1->tensor.shape;
      src1_tensor.data.resize(src1_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src1_tensor.data.size(); ++i) {
        src1_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src0_tensor, src1_tensor},
          std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      OperationDef op_def_two_input;
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.dst_tensors.push_back({data_type, storage, Layout::HWC});

      ConvGeneric conv_operation =
          CreateConvGeneric(env->GetGpuInfo(), op_def, precision, conv_attr);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src1_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
          conv_output->tensor.shape, &intermediate1));

      GPUOperation cos0_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate2;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate1,
          std::make_unique<GPUOperation>(std::move(cos0_operation)),
          cos0_output->tensor.shape, &intermediate2));

      GPUOperation add_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def_two_input, OperationType::ADD,
          add_output->tensor.shape, add_output->tensor.shape);
      TensorFloat32 intermediate3;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src0_tensor, intermediate2},
          std::make_unique<GPUOperation>(std::move(add_operation)),
          add_output->tensor.shape, &intermediate3));

      GPUOperation cos1_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate4;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate3,
          std::make_unique<GPUOperation>(std::move(cos1_operation)),
          cos1_output->tensor.shape, &intermediate4));

      GPUOperation sin_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIN);
      TensorFloat32 intermediate5;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate4,
          std::make_unique<GPUOperation>(std::move(sin_operation)),
          sin_output->tensor.shape, &intermediate5));

      GPUOperation abs_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::ABS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate5,
          std::make_unique<GPUOperation>(std::move(abs_operation)),
          abs_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

//     input1
//       |
//     slice
//       |
//      cast
//       |
//     output
absl::Status TestLinkingSliceCastOp(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 1, 1, 4);

  auto slice_node = graph.NewNode();
  slice_node->operation.type = ToString(OperationType::SLICE);

  SliceAttributes slice_attr;
  slice_attr.strides = BHWC(1, 1, 1, 1);
  slice_attr.starts = BHWC(0, 0, 0, 0);
  slice_attr.ends = BHWC(1, 1, 1, 1);

  slice_node->operation.attributes = slice_attr;
  graph.AddConsumer(slice_node->id, input->id);

  auto cast_int_node = graph.NewNode();
  cast_int_node->operation.type = ToString(OperationType::CAST);
  Value* slice_output = nullptr;
  RETURN_IF_ERROR(
      ConnectTwoNodes(&graph, slice_node, cast_int_node, &slice_output));
  slice_output->tensor.type = DataType::FLOAT32;
  slice_output->tensor.shape = BHWC(1, 1, 1, 1);

  Value* cast_int_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, cast_int_node, &cast_int_output));
  cast_int_output->tensor.type = DataType::INT32;
  cast_int_output->tensor.shape = BHWC(1, 1, 1, 1);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));
      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node");
      }
      TensorInt32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorInt32*>{&dst_tensor_v1}, &gpu_model));

      TensorFloat32 intermediate;
      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      StridedSlice slice_operation = CreateStridedSlice(op_def, slice_attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor,
          std::make_unique<StridedSlice>(std::move(slice_operation)),
          slice_output->tensor.shape, &intermediate));

      OperationDef cast_int_op_def;
      DataType int_data_type = DataType::INT32;
      cast_int_op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      cast_int_op_def.dst_tensors.push_back(
          {int_data_type, storage, Layout::HWC});

      GPUOperation cast_int_operation =
          CreateCast(cast_int_op_def, env->GetGpuInfo());
      using TensorInt32 = Tensor<BHWC, DataType::INT32>;
      TensorInt32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          std::vector<TensorFloat32>{intermediate},
          std::make_unique<GPUOperation>(std::move(cast_int_operation)),
          std::vector<BHWC>{slice_output->tensor.shape},
          std::vector<TensorInt32*>{&dst_tensor_v0}));

      if (dst_tensor_v0.data != dst_tensor_v1.data) {
        return absl::InternalError("Expected output to be equal");
      }
    }
  }
  return absl::OkStatus();
}

//       input
//         |
//      Reshape
//       /   \
//     Add   Add
//       \   /
//        Mul
//         |
//       output
absl::Status TestLinkingAddAddMulOp(TestExecutionEnvironment* env,
                                    bool use_second_input_add) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 1, 5, 4);
  auto reshape_node = graph.NewNode();
  reshape_node->operation.type = ToString(OperationType::RESHAPE);
  ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = BHWC(1, 1, 1, 20);
  reshape_node->operation.attributes = reshape_attr;
  graph.AddConsumer(reshape_node->id, input->id);

  ElementwiseAttributes add_attr;
  add_attr.param = 1.0f;
  auto add_left_node = graph.NewNode();
  add_left_node->operation.type = ToString(OperationType::ADD);
  add_left_node->operation.attributes = add_attr;
  Value* reshape_output = nullptr;
  RETURN_IF_ERROR(
      ConnectTwoNodes(&graph, reshape_node, add_left_node, &reshape_output));
  reshape_output->tensor.type = DataType::FLOAT32;
  reshape_output->tensor.shape = BHWC(1, 1, 1, 20);

  Node* second_input_node = reshape_node;
  if (use_second_input_add) {
    auto add_right_node = graph.NewNode();
    add_right_node->operation.type = ToString(OperationType::ADD);
    add_right_node->operation.attributes = add_attr;
    RETURN_IF_ERROR(
        ConnectTwoNodes(&graph, reshape_node, add_right_node, &reshape_output));
    second_input_node = add_right_node;
  }
  auto mul_node = graph.NewNode();
  mul_node->operation.type = ToString(OperationType::MUL);
  Value* add_left_output = nullptr;
  RETURN_IF_ERROR(
      ConnectTwoNodes(&graph, add_left_node, mul_node, &add_left_output));
  add_left_output->tensor.type = DataType::FLOAT32;
  add_left_output->tensor.shape = BHWC(1, 1, 1, 20);
  if (use_second_input_add) {
    Value* add_right_output = nullptr;
    RETURN_IF_ERROR(ConnectTwoNodes(&graph, second_input_node, mul_node,
                                    &add_right_output));
    add_right_output->tensor.type = DataType::FLOAT32;
    add_right_output->tensor.shape = BHWC(1, 1, 1, 20);
  } else {
    RETURN_IF_ERROR(
        ConnectTwoNodes(&graph, second_input_node, mul_node, &reshape_output));
  }
  Value* mul_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, mul_node, &mul_output));
  mul_output->tensor.type = DataType::FLOAT32;
  mul_output->tensor.shape = BHWC(1, 1, 1, 20);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }
      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));
      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));
      TensorFloat32 intermediate;
      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation reshape_operation = CreateReshape(op_def);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor,
          std::make_unique<GPUOperation>(std::move(reshape_operation)),
          reshape_output->tensor.shape, &intermediate));

      TensorFloat32 add;
      TensorFloat32 ones;
      ones.shape = intermediate.shape;
      ones.data = std::vector(ones.shape.DimensionsProduct(), 1.0f);
      GPUOperation add_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def, OperationType::ADD,
          reshape_output->tensor.shape, add_left_output->tensor.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {intermediate, ones},
          std::make_unique<GPUOperation>(std::move(add_operation)),
          add_left_output->tensor.shape, &add));
      TensorFloat32 second_input = intermediate;
      if (use_second_input_add) {
        second_input = add;
      }
      TensorFloat32 dst_tensor_v0;
      GPUOperation mul_operation = CreateElementwiseTwoInput(
          env->GetGpuInfo(), op_def, OperationType::MUL,
          add_left_output->tensor.shape, add_left_output->tensor.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {add, second_input},
          std::make_unique<GPUOperation>(std::move(mul_operation)),
          add_left_output->tensor.shape, &dst_tensor_v0));
      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingConcatAndCosOp(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input0 = graph.NewValue();
  input0->tensor.type = DataType::FLOAT32;
  input0->tensor.shape = BHWC(1, 32, 32, 21);
  auto input1 = graph.NewValue();
  input1->tensor.type = DataType::FLOAT32;
  input1->tensor.shape = BHWC(1, 32, 32, 7);

  auto concat_node = graph.NewNode();
  concat_node->operation.type = ToString(OperationType::CONCAT);

  ConcatAttributes concat_attr;
  concat_attr.axis = Axis::CHANNELS;

  concat_node->operation.attributes = concat_attr;
  graph.AddConsumer(concat_node->id, input0->id);
  graph.AddConsumer(concat_node->id, input1->id);

  auto cos_node = graph.NewNode();
  cos_node->operation.type = ToString(OperationType::COS);
  Value* concat_output = nullptr;
  RETURN_IF_ERROR(
      ConnectTwoNodes(&graph, concat_node, cos_node, &concat_output));
  concat_output->tensor.type = DataType::FLOAT32;
  concat_output->tensor.shape = BHWC(1, 32, 32, 28);

  Value* cos_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, cos_node, &cos_output));
  cos_output->tensor.type = DataType::FLOAT32;
  cos_output->tensor.shape = BHWC(1, 32, 32, 28);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor0;
      src_tensor0.shape = input0->tensor.shape;
      src_tensor0.data.resize(src_tensor0.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor0.data.size(); ++i) {
        src_tensor0.data[i] = std::sin(i * 0.12345f);
      }
      TensorFloat32 src_tensor1;
      src_tensor1.shape = input1->tensor.shape;
      src_tensor1.data.resize(src_tensor1.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor1.data.size(); ++i) {
        src_tensor1.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor0, src_tensor1},
          std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      OperationDef op_def_two_input;
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def_two_input.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation concat_operation =
          CreateConcatZ(op_def_two_input, {21, 7}, env->GetGpuInfo());
      TensorFloat32 intermediate;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor0, src_tensor1},
          std::make_unique<GPUOperation>(std::move(concat_operation)),
          concat_output->tensor.shape, &intermediate));

      GPUOperation cos_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate,
          std::make_unique<GPUOperation>(std::move(cos_operation)),
          cos_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestLinkingCosAndCosOp(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 16);

  auto cos0_node = graph.NewNode();
  cos0_node->operation.type = ToString(OperationType::COS);

  graph.AddConsumer(cos0_node->id, input->id);

  auto cos1_node = graph.NewNode();
  cos1_node->operation.type = ToString(OperationType::COS);
  Value* cos0_output = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, cos0_node, cos1_node, &cos0_output));
  cos0_output->tensor.type = DataType::FLOAT32;
  cos0_output->tensor.shape = BHWC(1, 32, 32, 16);

  Value* cos1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, cos1_node, &cos1_output));
  cos1_output->tensor.type = DataType::FLOAT32;
  cos1_output->tensor.shape = BHWC(1, 32, 32, 16);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      GPUOperation cos0_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 intermediate;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(cos0_operation)),
          cos0_output->tensor.shape, &intermediate));

      GPUOperation cos1_operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate,
          std::make_unique<GPUOperation>(std::move(cos1_operation)),
          cos1_output->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

absl::Status TestFloatCastToBoolCastToFloat(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 16);

  auto cast0_node = graph.NewNode();
  cast0_node->operation.type = ToString(OperationType::CAST);

  graph.AddConsumer(cast0_node->id, input->id);

  auto cast1_node = graph.NewNode();
  cast1_node->operation.type = ToString(OperationType::CAST);
  Value* cast0_output = nullptr;
  RETURN_IF_ERROR(
      ConnectTwoNodes(&graph, cast0_node, cast1_node, &cast0_output));
  cast0_output->tensor.type = DataType::BOOL;
  cast0_output->tensor.shape = BHWC(1, 32, 32, 16);

  Value* cast1_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, cast1_node, &cast1_output));
  cast1_output->tensor.type = DataType::FLOAT32;
  cast1_output->tensor.shape = BHWC(1, 32, 32, 16);

  for (auto float_storage : env->GetSupportedStorages(DataType::FLOAT32)) {
    for (auto bool_storage : env->GetSupportedStorages(DataType::BOOL)) {
      CreateGpuModelInfo create_info;
      create_info.precision = CalculationsPrecision::F32;
      create_info.storage_type = float_storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f);
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op0_def;
      op0_def.src_tensors.push_back(
          {DataType::FLOAT32, float_storage, Layout::HWC});
      op0_def.dst_tensors.push_back(
          {DataType::BOOL, bool_storage, Layout::HWC});

      OperationDef op1_def;
      op1_def.src_tensors.push_back(
          {DataType::BOOL, bool_storage, Layout::HWC});
      op1_def.dst_tensors.push_back(
          {DataType::FLOAT32, float_storage, Layout::HWC});

      GPUOperation cast0_operation = CreateCast(op0_def, env->GetGpuInfo());
      Tensor<BHWC, DataType::BOOL> intermediate;
      {
        std::vector<TensorDescriptor> src_cpu_descs(1);
        std::vector<TensorDescriptor*> src_cpu_desc_ptrs(1);
        std::vector<TensorDescriptor> dst_cpu_descs(1);
        std::vector<TensorDescriptor*> dst_cpu_desc_ptrs(1);
        src_cpu_descs[0] = op0_def.src_tensors[0];
        src_cpu_descs[0].UploadData(src_tensor);
        src_cpu_desc_ptrs[0] = &src_cpu_descs[0];
        dst_cpu_descs[0] = op0_def.dst_tensors[0];
        dst_cpu_descs[0].SetBHWCShape(cast0_output->tensor.shape);
        dst_cpu_desc_ptrs[0] = &dst_cpu_descs[0];
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
            std::make_unique<GPUOperation>(std::move(cast0_operation))));
        dst_cpu_descs[0].DownloadData(&intermediate);
      }

      GPUOperation cast1_operation = CreateCast(op1_def, env->GetGpuInfo());
      TensorFloat32 dst_tensor_v0;
      {
        std::vector<TensorDescriptor> src_cpu_descs(1);
        std::vector<TensorDescriptor*> src_cpu_desc_ptrs(1);
        std::vector<TensorDescriptor> dst_cpu_descs(1);
        std::vector<TensorDescriptor*> dst_cpu_desc_ptrs(1);
        src_cpu_descs[0] = op1_def.src_tensors[0];
        src_cpu_descs[0].UploadData(intermediate);
        src_cpu_desc_ptrs[0] = &src_cpu_descs[0];
        dst_cpu_descs[0] = op1_def.dst_tensors[0];
        dst_cpu_descs[0].SetBHWCShape(cast1_output->tensor.shape);
        dst_cpu_desc_ptrs[0] = &dst_cpu_descs[0];
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
            std::make_unique<GPUOperation>(std::move(cast1_operation))));
        dst_cpu_descs[0].DownloadData(&dst_tensor_v0);
      }

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

//   input_tensor
//        |
//     reshape
//        |
//  interm_tensor
//        |
//    transpose
//        |
//  output_tensor
absl::Status TestReshapeTranspose(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 2, 4, 8);

  auto reshape_node = graph.NewNode();
  reshape_node->operation.type = ToString(OperationType::RESHAPE);
  ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = BHWC(1, 2, 8, 4);
  reshape_node->operation.attributes = reshape_attr;

  graph.AddConsumer(reshape_node->id, input->id);

  auto transpose_node = graph.NewNode();
  transpose_node->operation.type = ToString(OperationType::TRANSPOSE);
  TransposeAttributes transpose_attr;
  transpose_attr.perm = BHWC(0, 2, 1, 3);
  transpose_node->operation.attributes = transpose_attr;
  Value* interm_tensor_ptr = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, reshape_node, transpose_node,
                                  &interm_tensor_ptr));
  interm_tensor_ptr->tensor.type = DataType::FLOAT32;
  interm_tensor_ptr->tensor.shape = BHWC(1, 2, 8, 4);

  Value* output_tensor_ptr = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, transpose_node, &output_tensor_ptr));
  output_tensor_ptr->tensor.type = DataType::FLOAT32;
  output_tensor_ptr->tensor.shape = BHWC(1, 8, 2, 4);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f) + 1.0f;
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      Reshapex4 reshape_operation = CreateReshapex4(op_def);
      TensorFloat32 intermediate;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<Reshapex4>(std::move(reshape_operation)),
          interm_tensor_ptr->tensor.shape, &intermediate));

      GPUOperation transpose_operation =
          CreateTranspose(op_def, transpose_attr);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate,
          std::make_unique<GPUOperation>(std::move(transpose_operation)),
          output_tensor_ptr->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

//   input_tensor
//        |
//     reshape0
//        |
//  interm_tensor0
//        |
//    transpose
//        |
//  interm_tensor1
//        |
//    reshape1
//        |
//  output_tensor
absl::Status TestReshapeTransposeReshape(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 2, 4, 8);

  auto reshape0_node = graph.NewNode();
  reshape0_node->operation.type = ToString(OperationType::RESHAPE);
  ReshapeAttributes reshape0_attr;
  reshape0_attr.new_shape = BHWC(1, 2, 8, 4);
  reshape0_node->operation.attributes = reshape0_attr;

  graph.AddConsumer(reshape0_node->id, input->id);

  auto transpose_node = graph.NewNode();
  transpose_node->operation.type = ToString(OperationType::TRANSPOSE);
  TransposeAttributes transpose_attr;
  transpose_attr.perm = BHWC(0, 2, 1, 3);
  transpose_node->operation.attributes = transpose_attr;
  Value* interm0_tensor_ptr = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, reshape0_node, transpose_node,
                                  &interm0_tensor_ptr));
  interm0_tensor_ptr->tensor.type = DataType::FLOAT32;
  interm0_tensor_ptr->tensor.shape = BHWC(1, 2, 8, 4);

  auto reshape1_node = graph.NewNode();
  reshape1_node->operation.type = ToString(OperationType::RESHAPE);
  ReshapeAttributes reshape1_attr;
  reshape1_attr.new_shape = BHWC(1, 2, 8, 4);
  reshape1_node->operation.attributes = reshape1_attr;
  Value* interm1_tensor_ptr = nullptr;
  RETURN_IF_ERROR(ConnectTwoNodes(&graph, transpose_node, reshape1_node,
                                  &interm1_tensor_ptr));
  interm1_tensor_ptr->tensor.type = DataType::FLOAT32;
  interm1_tensor_ptr->tensor.shape = BHWC(1, 8, 2, 4);

  Value* output_tensor_ptr = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, reshape1_node, &output_tensor_ptr));
  output_tensor_ptr->tensor.type = DataType::FLOAT32;
  output_tensor_ptr->tensor.shape = BHWC(1, 1, 1, 64);

  for (auto data_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    CalculationsPrecision precision = data_type == DataType::FLOAT32
                                          ? CalculationsPrecision::F32
                                          : CalculationsPrecision::F16;
    for (auto storage : env->GetSupportedStorages(data_type)) {
      CreateGpuModelInfo create_info;
      create_info.precision = precision;
      create_info.storage_type = storage;
      create_info.hints.Add(ModelHints::kAllowSpecialKernels);

      GpuModel gpu_model;
      RETURN_IF_ERROR(
          GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

      if (gpu_model.nodes.size() != 1) {
        return absl::InternalError("Expected model with one node.");
      }

      TensorFloat32 src_tensor;
      src_tensor.shape = input->tensor.shape;
      src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
      for (int i = 0; i < src_tensor.data.size(); ++i) {
        src_tensor.data[i] = std::sin(i * 0.12345f) + 1.0f;
      }

      TensorFloat32 dst_tensor_v1;
      RETURN_IF_ERROR(env->ExecuteGpuModel(
          {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
          &gpu_model));

      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      Reshapex4 reshape0_operation = CreateReshapex4(op_def);
      TensorFloat32 intermediate0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor,
          std::make_unique<Reshapex4>(std::move(reshape0_operation)),
          interm0_tensor_ptr->tensor.shape, &intermediate0));

      GPUOperation transpose_operation =
          CreateTranspose(op_def, transpose_attr);
      TensorFloat32 intermediate1;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate0,
          std::make_unique<GPUOperation>(std::move(transpose_operation)),
          interm1_tensor_ptr->tensor.shape, &intermediate1));

      Reshapex4 reshape1_operation = CreateReshapex4(op_def);
      TensorFloat32 dst_tensor_v0;
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          intermediate1,
          std::make_unique<Reshapex4>(std::move(reshape1_operation)),
          output_tensor_ptr->tensor.shape, &dst_tensor_v0));

      EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
    }
  }
  return absl::OkStatus();
}

//   input_tensor
//        |
//     reshape
//     /     \
//   sin0    cos0
//     \     /
//   subtraction0
//     /     \
//   sin1    cos1
//     \     /
//   subtraction1
//        |
//  output_tensor
absl::Status TestTwoInputTwise(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 64, 64, 32);

  auto reshape_node = graph.NewNode();
  reshape_node->operation.type = ToString(OperationType::RESHAPE);
  ReshapeAttributes reshape_attr;
  reshape_attr.new_shape = BHWC(1, 32, 32, 128);
  reshape_node->operation.attributes = reshape_attr;
  auto reshape_output = graph.NewValue();
  reshape_output->tensor.type = DataType::FLOAT32;
  reshape_output->tensor.shape = BHWC(1, 32, 32, 128);
  graph.AddConsumer(reshape_node->id, input->id);
  graph.SetProducer(reshape_node->id, reshape_output->id);

  auto sin0_node = graph.NewNode();
  sin0_node->operation.type = ToString(OperationType::SIN);
  auto sin0_output = graph.NewValue();
  sin0_output->tensor.type = DataType::FLOAT32;
  sin0_output->tensor.shape = BHWC(1, 32, 32, 128);
  graph.AddConsumer(sin0_node->id, reshape_output->id);
  graph.SetProducer(sin0_node->id, sin0_output->id);

  auto cos0_node = graph.NewNode();
  cos0_node->operation.type = ToString(OperationType::COS);
  auto cos0_output = graph.NewValue();
  cos0_output->tensor.type = DataType::FLOAT32;
  cos0_output->tensor.shape = BHWC(1, 32, 32, 128);
  graph.AddConsumer(cos0_node->id, reshape_output->id);
  graph.SetProducer(cos0_node->id, cos0_output->id);

  auto sub0_node = graph.NewNode();
  sub0_node->operation.type = ToString(OperationType::SUB);
  auto sub0_output = graph.NewValue();
  sub0_output->tensor.type = DataType::FLOAT32;
  sub0_output->tensor.shape = BHWC(1, 32, 32, 128);
  graph.AddConsumer(sub0_node->id, sin0_output->id);
  graph.AddConsumer(sub0_node->id, cos0_output->id);
  graph.SetProducer(sub0_node->id, sub0_output->id);

  auto sin1_node = graph.NewNode();
  sin1_node->operation.type = ToString(OperationType::SIN);
  auto sin1_output = graph.NewValue();
  sin1_output->tensor.type = DataType::FLOAT32;
  sin1_output->tensor.shape = BHWC(1, 32, 32, 128);
  graph.AddConsumer(sin1_node->id, sub0_output->id);
  graph.SetProducer(sin1_node->id, sin1_output->id);

  auto cos1_node = graph.NewNode();
  cos1_node->operation.type = ToString(OperationType::COS);
  auto cos1_output = graph.NewValue();
  cos1_output->tensor.type = DataType::FLOAT32;
  cos1_output->tensor.shape = BHWC(1, 32, 32, 128);
  graph.AddConsumer(cos1_node->id, sub0_output->id);
  graph.SetProducer(cos1_node->id, cos1_output->id);

  auto sub1_node = graph.NewNode();
  sub1_node->operation.type = ToString(OperationType::SUB);
  auto sub1_output = graph.NewValue();
  sub1_output->tensor.type = DataType::FLOAT32;
  sub1_output->tensor.shape = BHWC(1, 32, 32, 128);
  graph.AddConsumer(sub1_node->id, sin1_output->id);
  graph.AddConsumer(sub1_node->id, cos1_output->id);
  graph.SetProducer(sub1_node->id, sub1_output->id);

  TensorStorageType storage_type = TensorStorageType::BUFFER;
  CalculationsPrecision precision = CalculationsPrecision::F32;
  DataType data_type = DeduceDataTypeFromPrecision(precision);
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage_type;
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  GpuModel gpu_model;
  RETURN_IF_ERROR(
      GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

  if (gpu_model.nodes.size() != 1) {
    return absl::InternalError("Expected model with one node.");
  }

  TensorFloat32 src_tensor;
  src_tensor.shape = input->tensor.shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i * 0.12345f);
  }

  TensorFloat32 dst_tensor_v1;
  RETURN_IF_ERROR(env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage_type, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage_type, Layout::HWC});

  OperationDef op_def_sub;
  op_def_sub.src_tensors.push_back({data_type, storage_type, Layout::HWC});
  op_def_sub.src_tensors.push_back({data_type, storage_type, Layout::HWC});
  op_def_sub.dst_tensors.push_back({data_type, storage_type, Layout::HWC});

  Reshapex4 reshape_operation = CreateReshapex4(op_def);
  TensorFloat32 input_reshaped;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      src_tensor, std::make_unique<Reshapex4>(std::move(reshape_operation)),
      reshape_output->tensor.shape, &input_reshaped));

  GPUOperation sin0_operation =
      CreateElementwiseOneInput(env->GetGpuInfo(), op_def, OperationType::SIN);
  TensorFloat32 interm0;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      input_reshaped, std::make_unique<GPUOperation>(std::move(sin0_operation)),
      sin0_output->tensor.shape, &interm0));

  GPUOperation cos0_operation =
      CreateElementwiseOneInput(env->GetGpuInfo(), op_def, OperationType::COS);
  TensorFloat32 interm1;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      input_reshaped, std::make_unique<GPUOperation>(std::move(cos0_operation)),
      cos0_output->tensor.shape, &interm1));

  TensorFloat32 interm2;
  GPUOperation sub0_operation = CreateElementwiseTwoInput(
      env->GetGpuInfo(), op_def_sub, OperationType::SUB,
      cos0_output->tensor.shape, sub0_output->tensor.shape);
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {interm0, interm1},
      std::make_unique<GPUOperation>(std::move(sub0_operation)),
      sub0_output->tensor.shape, &interm2));

  GPUOperation sin1_operation =
      CreateElementwiseOneInput(env->GetGpuInfo(), op_def, OperationType::SIN);
  TensorFloat32 interm3;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      interm2, std::make_unique<GPUOperation>(std::move(sin1_operation)),
      sin1_output->tensor.shape, &interm3));

  GPUOperation cos1_operation =
      CreateElementwiseOneInput(env->GetGpuInfo(), op_def, OperationType::COS);
  TensorFloat32 interm4;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      interm2, std::make_unique<GPUOperation>(std::move(cos1_operation)),
      cos0_output->tensor.shape, &interm4));

  TensorFloat32 interm5;
  GPUOperation sub1_operation = CreateElementwiseTwoInput(
      env->GetGpuInfo(), op_def_sub, OperationType::SUB,
      cos1_output->tensor.shape, sub1_output->tensor.shape);
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {interm3, interm4},
      std::make_unique<GPUOperation>(std::move(sub1_operation)),
      sub1_output->tensor.shape, &interm5));

  EXPECT_EQ(dst_tensor_v1.data, interm5.data);

  return absl::OkStatus();
}

//   input_tensor
//     /     \
//   conv    pad
//     \     /
//    addition
//        |
//  output_tensor
absl::Status TestConvWithPaddedAdd(TestExecutionEnvironment* env) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = BHWC(1, 32, 32, 36);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(42, 1, 1, 36);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = std::sin(i * 0.12345f);
  }
  conv_attr.bias.shape = Linear(42);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = std::sin(i * 0.12345f);
  }

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  auto conv_output = graph.NewValue();
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = BHWC(1, 32, 32, 42);
  graph.AddConsumer(conv_node->id, input->id);
  graph.SetProducer(conv_node->id, conv_output->id);

  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 0, 0);
  pad_attr.appended = BHWC(0, 0, 0, 6);
  pad_attr.type = PaddingContentType::ZEROS;

  auto pad_node = graph.NewNode();
  pad_node->operation.type = ToString(OperationType::PAD);
  pad_node->operation.attributes = pad_attr;
  auto pad_output = graph.NewValue();
  pad_output->tensor.type = DataType::FLOAT32;
  pad_output->tensor.shape = BHWC(1, 32, 32, 42);
  graph.AddConsumer(pad_node->id, input->id);
  graph.SetProducer(pad_node->id, pad_output->id);

  auto add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::ADD);
  ElementwiseAttributes add_attr;
  add_node->operation.attributes = add_attr;
  auto add_output = graph.NewValue();
  add_output->tensor.type = DataType::FLOAT32;
  add_output->tensor.shape = BHWC(1, 32, 32, 42);
  graph.AddConsumer(add_node->id, conv_output->id);
  graph.AddConsumer(add_node->id, pad_output->id);
  graph.SetProducer(add_node->id, add_output->id);

  TensorStorageType storage_type = TensorStorageType::BUFFER;
  CalculationsPrecision precision = CalculationsPrecision::F32;
  DataType data_type = DeduceDataTypeFromPrecision(precision);
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage_type;
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  TensorFloat32 src_tensor;
  src_tensor.shape = input->tensor.shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i * 0.12345f);
  }

  OperationDef op_def_conv;
  op_def_conv.src_tensors.push_back({data_type, storage_type, Layout::HWC});
  op_def_conv.dst_tensors.push_back({data_type, storage_type, Layout::HWC});

  OperationDef op_def_pad;
  op_def_pad.src_tensors.push_back({data_type, storage_type, Layout::HWC});
  op_def_pad.dst_tensors.push_back({data_type, storage_type, Layout::HWC});

  OperationDef op_def_add;
  op_def_add.src_tensors.push_back({data_type, storage_type, Layout::HWC});
  op_def_add.src_tensors.push_back({data_type, storage_type, Layout::HWC});
  op_def_add.dst_tensors.push_back({data_type, storage_type, Layout::HWC});

  ConvGeneric conv_operation =
      CreateConvGeneric(env->GetGpuInfo(), op_def_conv, precision, conv_attr);
  TensorFloat32 interm0;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(conv_operation)),
      conv_output->tensor.shape, &interm0));

  GPUOperation pad_operation =
      CreatePadding(env->GetGpuInfo(), op_def_pad, pad_attr);
  TensorFloat32 interm1;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(pad_operation)),
      pad_output->tensor.shape, &interm1));

  GPUOperation add_operation = CreateElementwiseTwoInput(
      env->GetGpuInfo(), op_def_add, OperationType::ADD,
      pad_output->tensor.shape, add_output->tensor.shape);
  TensorFloat32 interm2;
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {interm0, interm1},
      std::make_unique<GPUOperation>(std::move(add_operation)),
      add_output->tensor.shape, &interm2));

  RETURN_IF_ERROR(ApplyGpuModelTransformations(&graph));

  GpuModel gpu_model;
  RETURN_IF_ERROR(
      GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));

  if (gpu_model.nodes.size() != 1) {
    return absl::InternalError("Expected model with one node.");
  }

  TensorFloat32 dst_tensor_v1;
  RETURN_IF_ERROR(env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

  EXPECT_EQ(dst_tensor_v1.data, interm2.data);

  return absl::OkStatus();
}

}  // namespace ml_drift
