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

#include "ml_drift/common/gpu_model_external_weights_test_util.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

absl::Status TestDynamicConvolution(TestExecutionEnvironment* env) {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::BUFFER;

  const BHWC input_shape = BHWC(1, 32, 32, 128);
  const BHWC output_shape = BHWC(1, 32, 32, 16);
  GpuModel gpu_model;
  Convolution2DAttributes conv_attr;
  auto& attr_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  {
    GraphFloat32 graph;
    auto input = graph.NewValue();
    input->tensor.type = DataType::FLOAT32;
    input->tensor.shape = input_shape;

    auto conv_node = graph.NewNode();
    conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);

    conv_attr.padding.prepended = HW(0, 0);
    conv_attr.padding.appended = HW(0, 0);
    conv_attr.strides = HW(1, 1);
    conv_attr.dilations = HW(1, 1);
    attr_weights.shape = OHWI(output_shape.c, 1, 1, input_shape.c);
    attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                             XNN_EXTRA_BYTES / sizeof(float));
    for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
      attr_weights.data[i] = std::sin(i * 0.12345f);
    }
    conv_attr.bias.shape = Linear(output_shape.c);
    conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
    for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
      conv_attr.bias.data[i] = std::sin(i * 0.12345f);
    }
    conv_node->operation.attributes = conv_attr;
    graph.AddConsumer(conv_node->id, input->id);
    Value* conv_output = nullptr;
    RETURN_IF_ERROR(AddOutput(&graph, conv_node, &conv_output));
    conv_output->tensor.type = DataType::FLOAT32;
    conv_output->tensor.shape = output_shape;

    RETURN_IF_ERROR(
        GraphToGpuModel(graph, create_info, env->GetGpuInfo(), &gpu_model));
  }

  GpuModel gpu_model_dynamic_conv;
  {
    GraphFloat32 graph_dynamic_conv;
    auto input = graph_dynamic_conv.NewValue();
    input->tensor.type = DataType::FLOAT32;
    input->tensor.shape = input_shape;

    auto const_node = graph_dynamic_conv.NewNode();
    const_node->operation.type = ToString(OperationType::CONSTANT);
    TensorFloat32 const_data;
    const_data.shape = BHWC(attr_weights.shape.o, attr_weights.shape.h,
                            attr_weights.shape.w, attr_weights.shape.i);
    const_data.data.resize(const_data.shape.DimensionsProduct());
    for (int i = 0; i < const_data.data.size(); ++i) {
      const_data.data[i] = attr_weights.data[i];
    }
    ConstTensorAttributes const_attr;
    const_attr.tensor = const_data;
    const_node->operation.attributes = const_attr;

    Value* const_value = graph_dynamic_conv.NewValue();
    graph_dynamic_conv.SetProducer(const_node->id, const_value->id);
    const_value->tensor.type = const_data.kType;
    const_value->tensor.shape = const_data.shape;

    auto conv_node = graph_dynamic_conv.NewNode();
    conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
    attr_weights.data.clear();
    conv_node->operation.attributes = conv_attr;
    graph_dynamic_conv.AddConsumer(conv_node->id, input->id);
    graph_dynamic_conv.AddConsumer(conv_node->id, const_value->id);
    Value* conv_output = nullptr;
    RETURN_IF_ERROR(AddOutput(&graph_dynamic_conv, conv_node, &conv_output));

    conv_output->tensor.type = DataType::FLOAT32;
    conv_output->tensor.shape = output_shape;
    RETURN_IF_ERROR(GraphToGpuModel(graph_dynamic_conv, create_info,
                                    env->GetGpuInfo(),
                                    &gpu_model_dynamic_conv));
  }

  TensorFloat32 src_tensor;
  src_tensor.shape = input_shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i * 0.12345f);
  }

  TensorFloat32 dst_tensor_v1;
  RETURN_IF_ERROR(env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1}, &gpu_model));

  TensorFloat32 dst_tensor_v2;
  RETURN_IF_ERROR(env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v2},
      &gpu_model_dynamic_conv));

  EXPECT_EQ(dst_tensor_v1.data, dst_tensor_v2.data);
  return absl::OkStatus();
}

absl::Status TestExternalConvWeights(TestExecutionEnvironment* env) {
  TensorStorageType storage_type = TensorStorageType::BUFFER;
  CalculationsPrecision precision = CalculationsPrecision::F32;
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage_type;
  DataType float_type = DeduceDataTypeFromPrecision(create_info.precision);

  for (const auto src_shape : {BHWC(1, 1, 1, 128), BHWC(1, 1, 128, 64)}) {
    GpuModelBuilder model_builder(env->GetGpuInfo(), create_info.hints,
                                  create_info.precision,
                                  create_info.storage_type);

    constexpr int kDstChannels = 128;

    Convolution2DAttributes conv_attr;
    conv_attr.padding.prepended = HW(0, 0);
    conv_attr.padding.appended = HW(0, 0);
    conv_attr.strides = HW(1, 1);
    conv_attr.dilations = HW(1, 1);
    auto& attr_weights =
        conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    attr_weights.shape = OHWI(kDstChannels, 1, 1, src_shape.c);
    attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                             XNN_EXTRA_BYTES / sizeof(float));
    for (int i = 0; i < attr_weights.data.size(); ++i) {
      attr_weights.data[i] = std::sin(i * 0.12345f);
    }
    conv_attr.bias.shape = Linear(kDstChannels);
    conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
    for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
      conv_attr.bias.data[i] = std::sin(i * 0.12345f);
    }

    TensorFloat32 src_tensor;
    src_tensor.shape = src_shape;
    src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
    for (int i = 0; i < src_tensor.data.size(); ++i) {
      src_tensor.data[i] = std::sin(i * 0.123f);
    }

    GpuModel gpu_model;
    {
      auto src_tensor_handle = model_builder.AddTensor(src_shape, float_type);
      auto out = model_builder.Convolution(src_tensor_handle, conv_attr);

      RETURN_IF_ERROR(model_builder.GetGpuModel(
          std::vector<unsigned int>{src_tensor_handle.id},
          std::vector<unsigned int>{out.id}, &gpu_model));
    }

    GpuModel gpu_model_external;
    {
      TensorDescriptor bias_td = CreateConstantLinearTensorDescriptor(
          env->GetGpuInfo(), float_type, conv_attr.bias);

      TensorDescriptor weights_td;
      WeightsDescription weights_desc =
          model_builder.GetFullyConnectedWeightsDesc(float_type,
                                                     attr_weights.shape);
      {
        weights_td = TensorDescriptor(
            weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
        std::vector<uint8_t> data(
            GetTotalElementsCountForLayout(weights_desc, attr_weights.shape) *
            SizeOf(weights_desc.type));
        RearrangeWeights(attr_weights, weights_desc, absl::MakeSpan(data));

        weights_td.SetBHWCShape(
            BHWC(1, 1, 1, data.size() / SizeOf(weights_desc.type)));
        weights_td.UploadDataRaw(absl::MakeConstSpan(data));
      }

      auto src_tensor_handle = model_builder.AddTensor(src_shape, float_type);
      auto weights_th = model_builder.AddConstantTensor(std::move(weights_td));
      auto bias_th = model_builder.AddConstantTensor(std::move(bias_td));
      const GpuModelBuilder::Weights external_weights =
          CreateExternalWeights(weights_th, weights_desc, attr_weights.shape);
      ASSIGN_OR_RETURN(auto out,
                       model_builder.FullyConnectedExternalWeights(
                           src_tensor_handle, external_weights, &bias_th));

      RETURN_IF_ERROR(model_builder.GetGpuModel(
          std::vector<unsigned int>{src_tensor_handle.id},
          std::vector<unsigned int>{out.id}, &gpu_model_external));
    }

    TensorFloat32 dst_tensor_v0;
    RETURN_IF_ERROR(env->ExecuteGpuModel(
        {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v0}, &gpu_model));

    TensorFloat32 dst_tensor_v1;
    RETURN_IF_ERROR(env->ExecuteGpuModel(
        {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_v1},
        &gpu_model_external));

    EXPECT_EQ(dst_tensor_v0.data, dst_tensor_v1.data);
  }
  return absl::OkStatus();
}

absl::Status TestFullyConnected(TestExecutionEnvironment* env,
                                const BHWC& src_shape) {
  TensorStorageType storage_type = TensorStorageType::BUFFER;
  CalculationsPrecision precision = CalculationsPrecision::F32;
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage_type;
  DataType float_type = DeduceDataTypeFromPrecision(create_info.precision);

  GpuModelBuilder model_builder(env->GetGpuInfo(), create_info.hints,
                                create_info.precision,
                                create_info.storage_type);
  GpuModel gpu_model;

  auto src_th = model_builder.AddTensor(src_shape, float_type);
  FullyConnectedAttributes fc_attr;
  fc_attr.weights.shape = OHWI(16, 1, 1, 16);
  fc_attr.weights.data.resize(fc_attr.weights.shape.DimensionsProduct() +
                              XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < fc_attr.weights.data.size(); ++i) {
    fc_attr.weights.data[i] = std::sin(i * 0.12345f);
  }
  fc_attr.bias.shape = Linear(16);
  fc_attr.bias.data.resize(fc_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < fc_attr.bias.data.size(); ++i) {
    fc_attr.bias.data[i] = std::sin(i * 0.12345f);
  }
  auto out = model_builder.FullyConnected(src_th, fc_attr);

  RETURN_IF_ERROR(
      model_builder.GetGpuModel(std::vector<unsigned int>{src_th.id},
                                std::vector<unsigned int>{out.id}, &gpu_model));

  TensorFloat32 src_tensor;
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i * 0.123f);
  }

  TensorFloat32 dst_tensor_gpu;
  RETURN_IF_ERROR(env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_tensor_gpu}, &gpu_model));

  TensorFloat32 dst_ref_tensor = FullyConnectedReference(fc_attr, src_tensor);
  const float eps = GetEpsilon(precision, env->GetGpuInfo(), fc_attr);
  EXPECT_THAT(dst_tensor_gpu.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));

  return absl::OkStatus();
}

}  // namespace ml_drift
