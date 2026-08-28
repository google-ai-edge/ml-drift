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

#include "ml_drift/common/kernels/special/thin_pointwise_fuser.h"

#include <set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

namespace {
absl::Status CreateLinearGraph(const BHWC& input_shape,
                               DepthwiseConvolution2DAttributes dw_attr,
                               Convolution2DAttributes conv_attr,
                               GraphFloat32* graph,
                               std::vector<ValueId>* values_ids) {
  auto input = graph->NewValue();
  input->tensor.shape = input_shape;

  auto dw_node = graph->NewNode();
  dw_node->operation.type = ToString(OperationType::DEPTHWISE_CONVOLUTION);
  dw_node->operation.attributes = dw_attr;
  graph->AddConsumer(dw_node->id, input->id);

  auto conv_node = graph->NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  Value* dw_output = nullptr;
  ABSL_RETURN_IF_ERROR(ConnectTwoNodes(graph, dw_node, conv_node, &dw_output));
  dw_output->tensor.shape = CalculateOutputShape(input->tensor.shape, dw_attr);
  Value* conv_output = nullptr;
  ABSL_RETURN_IF_ERROR(AddOutput(graph, conv_node, &conv_output));
  conv_output->tensor.shape =
      CalculateOutputShape(input->tensor.shape, conv_attr);

  values_ids->push_back(input->id);
  values_ids->push_back(dw_output->id);
  values_ids->push_back(conv_output->id);
  return absl::OkStatus();
}

absl::Status CreateLinearGraph(const BHWC& input_shape,
                               Convolution2DAttributes conv2d_attr,
                               Convolution2DAttributes conv1x1_attr,
                               GraphFloat32* graph,
                               std::vector<ValueId>* values_ids) {
  auto input = graph->NewValue();
  input->tensor.shape = input_shape;

  auto conv2d_node = graph->NewNode();
  conv2d_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv2d_node->operation.attributes = conv2d_attr;
  graph->AddConsumer(conv2d_node->id, input->id);

  auto conv1x1_node = graph->NewNode();
  conv1x1_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv1x1_node->operation.attributes = conv1x1_attr;
  Value* conv2d_output = nullptr;
  ABSL_RETURN_IF_ERROR(
      ConnectTwoNodes(graph, conv2d_node, conv1x1_node, &conv2d_output));
  conv2d_output->tensor.shape =
      CalculateOutputShape(input->tensor.shape, conv2d_attr);
  Value* conv_output = nullptr;
  ABSL_RETURN_IF_ERROR(AddOutput(graph, conv1x1_node, &conv_output));
  conv_output->tensor.shape =
      CalculateOutputShape(conv2d_output->tensor.shape, conv1x1_attr);

  values_ids->push_back(input->id);
  values_ids->push_back(conv2d_output->id);
  values_ids->push_back(conv_output->id);
  return absl::OkStatus();
}

absl::flat_hash_map<ValueId, TensorDescriptor> GetTensorDescriptors(
    const GraphFloat32& graph, DataType data_type,
    TensorStorageType storage_type) {
  absl::flat_hash_map<ValueId, TensorDescriptor> result;
  for (Value* value : graph.values()) {
    Layout layout = value->tensor.shape.b == 1 ? Layout::HWC : Layout::BHWC;
    auto tensor_desc = TensorDescriptor{data_type, storage_type, layout};
    tensor_desc.SetBHWCShape(value->tensor.shape);
    result[value->id] = tensor_desc;
  }
  return result;
}

absl::Status DWPlusConv1x1Test(TestExecutionEnvironment* exec_env,
                               const std::vector<int>& channels,
                               OperationDef op_def,
                               CalculationsPrecision precision) {
  DepthwiseConvolution2DAttributes dw_attr;
  dw_attr.padding.prepended = HW(1, 1);
  dw_attr.padding.appended = HW(1, 1);
  dw_attr.strides = HW(1, 1);
  dw_attr.dilations = HW(1, 1);
  dw_attr.weights = MakeSyntheticTensor(OHWI(1, 3, 3, channels[0]));
  dw_attr.bias = MakeSyntheticTensor(Linear(channels[0]));

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  conv_attr.weights = MakeSyntheticTensor(OHWI(channels[1], 1, 1, channels[0]));
  conv_attr.bias = MakeSyntheticTensor(Linear(channels[1]));

  auto src_shape = BHWC(1, 7, 8, channels[0]);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 dw_result = DepthWiseConvolutionReference(dw_attr, src_tensor);
  TensorFloat32 result = ConvolutionReference(conv_attr, dw_result);

  GpuModel gpu_model;
  {
    GraphFloat32 graph;
    std::vector<ValueId> values_ids;
    ABSL_RETURN_IF_ERROR(
        CreateLinearGraph(src_shape, dw_attr, conv_attr, &graph, &values_ids));

    absl::flat_hash_map<ValueId, TensorDescriptor> tensor_descriptors_copy =
        GetTensorDescriptors(graph, DeduceDataTypeFromPrecision(precision),
                             op_def.dst_tensors[0].GetStorageType());
    std::set<NodeId> consumed_nodes;
    std::set<NodeId> new_consumed_nodes;
    GpuModelBuilderOptions options = {
        .hints = {},
        .storage = op_def.dst_tensors[0].GetStorageType(),
        .use_f32_accum_for_f16_convolutions =
            precision == CalculationsPrecision::F32_F16,
    };
    GpuModelBuilder model_builder =
        GpuModelBuilder(exec_env->GetGpuInfo(), options,
                        std::move(tensor_descriptors_copy), {});
    ABSL_RETURN_IF_ERROR(TryThinPointwiseFuser(
        exec_env->GetGpuInfo(), graph, values_ids[0], consumed_nodes,
        &new_consumed_nodes, &model_builder));
    ABSL_RETURN_IF_ERROR(model_builder.GetGpuModel(
        {values_ids[0]}, {values_ids[2]}, &gpu_model));
  }

  TensorFloat32 dst_0;
  ABSL_EXPECT_OK(exec_env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_0}, &gpu_model));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.02f), result.data));

  // Batch test
  for (auto& src : op_def.src_tensors) {
    src =
        TensorDescriptor(src.GetDataType(), src.GetStorageType(), Layout::BHWC);
  }
  for (auto& dst : op_def.dst_tensors) {
    dst =
        TensorDescriptor(dst.GetDataType(), dst.GetStorageType(), Layout::BHWC);
  }

  src_shape.b = 3;
  src_tensor = MakeSyntheticTensor(src_shape);
  dw_result = DepthWiseConvolutionReference(dw_attr, src_tensor);
  result = ConvolutionReference(conv_attr, dw_result);

  GpuModel gpu_model_batched;
  {
    GraphFloat32 graph;
    std::vector<ValueId> values_ids;
    ABSL_RETURN_IF_ERROR(
        CreateLinearGraph(src_shape, dw_attr, conv_attr, &graph, &values_ids));

    absl::flat_hash_map<ValueId, TensorDescriptor> tensor_descriptors_copy =
        GetTensorDescriptors(graph, DeduceDataTypeFromPrecision(precision),
                             op_def.dst_tensors[0].GetStorageType());
    std::set<NodeId> consumed_nodes;
    std::set<NodeId> new_consumed_nodes;
    GpuModelBuilderOptions options = {
        .hints = {},
        .storage = op_def.dst_tensors[0].GetStorageType(),
        .use_f32_accum_for_f16_convolutions =
            precision == CalculationsPrecision::F32_F16,
    };
    GpuModelBuilder model_builder =
        GpuModelBuilder(exec_env->GetGpuInfo(), options,
                        std::move(tensor_descriptors_copy), {});
    ABSL_RETURN_IF_ERROR(TryThinPointwiseFuser(
        exec_env->GetGpuInfo(), graph, values_ids[0], consumed_nodes,
        &new_consumed_nodes, &model_builder));
    ABSL_RETURN_IF_ERROR(model_builder.GetGpuModel(
        {values_ids[0]}, {values_ids[2]}, &gpu_model_batched));
  }

  ABSL_EXPECT_OK(exec_env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_0}, &gpu_model_batched));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.02f), result.data));

  return absl::OkStatus();
}

absl::Status Conv2dConv1x1Test(TestExecutionEnvironment* exec_env, int kernel_x,
                               int kernel_y, const std::vector<int>& channels,
                               OperationDef op_def,
                               CalculationsPrecision precision) {
  Convolution2DAttributes conv2d_attr;
  conv2d_attr.padding.prepended = HW(kernel_x / 2, kernel_y / 2);
  conv2d_attr.padding.appended = HW(kernel_x / 2, kernel_y / 2);
  conv2d_attr.strides = HW(1, 1);
  conv2d_attr.dilations = HW(1, 1);
  conv2d_attr.weights =
      MakeSyntheticTensor(OHWI(channels[1], kernel_y, kernel_x, channels[0]));
  conv2d_attr.bias = MakeSyntheticTensor(Linear(channels[1]));

  Convolution2DAttributes conv1x1_attr;
  conv1x1_attr.padding.prepended = HW(0, 0);
  conv1x1_attr.padding.appended = HW(0, 0);
  conv1x1_attr.strides = HW(1, 1);
  conv1x1_attr.dilations = HW(1, 1);
  conv1x1_attr.weights =
      MakeSyntheticTensor(OHWI(channels[2], 1, 1, channels[1]));
  conv1x1_attr.bias = MakeSyntheticTensor(Linear(channels[2]));

  auto src_shape = BHWC(1, 7, 8, channels[0]);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 conv2d_result = ConvolutionReference(conv2d_attr, src_tensor);
  TensorFloat32 result = ConvolutionReference(conv1x1_attr, conv2d_result);

  GpuModel gpu_model;
  {
    GraphFloat32 graph;
    std::vector<ValueId> values_ids;
    ABSL_RETURN_IF_ERROR(CreateLinearGraph(src_shape, conv2d_attr, conv1x1_attr,
                                           &graph, &values_ids));

    absl::flat_hash_map<ValueId, TensorDescriptor> tensor_descriptors_copy =
        GetTensorDescriptors(graph, DeduceDataTypeFromPrecision(precision),
                             op_def.dst_tensors[0].GetStorageType());
    std::set<NodeId> consumed_nodes;
    std::set<NodeId> new_consumed_nodes;
    GpuModelBuilderOptions options = {
        .hints = {},
        .storage = op_def.dst_tensors[0].GetStorageType(),
        .use_f32_accum_for_f16_convolutions =
            precision == CalculationsPrecision::F32_F16,
    };
    GpuModelBuilder model_builder =
        GpuModelBuilder(exec_env->GetGpuInfo(), options,
                        std::move(tensor_descriptors_copy), {});
    ABSL_RETURN_IF_ERROR(TryThinPointwiseFuser(
        exec_env->GetGpuInfo(), graph, values_ids[0], consumed_nodes,
        &new_consumed_nodes, &model_builder));
    ABSL_RETURN_IF_ERROR(model_builder.GetGpuModel(
        {values_ids[0]}, {values_ids[2]}, &gpu_model));
  }

  TensorFloat32 dst_0;
  ABSL_EXPECT_OK(exec_env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_0}, &gpu_model));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.02f), result.data));

  // Batch test
  for (auto& src : op_def.src_tensors) {
    src =
        TensorDescriptor(src.GetDataType(), src.GetStorageType(), Layout::BHWC);
  }
  for (auto& dst : op_def.dst_tensors) {
    dst =
        TensorDescriptor(dst.GetDataType(), dst.GetStorageType(), Layout::BHWC);
  }

  src_shape.b = 3;
  src_tensor = MakeSyntheticTensor(src_shape);
  conv2d_result = ConvolutionReference(conv2d_attr, src_tensor);
  result = ConvolutionReference(conv1x1_attr, conv2d_result);

  GpuModel gpu_model_batched;
  {
    GraphFloat32 graph;
    std::vector<ValueId> values_ids;
    ABSL_RETURN_IF_ERROR(CreateLinearGraph(src_shape, conv2d_attr, conv1x1_attr,
                                           &graph, &values_ids));

    absl::flat_hash_map<ValueId, TensorDescriptor> tensor_descriptors_copy =
        GetTensorDescriptors(graph, DeduceDataTypeFromPrecision(precision),
                             op_def.dst_tensors[0].GetStorageType());
    std::set<NodeId> consumed_nodes;
    std::set<NodeId> new_consumed_nodes;
    GpuModelBuilderOptions options = {
        .hints = {},
        .storage = op_def.dst_tensors[0].GetStorageType(),
        .use_f32_accum_for_f16_convolutions =
            precision == CalculationsPrecision::F32_F16,
    };
    GpuModelBuilder model_builder =
        GpuModelBuilder(exec_env->GetGpuInfo(), options,
                        std::move(tensor_descriptors_copy), {});
    ABSL_RETURN_IF_ERROR(TryThinPointwiseFuser(
        exec_env->GetGpuInfo(), graph, values_ids[0], consumed_nodes,
        &new_consumed_nodes, &model_builder));
    ABSL_RETURN_IF_ERROR(model_builder.GetGpuModel(
        {values_ids[0]}, {values_ids[2]}, &gpu_model_batched));
  }

  ABSL_EXPECT_OK(exec_env->ExecuteGpuModel(
      {src_tensor}, std::vector<TensorFloat32*>{&dst_0}, &gpu_model_batched));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.02f), result.data));

  return absl::OkStatus();
}

}  // namespace

TEST_P(FloatTest, DWPlusConv1x1_4_4) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
  ABSL_EXPECT_OK(DWPlusConv1x1Test(exec_env, {4, 4}, op_def, precision()));
}

TEST_P(FloatTest, DWPlusConv1x1_3_11) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
  ABSL_EXPECT_OK(DWPlusConv1x1Test(exec_env, {3, 11}, op_def, precision()));
}

TEST_P(FloatTest, DWPlusConv1x1_7_9) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
  ABSL_EXPECT_OK(DWPlusConv1x1Test(exec_env, {7, 9}, op_def, precision()));
}

TEST_P(FloatTest, Conv3x2from4to4Conv1x1from4to8) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
  ABSL_EXPECT_OK(Conv2dConv1x1Test(exec_env, 3, 2, {4, 4, 8}, op_def, precision()));
}

TEST_P(FloatTest, Conv1x1from8to4Conv1x1from4to8) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
  ABSL_EXPECT_OK(Conv2dConv1x1Test(exec_env, 1, 1, {8, 4, 8}, op_def, precision()));
}

INSTANTIATE_TEST_SUITE_P(FloatTestSuite, FloatTest,
                         Combine(ValuesIn(GetCalculationsPrecisions()),
                                 ValuesIn(GetTensorStoragesTypes())),
                         [](const TestParamInfo<FloatTest::ParamType>& info) {
                           return ToString(info.param);
                         });

}  // namespace ml_drift
