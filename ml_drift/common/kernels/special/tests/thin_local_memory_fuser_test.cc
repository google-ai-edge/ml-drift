// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/special/thin_local_memory_fuser.h"

#include <iostream>
#include <memory>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/container/flat_hash_map.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

namespace {
absl::Status CreateGraph(const BHWC& resize_shape, const BHWC& add_shape,
                         Resize2DAttributes resize_attr,
                         Convolution2DAttributes conv_attr, GraphFloat32* graph,
                         std::vector<ValueId>* values_ids) {
  auto resize_input = graph->NewValue();
  resize_input->tensor.shape = resize_shape;
  auto add_input = graph->NewValue();
  add_input->tensor.shape = add_shape;

  auto resize_node = graph->NewNode();
  resize_node->operation.type = ToString(OperationType::RESIZE);
  resize_node->operation.attributes = resize_attr;
  graph->AddConsumer(resize_node->id, resize_input->id);
  auto resize_output = graph->NewValue();
  resize_output->tensor.shape =
      CalculateOutputShape(resize_input->tensor.shape, resize_attr);
  graph->SetProducer(resize_node->id, resize_output->id);

  auto add_node = graph->NewNode();
  add_node->operation.type = ToString(OperationType::ADD);
  graph->AddConsumer(add_node->id, resize_output->id);
  graph->AddConsumer(add_node->id, add_input->id);

  auto add_output = graph->NewValue();
  add_output->tensor.shape = add_shape;
  graph->SetProducer(add_node->id, add_output->id);

  auto conv_node = graph->NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  graph->AddConsumer(conv_node->id, add_output->id);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(AddOutput(graph, conv_node, &conv_output));
  conv_output->tensor.shape =
      CalculateOutputShape(add_output->tensor.shape, conv_attr);

  values_ids->push_back(resize_input->id);
  values_ids->push_back(resize_output->id);
  values_ids->push_back(add_input->id);
  values_ids->push_back(add_output->id);
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

absl::Status ResizeAddConv(TestExecutionEnvironment* exec_env,
                           const OHWI& weights_shape, const BHWC& resize_shape,
                           const BHWC& add_shape, OperationDef op_def,
                           bool perf_test = false) {
  Resize2DAttributes resize_attr;
  resize_attr.align_corners = false;
  resize_attr.half_pixel_centers = true;
  resize_attr.type = SamplingType::BILINEAR;
  resize_attr.new_shape = HW(add_shape.h, add_shape.w);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(1, 1);
  conv_attr.padding.appended = HW(1, 1);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  conv_attr.weights = MakeSyntheticTensor(weights_shape);
  conv_attr.bias = MakeSyntheticTensor(Linear(weights_shape.o));

  auto data_type = op_def.src_tensors[0].GetDataType();
  auto precision = data_type == DataType::FLOAT16 ? CalculationsPrecision::F16
                                                  : CalculationsPrecision::F32;

  GpuModel gpu_model;
  {
    GraphFloat32 graph;
    std::vector<ValueId> values_ids;
    RETURN_IF_ERROR(CreateGraph(resize_shape, add_shape, resize_attr, conv_attr,
                                &graph, &values_ids));

    std::set<NodeId> consumed_nodes;
    std::set<NodeId> new_consumed_nodes;
    GpuModelBuilderOptions options = {
        .hints = {},
        .storage = op_def.dst_tensors[0].GetStorageType(),
        .use_f32_accum_for_f16_convolutions =
            precision == CalculationsPrecision::F32_F16,
    };
    GpuModelBuilder model_builder = GpuModelBuilder(
        exec_env->GetGpuInfo(), options,
        GetTensorDescriptors(graph, data_type,
                             op_def.dst_tensors[0].GetStorageType()),
        {});
    auto s = TryThinLocalMemoryFuser(exec_env->GetGpuInfo(), graph,
                                     values_ids[0], consumed_nodes,
                                     &new_consumed_nodes, &model_builder);
    RETURN_IF_ERROR(model_builder.GetGpuModel({values_ids[0], values_ids[2]},
                                              {values_ids[4]}, &gpu_model));
    if (!s.ok()) {
      std::cout << "This case not supported on this device." << std::endl;
      return absl::OkStatus();
    }
  }

  TensorFloat32 src0_tensor = MakeSyntheticTensor(resize_shape);
  TensorFloat32 src1_tensor = MakeSyntheticTensor(add_shape);

  if (perf_test) {
    const BHWC dst_shape = CalculateOutputShape(add_shape, conv_attr);
    const int num_repeats = 5;
    auto operation = std::move(gpu_model.nodes[0].gpu_operation);
    ASSIGN_OR_RETURN(auto report,
                     exec_env->GetGpuOperationTimeMs({src0_tensor, src1_tensor},
                                                     std::move(operation),
                                                     {dst_shape}, num_repeats));

    std::cout << report.GetDetailedReport() << std::endl;
    return absl::OkStatus();
  }

  TensorFloat32 resize_result = ResizeReference(resize_attr, src0_tensor);
  TensorFloat32 add_result = AddTableReference({resize_result, src1_tensor});
  TensorFloat32 dst_ref = ConvolutionReference(conv_attr, add_result);

  TensorFloat32 dst_gpu;
  MLD_EXPECT_OK(exec_env->ExecuteGpuModel({src0_tensor, src1_tensor},
                                      std::vector<TensorFloat32*>{&dst_gpu},
                                      &gpu_model));
  float eps = GetEpsilon(precision, exec_env->GetGpuInfo(), conv_attr);
  EXPECT_THAT(dst_gpu.data, Pointwise(FloatNear(eps), dst_ref.data));

  return absl::OkStatus();
}

}  // namespace

// Perf tests. No parameterization.
TEST_F(Test, PerfResizeAddConv1x3x3x4) {
  const DataType data_type = DataType::FLOAT16;
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  MLD_EXPECT_OK(ResizeAddConv(exec_env, OHWI(1, 3, 3, 4), BHWC(1, 1512, 2016, 4),
                          BHWC(1, 3024, 4032, 4), op_def, /*perf_test=*/true));
}

TEST_F(Test, PerfResizeAddConv4x3x3x8) {
  const DataType data_type = DataType::FLOAT16;
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  MLD_EXPECT_OK(ResizeAddConv(exec_env, OHWI(4, 3, 3, 8), BHWC(1, 756, 1008, 8),
                          BHWC(1, 1512, 2016, 8), op_def, /*perf_test=*/true));
}

// Correctness tests. Parameterized.
TEST_P(DataTypeTest, ResizeAddConv1x3x3x4) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  MLD_EXPECT_OK(ResizeAddConv(exec_env, OHWI(1, 3, 3, 4), BHWC(1, 31, 49, 4),
                          BHWC(1, 62, 98, 4), op_def));
}

TEST_P(DataTypeTest, ResizeAddConv4x3x3x8) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  MLD_EXPECT_OK(ResizeAddConv(exec_env, OHWI(4, 3, 3, 8), BHWC(1, 31, 49, 8),
                          BHWC(1, 62, 98, 8), op_def));
}

INSTANTIATE_TEST_SUITE_P(
    FloatTestSuite, DataTypeTest,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D())),
    [](const TestParamInfo<DataTypeTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
