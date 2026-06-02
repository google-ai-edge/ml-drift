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

#include "ml_drift/common/kernels/special/concat_conv.h"

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
#include "ml_drift/common/kernels/special/thin_local_memory_fuser.h"
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

using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::Test;

namespace {
absl::Status CreateGraph(const BHWC& src_shape,
                         Convolution2DAttributes conv_attr, GraphFloat32* graph,
                         std::vector<ValueId>* values_ids) {
  auto concat_input0 = graph->NewValue();
  auto concat_input1 = graph->NewValue();
  auto concat_input2 = graph->NewValue();
  concat_input0->tensor.shape = src_shape;
  concat_input1->tensor.shape = src_shape;
  concat_input2->tensor.shape = src_shape;

  ConcatAttributes concat_attr;
  concat_attr.axis = Axis::CHANNELS;

  auto concat_node = graph->NewNode();
  concat_node->operation.type = ToString(OperationType::CONCAT);
  concat_node->operation.attributes = concat_attr;
  graph->AddConsumer(concat_node->id, concat_input0->id);
  graph->AddConsumer(concat_node->id, concat_input1->id);
  graph->AddConsumer(concat_node->id, concat_input2->id);
  auto concat_output = graph->NewValue();
  concat_output->tensor.shape = BHWC(src_shape.b, src_shape.h, src_shape.w, 3);
  graph->SetProducer(concat_node->id, concat_output->id);

  auto conv_node = graph->NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  graph->AddConsumer(conv_node->id, concat_output->id);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(AddOutput(graph, conv_node, &conv_output));
  conv_output->tensor.shape =
      CalculateOutputShape(concat_output->tensor.shape, conv_attr);

  values_ids->push_back(concat_input0->id);
  values_ids->push_back(concat_input1->id);
  values_ids->push_back(concat_input2->id);
  values_ids->push_back(concat_output->id);
  values_ids->push_back(conv_output->id);
  return absl::OkStatus();
}

absl::flat_hash_map<ValueId, TensorDescriptor> GetTensorDescriptors(
    const GraphFloat32& graph, CalculationsPrecision precision,
    TensorStorageType storage_type) {
  absl::flat_hash_map<ValueId, TensorDescriptor> result;
  for (Value* value : graph.values()) {
    Layout layout = value->tensor.shape.b == 1 ? Layout::HWC : Layout::BHWC;
    DataType data_type = DeduceDataTypeFromPrecision(precision);
    auto tensor_desc = TensorDescriptor{data_type, storage_type, layout};
    tensor_desc.SetBHWCShape(value->tensor.shape);
    result[value->id] = tensor_desc;
  }
  return result;
}

absl::Status ConcatConvTest(TestExecutionEnvironment* exec_env,
                            const BHWC& src_shape, OperationDef op_def,
                            CalculationsPrecision precision,
                            bool use_thin_local_memory_fuser = false,
                            bool perf_test = false) {
  ConcatAttributes concat_attr;
  concat_attr.axis = Axis::CHANNELS;

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(1, 1);
  conv_attr.padding.appended = HW(1, 1);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  conv_attr.weights = MakeSyntheticTensor(OHWI(4, 3, 3, 3));
  conv_attr.bias = MakeSyntheticTensor(Linear(4));

  std::unique_ptr<GPUOperation> operation;
  {
    GraphFloat32 graph;
    std::vector<ValueId> values_ids;
    RETURN_IF_ERROR(CreateGraph(src_shape, conv_attr, &graph, &values_ids));

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
        GetTensorDescriptors(graph, precision,
                             op_def.dst_tensors[0].GetStorageType()),
        {});
    if (use_thin_local_memory_fuser) {
      RETURN_IF_ERROR(TryThinLocalMemoryFuser(
          exec_env->GetGpuInfo(), graph, values_ids[0], consumed_nodes,
          &new_consumed_nodes, &model_builder));
    } else {
      RETURN_IF_ERROR(TryConcatConv(exec_env->GetGpuInfo(), graph,
                                    values_ids[0], consumed_nodes,
                                    &new_consumed_nodes, &model_builder));
    }
    GpuModel gpu_model;
    RETURN_IF_ERROR(model_builder.GetGpuModel({values_ids[0], values_ids[2]},
                                              {values_ids[4]}, &gpu_model));
    operation = std::move(gpu_model.nodes[0].gpu_operation);
  }

  TensorFloat32 src0_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 src1_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 src2_tensor = MakeSyntheticTensor(src_shape);

  if (perf_test) {
    BHWC interm_shape = src0_tensor.shape;
    interm_shape.c =
        src0_tensor.shape.c + src1_tensor.shape.c + src2_tensor.shape.c;
    const BHWC dst_shape = CalculateOutputShape(interm_shape, conv_attr);
    const int num_repeats = 5;
    ASSIGN_OR_RETURN(auto report,
                     exec_env->GetGpuOperationTimeMs(
                         {src0_tensor, src1_tensor, src2_tensor},
                         std::move(operation), {dst_shape}, num_repeats));

    std::cout << report.GetDetailedReport() << std::endl;
    return absl::OkStatus();
  }

  TensorFloat32 concat_result =
      ConcatReference(concat_attr, {src0_tensor, src1_tensor, src2_tensor});
  TensorFloat32 dst_ref = ConvolutionReference(conv_attr, concat_result);

  TensorFloat32 dst_gpu;
  MLD_EXPECT_OK(exec_env->ExecuteGPUOperation(
      {src0_tensor, src1_tensor, src2_tensor}, std::move(operation),
      dst_ref.shape, &dst_gpu));
  float eps = GetEpsilon(precision, exec_env->GetGpuInfo(), conv_attr);
  EXPECT_THAT(dst_gpu.data, Pointwise(FloatNear(eps), dst_ref.data));

  return absl::OkStatus();
}

}  // namespace

TEST_F(Test, ConcatConv4x3x3x3) {
  if (!IsConcatConvRecommended(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "ConcatConv not suitable.";
  }
  OperationDef op_def;
  const DataType data_type = DataType::FLOAT32;
  const TensorStorageType storage = TensorStorageType::TEXTURE_2D;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(ConcatConvTest(exec_env, BHWC(1, 31, 49, 1), op_def,
                           CalculationsPrecision::F32,
                           /*use_thin_local_memory_fuser=*/false));
}

TEST_F(Test, ConcatConv4x3x3x3Perf) {
  if (!exec_env->IsStorageSupported(TensorStorageType::SINGLE_TEXTURE_2D,
                                    DataType::FLOAT16)) {
    GTEST_SKIP() << "ConcatConv4x3x3x3Perf not suitable.";
  }
  const DataType data_type = DataType::FLOAT16;
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  MLD_EXPECT_OK(ConcatConvTest(exec_env, BHWC(1, 3024, 4032, 1), op_def,
                           CalculationsPrecision::F16,
                           /*use_thin_local_memory_fuser=*/false,
                           /*perf_test=*/true));
}

TEST_F(Test, ConcatConv4x3x3x3LocalMemory) {
  if (!IsThinLocalMemoryFuserRecommended(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "ThinLocalMemoryFuser not suitable.";
  }
  OperationDef op_def;
  const DataType data_type = DataType::FLOAT32;
  const TensorStorageType storage = TensorStorageType::TEXTURE_2D;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(ConcatConvTest(exec_env, BHWC(1, 31, 49, 1), op_def,
                           CalculationsPrecision::F32,
                           /*use_thin_local_memory_fuser=*/true));
}

TEST_F(Test, ConcatConv4x3x3x3LocalMemoryPerf) {
  if (!IsThinLocalMemoryFuserRecommended(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "ThinLocalMemoryFuser not suitable.";
  }
  const DataType data_type = DataType::FLOAT16;
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  MLD_EXPECT_OK(ConcatConvTest(exec_env, BHWC(1, 3024, 4032, 1), op_def,
                           CalculationsPrecision::F16,
                           /*use_thin_local_memory_fuser=*/true,
                           /*perf_test=*/true));
}

}  // namespace ml_drift
