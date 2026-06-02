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

#include "ml_drift/common/merge_nodes.h"

#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/container/flat_hash_set.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/kernels/transpose.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

void CreateGpuModelContainingLinkableNode(const GpuInfo& gpu_info,
                                          GpuModel& gpu_model) {
  //    input         input
  //      |             |
  //    conv            |
  //      |    -->  conv->relu
  //    relu            |
  //      |             |
  //    output        output
  // The created GpuModel's nodes can be merged by LinkNodes().
  gpu_model.nodes.resize(2);

  TensorDescriptor in_tensor_desc;
  TensorDescriptor tmp_tensor_desc;
  TensorDescriptor out_tensor_desc;

  OperationDef conv_definition;
  conv_definition.src_tensors.push_back(in_tensor_desc);
  conv_definition.dst_tensors.push_back(tmp_tensor_desc);
  ElementwiseDescriptor conv_descriptor;
  GPUOperation conv =
      CreateGpuOperation(conv_definition, std::move(conv_descriptor));

  OperationDef relu_definition;
  relu_definition.src_tensors.push_back(tmp_tensor_desc);
  relu_definition.dst_tensors.push_back(out_tensor_desc);
  ElementwiseDescriptor relu_descriptor;
  GPUOperation relu =
      CreateGpuOperation(relu_definition, std::move(relu_descriptor));

  cl::Tensor in_tensor;
  cl::Tensor tmp_tensor;
  cl::Tensor out_tensor;
  conv.SetSrc(&in_tensor, /*index=*/0);
  conv.SetDst(&tmp_tensor, /*index=*/0);
  relu.SetSrc(&tmp_tensor, /*index=*/0);
  relu.SetDst(&out_tensor, /*index=*/0);
  gpu_model.nodes[0].gpu_operation =
      std::make_unique<GPUOperation>(std::move(conv));
  gpu_model.nodes[0].inputs.resize(1);
  gpu_model.nodes[0].outputs.resize(1);
  gpu_model.nodes[0].name = "conv";
  gpu_model.nodes[1].gpu_operation =
      std::make_unique<GPUOperation>(std::move(relu));
  gpu_model.nodes[1].inputs.resize(1);
  gpu_model.nodes[1].outputs.resize(1);
  gpu_model.nodes[1].name = "relu";
}

TEST(MergeNodesTest, MergeSingleLinkable) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateGpuModelContainingLinkableNode(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 2);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "conv -> relu");
}

TEST(MergeNodesTest, MergeSingleLinkableWithOptional) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateGpuModelContainingLinkableNode(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 2);
  gpu_model.nodes[0].optional_tag.insert(1);
  gpu_model.nodes[1].optional_tag.insert(1);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "conv -> relu");
  EXPECT_EQ(gpu_model.nodes[0].optional_tag, absl::flat_hash_set<int>{1});
}

TEST(MergeNodesTest, NotMergeSingleLinkableWithDiffTag) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateGpuModelContainingLinkableNode(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 2);

  // The two nodes could not be merged because they have different optional tag.
  gpu_model.nodes[0].optional_tag.insert(1);
  gpu_model.nodes[1].optional_tag = {};

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_EQ(gpu_model.nodes[0].name, "conv");
  EXPECT_EQ(gpu_model.nodes[1].name, "relu");
}

void CreateMergeableGpuModelWithTwoReorderNodes(const GpuInfo& gpu_info,
                                                GpuModel& gpu_model) {
  //    input                   input
  //      |                       |
  //   transpose0                 |
  //      |         --> transpose0->transpose1
  //   transpose1                 |
  //      |                       |
  //    output                  output
  // The created GpuModel's nodes can be merged by MergeReorderNodes().
  gpu_model.nodes.resize(2);

  TensorDescriptor in_tensor_desc;
  TensorDescriptor tmp_tensor_desc;
  TensorDescriptor out_tensor_desc;

  TransposeAttributes attr;
  attr.perm = BHWC(0, 2, 1, 3);
  OperationDef transpose_def0;
  transpose_def0.src_tensors.push_back(in_tensor_desc);
  transpose_def0.dst_tensors.push_back(tmp_tensor_desc);
  OperationDef transpose_def1;
  transpose_def1.src_tensors.push_back(tmp_tensor_desc);
  transpose_def1.dst_tensors.push_back(out_tensor_desc);
  GPUOperation transpose0 = CreateTranspose(transpose_def0, attr);
  GPUOperation transpose1 = CreateTranspose(transpose_def1, attr);
  EXPECT_TRUE(transpose0.IsReorderOp());

  cl::Tensor in_tensor;
  cl::Tensor tmp_tensor;
  cl::Tensor out_tensor;
  transpose0.SetSrc(&in_tensor, /*index=*/0);
  transpose0.SetDst(&tmp_tensor, /*index=*/0);
  transpose1.SetSrc(&tmp_tensor, /*index=*/0);
  transpose1.SetDst(&out_tensor, /*index=*/0);
  gpu_model.nodes[0].gpu_operation =
      std::make_unique<GPUOperation>(std::move(transpose0));
  gpu_model.nodes[0].inputs.resize(1);
  gpu_model.nodes[0].outputs.resize(1);
  gpu_model.nodes[0].name = "transpose0";
  gpu_model.nodes[1].gpu_operation =
      std::make_unique<GPUOperation>(std::move(transpose1));
  gpu_model.nodes[1].inputs.resize(1);
  gpu_model.nodes[1].outputs.resize(1);
  gpu_model.nodes[1].name = "transpose1";
}

TEST(MergeNodesTest, MergeTwoReorder) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateMergeableGpuModelWithTwoReorderNodes(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 2);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "transpose0 -> transpose1");
}

TEST(MergeNodesTest, NotMergeTwoReorderWithDiffTag) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateMergeableGpuModelWithTwoReorderNodes(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 2);

  // The two nodes could not be merged because they have different optional tag.
  gpu_model.nodes[0].optional_tag.insert(1);
  gpu_model.nodes[1].optional_tag = {};

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_EQ(gpu_model.nodes[0].name, "transpose0");
  EXPECT_EQ(gpu_model.nodes[1].name, "transpose1");
}

void CreateTwoElementwiseGpuModel(const GpuInfo& gpu_info,
                                  GpuModel& gpu_model) {
  //    input       input
  //      |           |
  //     add          |
  //      |    --> add->relu
  //     relu         |
  //      |           |
  //    output      output
  // The created GpuModel's nodes can be merged by MergeElementwiseNodes().
  gpu_model.nodes.resize(2);

  TensorDescriptor in_tensor_desc;
  TensorDescriptor tmp_tensor_desc;
  TensorDescriptor out_tensor_desc;

  OperationDef add_definition;
  add_definition.src_tensors.push_back(tmp_tensor_desc);
  add_definition.dst_tensors.push_back(out_tensor_desc);
  ElementwiseDescriptor add_descriptor;
  GPUOperation add =
      CreateGpuOperation(add_definition, std::move(add_descriptor));

  OperationDef relu_definition;
  relu_definition.src_tensors.push_back(tmp_tensor_desc);
  relu_definition.dst_tensors.push_back(out_tensor_desc);
  ElementwiseDescriptor relu_descriptor;
  GPUOperation relu =
      CreateGpuOperation(relu_definition, std::move(relu_descriptor));

  cl::Tensor in_tensor;
  cl::Tensor tmp_tensor;
  cl::Tensor out_tensor;
  add.SetSrc(&in_tensor, /*index=*/0);
  add.SetDst(&tmp_tensor, /*index=*/0);
  relu.SetSrc(&tmp_tensor, /*index=*/0);
  relu.SetDst(&out_tensor, /*index=*/0);
  gpu_model.nodes[0].gpu_operation =
      std::make_unique<GPUOperation>(std::move(add));
  gpu_model.nodes[0].inputs.resize(1);
  gpu_model.nodes[0].outputs.resize(1);
  gpu_model.nodes[0].name = "add";
  gpu_model.nodes[1].gpu_operation =
      std::make_unique<GPUOperation>(std::move(relu));
  gpu_model.nodes[1].inputs.resize(1);
  gpu_model.nodes[1].outputs.resize(1);
  gpu_model.nodes[1].name = "relu";
}

TEST(MergeNodesTest, MergeTwoElementwise) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateTwoElementwiseGpuModel(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 2);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "add -> relu");
}

TEST(MergeNodesTest, NotMergeTwoElementwiseWithDiffTag) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateTwoElementwiseGpuModel(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 2);

  // The two nodes could not be merged because they have different optional tag.
  gpu_model.nodes[0].optional_tag.insert(1);
  gpu_model.nodes[1].optional_tag = {};

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_EQ(gpu_model.nodes[0].name, "add");
  EXPECT_EQ(gpu_model.nodes[1].name, "relu");
}

void CreateMergeableSumReluAddGpuModel(const GpuInfo& gpu_info,
                                       GpuModel& gpu_model) {
  //      sum              sum
  //     /    \             |
  //   relu    |            |
  //     \    /   --1->  relu->add  --2->  sum->relu->add
  //       add              |                    |
  //       |                |                    |
  //     output           output               output
  // (1) can be merged by MergeElementwiseNodes().
  // (2) can be merged by LinkNodes().
  gpu_model.nodes.resize(3);

  TensorDescriptor in_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor intermediate_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor tmp_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor out_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};

  OperationDef sum_def;
  sum_def.src_tensors.push_back(in_tensor_desc);
  sum_def.dst_tensors.push_back(intermediate_tensor_desc);
  Reduce sum = CreateReduce({Axis::WIDTH}, BHWC(1, 1, 2, 8) /* input shape */,
                            OperationType::REDUCE_SUM, sum_def, gpu_info);

  OperationDef relu_definition;
  relu_definition.src_tensors.push_back(intermediate_tensor_desc);
  relu_definition.dst_tensors.push_back(tmp_tensor_desc);
  ElementwiseDescriptor relu_descriptor;
  GPUOperation relu =
      CreateGpuOperation(relu_definition, std::move(relu_descriptor));

  OperationDef add_definition;
  add_definition.src_tensors.push_back(intermediate_tensor_desc);
  add_definition.src_tensors.push_back(tmp_tensor_desc);
  add_definition.dst_tensors.push_back(out_tensor_desc);
  GPUOperation add =
      CreateElementwiseTwoInput(gpu_info, add_definition, OperationType::ADD,
                                BHWC(1, 1, 1, 8) /* second input shape */,
                                BHWC(1, 1, 1, 8) /* output shape */);
  EXPECT_EQ(add.GetElementwiseInputsCount(), 2);

  cl::Tensor in_tensor;
  ValueId in_tensor_id = 0;
  cl::Tensor intermediate_tensor;
  ValueId intermediate_tensor_id = 1;
  cl::Tensor tmp_tensor;
  ValueId tmp_tensor_id = 2;
  cl::Tensor out_tensor;
  ValueId out_tensor_id = 3;
  sum.SetSrc(&in_tensor, /*index=*/0);
  sum.SetDst(&intermediate_tensor, /*index=*/0);
  relu.SetSrc(&intermediate_tensor, /*index=*/0);
  relu.SetDst(&tmp_tensor, /*index=*/0);
  add.SetSrc(&tmp_tensor, /*index=*/0);
  add.SetSrc(&intermediate_tensor, /*index=*/1);
  add.SetDst(&out_tensor, /*index=*/0);
  gpu_model.nodes[0].gpu_operation =
      std::make_unique<GPUOperation>(std::move(sum));
  gpu_model.nodes[0].inputs.resize(1);
  gpu_model.nodes[0].inputs[0] = in_tensor_id;
  gpu_model.nodes[0].outputs.resize(1);
  gpu_model.nodes[0].outputs[0] = intermediate_tensor_id;
  gpu_model.nodes[0].name = "sum";
  gpu_model.nodes[1].gpu_operation =
      std::make_unique<GPUOperation>(std::move(relu));
  gpu_model.nodes[1].inputs.resize(1);
  gpu_model.nodes[1].inputs[0] = intermediate_tensor_id;
  gpu_model.nodes[1].outputs.resize(1);
  gpu_model.nodes[1].outputs[0] = tmp_tensor_id;
  gpu_model.nodes[1].name = "relu";
  gpu_model.nodes[2].gpu_operation =
      std::make_unique<GPUOperation>(std::move(add));
  gpu_model.nodes[2].inputs.resize(2);
  gpu_model.nodes[2].inputs[0] = intermediate_tensor_id;
  gpu_model.nodes[2].inputs[1] = tmp_tensor_id;
  gpu_model.nodes[2].outputs.resize(1);
  gpu_model.nodes[2].outputs[0] = out_tensor_id;
  gpu_model.nodes[2].name = "add";
}

TEST(MergeNodesTest, MergeElementwiseTwoInputRootAndLeftElementwise) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  GpuModel gpu_model;
  CreateMergeableSumReluAddGpuModel(gpu_info, gpu_model);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "sum -> relu -> add");
}

TEST(MergeNodesTest,
     NotMergeElementwiseTwoInputRootAndLeftElementwiseWithDiffTag) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  GpuModel gpu_model;
  CreateMergeableSumReluAddGpuModel(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 3);

  // The two nodes could not be merged because they have different optional tag.
  gpu_model.nodes[0].optional_tag = {};
  gpu_model.nodes[1].optional_tag.insert(1);
  gpu_model.nodes[2].optional_tag.insert(2);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 3);
  EXPECT_EQ(gpu_model.nodes[0].name, "sum");
  EXPECT_EQ(gpu_model.nodes[1].name, "relu");
  EXPECT_EQ(gpu_model.nodes[2].name, "add");
}

void CreateMergeableSumReluMulAddGpuModel(const GpuInfo& gpu_info,
                                          GpuModel& gpu_model) {
  //      sum
  //     /    \
  //   relu   mul
  //     \    /   -->  sum->relu->mul->add
  //       add              |
  //       |                |
  //     output           output
  //
  // The created GpuModel's nodes can be merged by MergeElementwiseNodes().
  gpu_model.nodes.resize(4);

  TensorDescriptor in_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor intermediate_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor left_tmp_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor right_tmp_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor out_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};

  OperationDef sum_def;
  sum_def.src_tensors.push_back(in_tensor_desc);
  sum_def.dst_tensors.push_back(intermediate_tensor_desc);
  Reduce sum = CreateReduce({Axis::WIDTH}, BHWC(1, 1, 2, 8) /* input shape */,
                            OperationType::REDUCE_SUM, sum_def, gpu_info);

  OperationDef relu_definition;
  relu_definition.src_tensors.push_back(intermediate_tensor_desc);
  relu_definition.dst_tensors.push_back(left_tmp_tensor_desc);
  ElementwiseDescriptor relu_descriptor;
  GPUOperation relu =
      CreateGpuOperation(relu_definition, std::move(relu_descriptor));

  OperationDef mul_definition;
  mul_definition.src_tensors.push_back(intermediate_tensor_desc);
  mul_definition.dst_tensors.push_back(right_tmp_tensor_desc);
  ElementwiseDescriptor mul_descriptor;
  GPUOperation mul =
      CreateGpuOperation(mul_definition, std::move(mul_descriptor));

  OperationDef add_definition;
  add_definition.src_tensors.push_back(left_tmp_tensor_desc);
  add_definition.src_tensors.push_back(right_tmp_tensor_desc);
  add_definition.dst_tensors.push_back(out_tensor_desc);
  GPUOperation add =
      CreateElementwiseTwoInput(gpu_info, add_definition, OperationType::ADD,
                                BHWC(1, 1, 1, 8) /* second input shape */,
                                BHWC(1, 1, 1, 8) /* output shape */);
  EXPECT_EQ(add.GetElementwiseInputsCount(), 2);

  cl::Tensor in_tensor;
  ValueId in_tensor_id = 0;
  cl::Tensor intermediate_tensor;
  ValueId intermediate_tensor_id = 1;
  cl::Tensor left_tmp_tensor;
  ValueId left_tmp_tensor_id = 2;
  cl::Tensor right_tmp_tensor;
  ValueId right_tmp_tensor_id = 3;
  cl::Tensor out_tensor;
  ValueId out_tensor_id = 4;
  sum.SetSrc(&in_tensor, /*index=*/0);
  sum.SetDst(&intermediate_tensor, /*index=*/0);
  relu.SetSrc(&intermediate_tensor, /*index=*/0);
  relu.SetDst(&left_tmp_tensor, /*index=*/0);
  mul.SetSrc(&intermediate_tensor, /*index=*/0);
  mul.SetDst(&right_tmp_tensor, /*index=*/0);
  add.SetSrc(&left_tmp_tensor, /*index=*/0);
  add.SetSrc(&right_tmp_tensor, /*index=*/1);
  add.SetDst(&out_tensor, /*index=*/0);
  gpu_model.nodes[0].gpu_operation =
      std::make_unique<GPUOperation>(std::move(sum));
  gpu_model.nodes[0].inputs.resize(1);
  gpu_model.nodes[0].inputs[0] = in_tensor_id;
  gpu_model.nodes[0].outputs.resize(1);
  gpu_model.nodes[0].outputs[0] = intermediate_tensor_id;
  gpu_model.nodes[0].name = "sum";
  gpu_model.nodes[1].gpu_operation =
      std::make_unique<GPUOperation>(std::move(relu));
  gpu_model.nodes[1].inputs.resize(1);
  gpu_model.nodes[1].inputs[0] = intermediate_tensor_id;
  gpu_model.nodes[1].outputs.resize(1);
  gpu_model.nodes[1].outputs[0] = left_tmp_tensor_id;
  gpu_model.nodes[1].name = "relu";
  gpu_model.nodes[2].gpu_operation =
      std::make_unique<GPUOperation>(std::move(mul));
  gpu_model.nodes[2].inputs.resize(1);
  gpu_model.nodes[2].inputs[0] = intermediate_tensor_id;
  gpu_model.nodes[2].outputs.resize(1);
  gpu_model.nodes[2].outputs[0] = right_tmp_tensor_id;
  gpu_model.nodes[2].name = "mul";
  gpu_model.nodes[3].gpu_operation =
      std::make_unique<GPUOperation>(std::move(add));
  gpu_model.nodes[3].inputs.resize(2);
  gpu_model.nodes[3].inputs[0] = left_tmp_tensor_id;
  gpu_model.nodes[3].inputs[1] = right_tmp_tensor_id;
  gpu_model.nodes[3].outputs.resize(1);
  gpu_model.nodes[3].outputs[0] = out_tensor_id;
  gpu_model.nodes[3].name = "add";
}

TEST(MergeNodesTest, MergeElementwiseTwoInputRootAndParents) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  GpuModel gpu_model;
  CreateMergeableSumReluMulAddGpuModel(gpu_info, gpu_model);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_EQ(gpu_model.nodes[0].name, "sum -> relu -> mul -> add");
}

TEST(MergeNodesTest, NotMergeElementwiseTwoInputRootAndParentsWithDiffTag) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  GpuModel gpu_model;
  CreateMergeableSumReluMulAddGpuModel(gpu_info, gpu_model);
  EXPECT_EQ(gpu_model.nodes.size(), 4);

  // The two nodes could not be merged because they have different optional tag.
  gpu_model.nodes[0].optional_tag = {};
  gpu_model.nodes[1].optional_tag.insert(1);
  gpu_model.nodes[2].optional_tag.insert(2);
  gpu_model.nodes[3].optional_tag.insert(3);

  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 4);
  EXPECT_EQ(gpu_model.nodes[0].name, "sum");
  EXPECT_EQ(gpu_model.nodes[1].name, "relu");
  EXPECT_EQ(gpu_model.nodes[2].name, "mul");
  EXPECT_EQ(gpu_model.nodes[3].name, "add");
}

TEST(MergeNodesTest, NotMergeSubgraphNode) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  CreateGpuModelContainingLinkableNode(gpu_info, gpu_model);
  ASSERT_EQ(gpu_model.nodes.size(), 2);
  // The first node is a subgraph node so should not be merged with the second
  // node, even though it is linkable.
  gpu_model.nodes[0].subgraph_id = "a";

  ASSERT_TRUE(MergeNodes(gpu_info, &gpu_model).ok());
  ASSERT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_EQ(gpu_model.nodes[0].name, "conv");
  EXPECT_EQ(gpu_model.nodes[1].name, "relu");
}

void AddSubgraphNode(GpuModel& model, const std::string& subgraph_id) {
  const GpuModel& subgraph = model.subgraphs[subgraph_id];
  GpuNode node;
  node.subgraph_id = subgraph_id;
  node.gpu_operation = std::make_unique<GPUOperation>();
  node.name = "subgraph";
  node.inputs.resize(subgraph.input_ids_and_refs.size());
  node.outputs.resize(subgraph.output_ids_and_refs.size());
  model.nodes.push_back(std::move(node));
}

void AddSubgraphInputsAndOutputs(GpuModel& model) {
  for (ValueId in : model.nodes.front().inputs) {
    model.input_ids_and_refs.push_back({in, in});
  }
  for (ValueId out : model.nodes.back().outputs) {
    model.output_ids_and_refs.push_back({out, out});
  }
}

TEST(MergeNodesTest, ExpandSubgraphNodes) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;

  CreateGpuModelContainingLinkableNode(gpu_info, gpu_model.subgraphs["a"]);
  AddSubgraphInputsAndOutputs(gpu_model.subgraphs["a"]);
  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model.subgraphs["a"]));
  CreateMergeableSumReluMulAddGpuModel(gpu_info, gpu_model.subgraphs["b"]);
  AddSubgraphInputsAndOutputs(gpu_model.subgraphs["b"]);
  gpu_model.subgraphs["b"].nodes[0].optional_tag.insert(1);
  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model.subgraphs["b"]));

  AddSubgraphNode(gpu_model, "a");
  AddSubgraphNode(gpu_model, "b");
  AddSubgraphNode(gpu_model, "a");

  ASSERT_TRUE(MergeNodes(gpu_info, &gpu_model).ok());
  ASSERT_TRUE(AssembleCode(gpu_info, &gpu_model).ok());
  ASSERT_TRUE(ResolveArgs(&gpu_model).ok());
  ExpandSubgraphs(&gpu_model);
  ASSERT_EQ(gpu_model.nodes.size(), 4);
  EXPECT_EQ(gpu_model.nodes[0].name, "conv -> relu");
  EXPECT_EQ(gpu_model.nodes[0].subgraph_id, "a");
  EXPECT_EQ(gpu_model.nodes[1].name, "sum");
  EXPECT_EQ(gpu_model.nodes[1].optional_tag, absl::flat_hash_set<int>{1});
  EXPECT_EQ(gpu_model.nodes[1].subgraph_id, "b");
  EXPECT_EQ(gpu_model.nodes[2].name, "relu -> mul -> add");
  EXPECT_EQ(gpu_model.nodes[2].subgraph_id, "b");
  EXPECT_EQ(gpu_model.nodes[3].name, "conv -> relu");
  EXPECT_EQ(gpu_model.nodes[3].subgraph_id, "a");

  // Make sure intermediate tensors were created.
  EXPECT_EQ(gpu_model.nodes[1].outputs[0], gpu_model.nodes[2].inputs[0]);
}


TEST(MergeNodesTest, ExpandSubgraphWithConstTensor) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;

  // Before MergeNodes:
  // Main Graph:
  //   in_tensor -> op1 -> out_tensor
  //   Node: op1, subgraph
  //   Tensor: in_tensor (1), sub_in_tensor (2), out_tensor (3),
  //           sub_out_tensor (4)
  // Subgraph:
  //   sub_in_tensor + sub_in_const_tensor -> add_with_const -> sub_out_tensor
  //   Node: add_with_const
  //   Tensor: sub_in_tensor (1), sub_out_tensor (4)
  //   Const Tensor: sub_in_const_tensor (2), sub_internal_const_tensor (3)

  // 1. Create the main graph.
  const ValueId in_tensor_id = 1;
  const ValueId sub_in_tensor_id = 2;
  const ValueId out_tensor_id = 3;
  const ValueId sub_out_tensor_id = 4;

  gpu_model.nodes.resize(2);
  gpu_model.nodes[0].gpu_operation = std::make_unique<GPUOperation>();
  gpu_model.nodes[0].name = "op";
  gpu_model.nodes[0].inputs.push_back(in_tensor_id);
  gpu_model.nodes[0].outputs.push_back(sub_in_tensor_id);
  gpu_model.input_ids_and_refs.push_back({in_tensor_id, in_tensor_id});
  gpu_model.output_ids_and_refs.push_back({out_tensor_id, out_tensor_id});

  gpu_model.nodes[1].name = "subgraph";
  gpu_model.nodes[1].gpu_operation = std::make_unique<GPUOperation>();
  gpu_model.nodes[1].inputs.push_back(sub_in_tensor_id);
  gpu_model.nodes[1].outputs.push_back(out_tensor_id);
  gpu_model.nodes[1].subgraph_id = "sub";

  TensorDescriptor in_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor sub_in_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor out_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};
  TensorDescriptor sub_out_tensor_desc = {
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::BHWC};

  gpu_model.tensors[in_tensor_id] = in_tensor_desc;
  gpu_model.tensors[sub_in_tensor_id] = sub_in_tensor_desc;
  gpu_model.tensors[out_tensor_id] = out_tensor_desc;
  gpu_model.tensors[sub_out_tensor_id] = sub_out_tensor_desc;


  // 2. Create a subgraph "sub"
  auto& subgraph = gpu_model.subgraphs["sub"];
  subgraph.nodes.resize(1);

  const ValueId sub_in_tensor_id_in_subgraph = 1;
  const ValueId sub_const_tensor_id_in_subgraph = 2;
  const ValueId sub_internal_const_tensor_id = 3;
  const ValueId sub_out_tensor_id_in_subgraph = 4;

  subgraph.tensors[sub_in_tensor_id_in_subgraph] = sub_in_tensor_desc;
  subgraph.const_tensors[sub_const_tensor_id_in_subgraph] = {
      DataType::INT4, TensorStorageType::BUFFER, Layout::HWC};
  subgraph.const_tensors[sub_internal_const_tensor_id] = {
      DataType::INT4, TensorStorageType::BUFFER, Layout::HWC};
  subgraph.tensors[sub_out_tensor_id_in_subgraph] = sub_out_tensor_desc;

  // Create a simple elementwise add operation.
  OperationDef add_def;
  add_def.src_tensors.push_back(
      subgraph.tensors[sub_in_tensor_id_in_subgraph]);
  add_def.src_tensors.push_back(
      subgraph.const_tensors[sub_const_tensor_id_in_subgraph]);
  add_def.dst_tensors.push_back(
      subgraph.tensors[sub_out_tensor_id_in_subgraph]);
  ElementwiseDescriptor add_descriptor;
  GPUOperation add_op =
      CreateGpuOperation(add_def, std::move(add_descriptor));

  // Set up the node in the subgraph.
  auto& sub_node = subgraph.nodes[0];
  sub_node.gpu_operation = std::make_unique<GPUOperation>(std::move(add_op));
  sub_node.name = "add_with_const";
  sub_node.inputs = {
    sub_in_tensor_id_in_subgraph, sub_const_tensor_id_in_subgraph};
  sub_node.outputs = {sub_out_tensor_id_in_subgraph};

  // Define subgraph interface.
  subgraph.input_ids_and_refs.push_back(
      {sub_in_tensor_id_in_subgraph, sub_in_tensor_id_in_subgraph});
  subgraph.input_ids_and_refs.push_back(
        {sub_const_tensor_id_in_subgraph, sub_const_tensor_id_in_subgraph});
  subgraph.output_ids_and_refs.push_back(
      {sub_out_tensor_id_in_subgraph, sub_out_tensor_id_in_subgraph});

  // 3. Call MergeNodes. This will expand subgraphs and resolve arguments.
  MLD_ASSERT_OK(MergeNodes(gpu_info, &gpu_model));
  MLD_ASSERT_OK(AssembleCode(gpu_info, &gpu_model));
  MLD_ASSERT_OK(ResolveArgs(&gpu_model));
  ExpandSubgraphs(&gpu_model);

  // After MergeNodes:
  // Main Graph:
  //   in_tensor -> op1 -> out_tensor
  //   Node: op1, add_with_const
  //   Tensor: in_tensor (1), sub_in_tensor (2), out_tensor (3),
  //           sub_out_tensor (4)
  //   Const Tensor: sub_in_const_tensor (5), sub_internal_const_tensor(6)
  // Subgraph:
  //   sub_in_tensor + sub_in_const_tensor -> add_with_const -> sub_out_tensor
  //   Node: add_with_const
  //   Tensor: sub_in_tensor (1), sub_out_tensor (4)
  //   Const Tensor: sub_in_const_tensor (2), sub_internal_const_tensor (3)

  // 4. Assertions.
  // The subgraph node is expanded into the single "add_with_const" node.
  ASSERT_EQ(gpu_model.nodes.size(), 2);
  const auto& expanded_node = gpu_model.nodes[1];
  EXPECT_EQ(expanded_node.name, "add_with_const");
  EXPECT_EQ(gpu_model.const_tensors.size(), 2);
  // Verify the const tensors are copied to the main graph correctly.
  EXPECT_EQ(gpu_model.const_tensors[5], subgraph.const_tensors[2]);
  EXPECT_EQ(gpu_model.const_tensors[6], subgraph.const_tensors[3]);
}

}  // namespace ml_drift
