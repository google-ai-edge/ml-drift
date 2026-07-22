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

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

absl::Status CreateConvReluGpuModel(
    const GpuInfo& gpu_info, GpuModelBuilder& model_builder,
    std::vector<GpuModelBuilder::TensorHandle>& inputs,
    std::vector<GpuModelBuilder::TensorHandle>& outputs,
    int optional_conv_tag = -1, int optional_relu_tag = -1,
    bool conv_subgraph = false) {
  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 16);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());

  auto src_th = model_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
  GpuModelBuilder::OptionalNodeContext optional_context;
  if (optional_conv_tag >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_conv_tag, src_th);
  }
  GpuModelBuilder::TensorHandle conv_out;
  if (conv_subgraph) {
    GpuModelBuilder sub_builder = model_builder.CreateBuilder();
    auto in = sub_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
    auto out = sub_builder.Convolution(in, conv_attr);
    ABSL_RETURN_IF_ERROR(model_builder.RegisterSubgraph(
        std::move(sub_builder), "conv_subgraph", {in}, {out}));

    ABSL_ASSIGN_OR_RETURN(auto dsts,
                          model_builder.Subgraph("conv_subgraph", {src_th}));
    conv_out = dsts[0];
  } else {
    conv_out = model_builder.Convolution(src_th, conv_attr);
  }
  if (optional_conv_tag >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, conv_out,
                                       /*add_copy_to_src=*/false));
  }
  ReLUAttributes relu_attr;
  relu_attr.activation_max = 0.0f;
  relu_attr.activation_min = 0.0f;
  relu_attr.alpha = 0.0f;
  if (optional_relu_tag >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_relu_tag, conv_out);
  }
  auto out = model_builder.ReLU(conv_out, relu_attr);
  if (optional_relu_tag >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, out,
                                       /*add_copy_to_src=*/false));
  }

  //    input         input
  //      |             |
  //    conv            |
  //      |    -->  conv->relu
  //    relu            |
  //      |             |
  //    output        output
  // The created nodes can be merged by LinkNodes().
  inputs.push_back(src_th);
  outputs.push_back(out);
  return absl::OkStatus();
}

absl::Status CreateConvReluGpuModel(const GpuInfo& gpu_info,
                                    GpuModel& gpu_model,
                                    int optional_conv_tag = -1,
                                    int optional_relu_tag = -1,
                                    bool conv_subgraph = false) {
  GpuModelBuilder model_builder(gpu_info, {});

  std::vector<GpuModelBuilder::TensorHandle> inputs;
  std::vector<GpuModelBuilder::TensorHandle> outputs;
  ABSL_RETURN_IF_ERROR(CreateConvReluGpuModel(
      gpu_info, model_builder, inputs, outputs, optional_conv_tag,
      optional_relu_tag, conv_subgraph));

  ABSL_RETURN_IF_ERROR(model_builder.GetGpuModel(
      std::vector<unsigned int>{inputs[0].id},
      std::vector<unsigned int>{outputs[0].id}, &gpu_model));
  return absl::OkStatus();
}

TEST(MergeNodesTest, MergeSingleLinkable) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(CreateConvReluGpuModel(gpu_info, gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "conv"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "relu"));
}

TEST(MergeNodesTest, MergeSingleLinkableWithOptional) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(CreateConvReluGpuModel(gpu_info, gpu_model, /*optional_conv_tag=*/1,
                                   /*optional_relu_tag=*/1));

  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "conv"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "relu"));
  EXPECT_EQ(gpu_model.nodes[0].optional_tag, absl::flat_hash_set<int>{1});
}

TEST(MergeNodesTest, NotMergeSingleLinkableWithDiffTag) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(
      CreateConvReluGpuModel(gpu_info, gpu_model, /*optional_conv_tag=*/1));

  EXPECT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "conv"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[1].name, "relu"));
}

TEST(MergeNodesTest, NotMergeSubgraphNode) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(CreateConvReluGpuModel(gpu_info, gpu_model, /*opional_conv_tag*/ -1,
                                   /*optional_relu_tag*/ -1,
                                   /*conv_subgraph=*/true));

  ASSERT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "conv"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[1].name, "relu"));
}

absl::Status CreateTransposeTransposeGpuModel(
    const GpuInfo& gpu_info, GpuModel& gpu_model,
    int optional_transpose0_tag = -1, int optional_transpose1_tag = -1) {
  GpuModelBuilder model_builder(gpu_info, {});

  auto src_th = model_builder.AddTensor(BHWC(4, 32, 32, 16), DataType::FLOAT32);
  GpuModelBuilder::OptionalNodeContext optional_context;
  if (optional_transpose0_tag >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_transpose0_tag, src_th);
  }
  auto interm = model_builder.Transpose(src_th, BHWC(0, 2, 1, 3));
  if (optional_transpose0_tag >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, interm,
                                       /*add_copy_to_src=*/false));
  }
  if (optional_transpose1_tag >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_transpose1_tag, interm);
  }
  auto out = model_builder.Transpose(interm, BHWC(1, 0, 2, 3));
  if (optional_transpose1_tag >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, out,
                                       /*add_copy_to_src=*/false));
  }

  //    input                   input
  //      |                       |
  //   transpose0                 |
  //      |         --> transpose0->transpose1
  //   transpose1                 |
  //      |                       |
  //    output                  output
  // The created nodes can be merged by MergeReorderNodes().
  ABSL_RETURN_IF_ERROR(
      model_builder.GetGpuModel(std::vector<unsigned int>{src_th.id},
                                std::vector<unsigned int>{out.id}, &gpu_model));
  return absl::OkStatus();
}

TEST(MergeNodesTest, MergeTwoReorder) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(CreateTransposeTransposeGpuModel(gpu_info, gpu_model));

  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "transpose"));
  // If the word appears N times, the split vector will have size N + 1.
  std::vector<std::string> parts =
      absl::StrSplit(gpu_model.nodes[0].name, "transpose");
  EXPECT_GE(parts.size(), 3);
}

TEST(MergeNodesTest, NotMergeTwoReorderWithDiffTag) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(CreateTransposeTransposeGpuModel(gpu_info, gpu_model,
                                             /*optional_transpose0_tag=*/1));

  EXPECT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "transpose"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[1].name, "transpose"));
}

absl::Status CreateAddReluGpuModel(const GpuInfo& gpu_info, GpuModel& gpu_model,
                                   const std::vector<int>& optional_tags = {
                                       -1, -1}) {
  GpuModelBuilder model_builder(gpu_info, {});

  auto src_th = model_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
  GpuModelBuilder::OptionalNodeContext optional_context;
  if (optional_tags[0] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[0], src_th);
  }
  ElementwiseAttributes add_attr;
  add_attr.param = 1.0f;
  auto interm = model_builder.Elementwise(src_th, add_attr, OperationType::ADD);
  if (optional_tags[0] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, interm,
                                       /*add_copy_to_src=*/false));
  }
  ReLUAttributes relu_attr;
  relu_attr.activation_max = 0.0f;
  relu_attr.activation_min = 0.0f;
  relu_attr.alpha = 0.0f;
  if (optional_tags[1] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[1], interm);
  }
  auto out = model_builder.ReLU(interm, relu_attr);
  if (optional_tags[1] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, out,
                                       /*add_copy_to_src=*/false));
  }

  //    input       input
  //      |           |
  //     add          |
  //      |    --> add->relu
  //     relu         |
  //      |           |
  //    output      output
  // The created nodes can be merged by MergeElementwiseNodes().
  ABSL_RETURN_IF_ERROR(
      model_builder.GetGpuModel(std::vector<unsigned int>{src_th.id},
                                std::vector<unsigned int>{out.id}, &gpu_model));
  return absl::OkStatus();
}

TEST(MergeNodesTest, MergeTwoElementwise) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(CreateAddReluGpuModel(gpu_info, gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "add"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "relu"));
}

TEST(MergeNodesTest, NotMergeTwoElementwiseWithDiffTag) {
  GpuModel gpu_model;
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  MLD_ASSERT_OK(
      CreateAddReluGpuModel(gpu_info, gpu_model, /*optional_tags=*/{1, -1}));
  EXPECT_EQ(gpu_model.nodes.size(), 2);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "add"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[1].name, "relu"));
}

absl::Status CreateSumReluAddGpuModel(const GpuInfo& gpu_info,
                                      GpuModel& gpu_model,
                                      const std::vector<int>& optional_tags = {
                                          -1, -1, -1}) {
  GpuModelBuilder model_builder(gpu_info, {});

  auto src_th = model_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
  GpuModelBuilder::OptionalNodeContext optional_context;
  if (optional_tags[0] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[0], src_th);
  }
  auto interm0 =
      model_builder.Reduce(src_th, Reduce::Type::kSum, {Axis::CHANNELS});
  if (optional_tags[0] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, interm0,
                                       /*add_copy_to_src=*/false));
  }
  ReLUAttributes relu_attr;
  relu_attr.activation_max = 0.0f;
  relu_attr.activation_min = 0.0f;
  relu_attr.alpha = 0.0f;
  if (optional_tags[1] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[1], interm0);
  }
  auto interm1 = model_builder.ReLU(interm0, relu_attr);
  if (optional_tags[1] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, interm1,
                                       /*add_copy_to_src=*/false));
  }

  if (optional_tags[2] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[2], interm1);
  }
  auto out = model_builder.Add(interm0, interm1);
  if (optional_tags[2] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, out,
                                       /*add_copy_to_src=*/false));
  }

  //      sum              sum
  //     /    \             |
  //   relu    |            |
  //     \    /   --1->  relu->add  --2->  sum->relu->add
  //       add              |                    |
  //       |                |                    |
  //     output           output               output
  // (1) can be merged by MergeElementwiseNodes().
  // (2) can be merged by LinkNodes().
  ABSL_RETURN_IF_ERROR(
      model_builder.GetGpuModel(std::vector<unsigned int>{src_th.id},
                                std::vector<unsigned int>{out.id}, &gpu_model));
  return absl::OkStatus();
}

TEST(MergeNodesTest, MergeElementwiseTwoInputRootAndLeftElementwise) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  GpuModel gpu_model;
  MLD_ASSERT_OK(CreateSumReluAddGpuModel(gpu_info, gpu_model));

  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "reduce_sum"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "relu"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "add"));
}

TEST(MergeNodesTest,
     NotMergeElementwiseTwoInputRootAndLeftElementwiseWithDiffTag) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  GpuModel gpu_model;
  MLD_ASSERT_OK(CreateSumReluAddGpuModel(gpu_info, gpu_model,
                                     /*optional_tags=*/{-1, 1, 2}));

  EXPECT_EQ(gpu_model.nodes.size(), 3);
  EXPECT_EQ(gpu_model.nodes[0].name, "reduce_sum");
  EXPECT_EQ(gpu_model.nodes[1].name, "relu");
  EXPECT_EQ(gpu_model.nodes[2].name, "add");
}

absl::Status CreateTransposeReluMulAddGpuModel(
    const GpuInfo& gpu_info, GpuModelBuilder& model_builder,
    std::vector<GpuModelBuilder::TensorHandle>& inputs,
    std::vector<GpuModelBuilder::TensorHandle>& outputs,
    const std::vector<int>& optional_tags = {-1, -1, -1, -1},
    bool relu_output = false, bool mul_output = false) {
  auto src_th = model_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
  GpuModelBuilder::OptionalNodeContext optional_context;
  if (optional_tags[0] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[0], src_th);
  }
  auto interm0 = model_builder.Transpose(src_th, BHWC(0, 2, 1, 3));
  if (optional_tags[0] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, interm0,
                                       /*add_copy_to_src=*/false));
  }
  ReLUAttributes relu_attr;
  relu_attr.activation_max = 0.0f;
  relu_attr.activation_min = 0.0f;
  relu_attr.alpha = 0.0f;
  if (optional_tags[1] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[1], interm0);
  }
  auto interm1 = model_builder.ReLU(interm0, relu_attr);
  if (optional_tags[1] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, interm1,
                                       /*add_copy_to_src=*/false));
  }

  if (optional_tags[2] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[2], interm0);
  }
  ElementwiseAttributes mul_attr;
  mul_attr.param = 1.0f;
  auto interm2 =
      model_builder.Elementwise(interm0, mul_attr, OperationType::MUL);
  if (optional_tags[2] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, interm2,
                                       /*add_copy_to_src=*/false));
  }

  if (optional_tags[3] >= 0) {
    optional_context =
        model_builder.BeginOptionalNodes(optional_tags[3], interm1);
  }
  auto out = model_builder.Add(interm1, interm2);
  if (optional_tags[3] >= 0) {
    ABSL_RETURN_IF_ERROR(
        model_builder.EndOptionalNodes(optional_context, out,
                                       /*add_copy_to_src=*/false));
  }

  //    transpose
  //     /    \
  //   relu   mul
  //     \    /   -->  transpose->relu->mul->add
  //       add              |
  //       |                |
  //     output           output
  //
  // The created nodes can be merged by MergeElementwiseNodes().
  inputs.push_back(src_th);
  outputs.push_back(out);
  if (relu_output) {
    outputs.push_back(interm1);
  }
  if (mul_output) {
    outputs.push_back(interm2);
  }
  return absl::OkStatus();
}

absl::Status CreateTransposeReluMulAddGpuModel(
    const GpuInfo& gpu_info, GpuModel& gpu_model,
    const std::vector<int>& optional_tags = {-1, -1, -1, -1},
    bool relu_output = false, bool mul_output = false) {
  GpuModelBuilder model_builder(gpu_info, {});
  std::vector<GpuModelBuilder::TensorHandle> inputs;
  std::vector<GpuModelBuilder::TensorHandle> outputs;

  ABSL_RETURN_IF_ERROR(CreateTransposeReluMulAddGpuModel(
      gpu_info, model_builder, inputs, outputs, optional_tags, relu_output,
      mul_output));

  std::vector<ValueId> output_ids(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    output_ids[i] = outputs[i].id;
  }
  ABSL_RETURN_IF_ERROR(model_builder.GetGpuModel(
      std::vector<unsigned int>{inputs[0].id}, output_ids, &gpu_model));
  return absl::OkStatus();
}

TEST(MergeNodesTest, MergeElementwiseTwoInputRootAndParents) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  GpuModel gpu_model;
  MLD_ASSERT_OK(CreateTransposeReluMulAddGpuModel(gpu_info, gpu_model));

  EXPECT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "transpose"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "relu"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "mul"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "add"));
}

//    transpose
//       |
//      t0
//     /   \
//   relu  mul
//    |     |
//    t1    t2(output)
//     \   /
//      add
//       |
//     out1
TEST(MergeNodesTest, MergeNoMergeRightIntermediateOutput) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  GpuModel gpu_model;
  MLD_ASSERT_OK(CreateTransposeReluMulAddGpuModel(
      gpu_info, gpu_model, /*optional_tags=*/{-1, -1, -1, -1},
      /*relu_output=*/false, /*mul_output=*/true));
  EXPECT_EQ(gpu_model.nodes.size(), 4);
}

//    transpose
//       |
//      t0
//     /   \
//   relu  mul
//    |     |
// t1(out) t2
//     \   /
//      add
//       |
//     out1
TEST(MergeNodesTest, MergeNoMergeLeftIntermediateOutput) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  GpuModel gpu_model;
  MLD_ASSERT_OK(CreateTransposeReluMulAddGpuModel(
      gpu_info, gpu_model, /*optional_tags=*/{-1, -1, -1, -1},
      /*relu_output=*/true, /*mul_output=*/false));
  EXPECT_EQ(gpu_model.nodes.size(), 4);
}

TEST(MergeNodesTest, NotMergeElementwiseTwoInputRootAndParentsWithDiffTag) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  GpuModel gpu_model;
  MLD_ASSERT_OK(CreateTransposeReluMulAddGpuModel(gpu_info, gpu_model,
                                              /*optional_tags=*/{-1, 1, 2, 3}));

  EXPECT_EQ(gpu_model.nodes.size(), 4);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "transpose"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[1].name, "relu"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[2].name, "mul"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[3].name, "add"));
}

TEST(MergeNodesTest, ExpandSubgraphNodes) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  GpuModelBuilder model_builder(gpu_info, {});
  GpuModelBuilder builder_a = model_builder.CreateBuilder();
  std::vector<GpuModelBuilder::TensorHandle> inputs_a;
  std::vector<GpuModelBuilder::TensorHandle> outputs_a;
  MLD_ASSERT_OK(CreateConvReluGpuModel(gpu_info, builder_a, inputs_a, outputs_a));
  MLD_ASSERT_OK(model_builder.RegisterSubgraph(std::move(builder_a), "a_subgraph",
                                           {inputs_a}, {outputs_a}));

  GpuModelBuilder builder_b = model_builder.CreateBuilder();
  std::vector<GpuModelBuilder::TensorHandle> inputs_b;
  std::vector<GpuModelBuilder::TensorHandle> outputs_b;
  MLD_ASSERT_OK(CreateTransposeReluMulAddGpuModel(
      gpu_info, builder_b, inputs_b, outputs_b,
      /*optional_tags=*/{1, -1, -1, -1}));
  MLD_ASSERT_OK(model_builder.RegisterSubgraph(std::move(builder_b), "b_subgraph",
                                           {inputs_b}, {outputs_b}));

  auto src_th = model_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
  MLD_ASSERT_OK_AND_ASSIGN(auto a0_ths,
                       model_builder.Subgraph("a_subgraph", {src_th}));
  MLD_ASSERT_OK_AND_ASSIGN(auto b0_ths,
                       model_builder.Subgraph("b_subgraph", a0_ths));
  MLD_ASSERT_OK_AND_ASSIGN(auto a1_ths,
                       model_builder.Subgraph("a_subgraph", b0_ths));

  GpuModel gpu_model;
  MLD_ASSERT_OK(model_builder.GetGpuModel(std::vector<unsigned int>{src_th.id},
                                      std::vector<unsigned int>{a1_ths[0].id},
                                      &gpu_model));

  ASSERT_EQ(gpu_model.nodes.size(), 4);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "conv"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "relu"));
  EXPECT_EQ(gpu_model.nodes[0].subgraph_id, "a_subgraph");
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[1].name, "transpose"));
  EXPECT_EQ(gpu_model.nodes[1].optional_tag, absl::flat_hash_set<int>{1});
  EXPECT_EQ(gpu_model.nodes[1].subgraph_id, "b_subgraph");
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[2].name, "relu"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[2].name, "mul"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[2].name, "add"));
  EXPECT_EQ(gpu_model.nodes[2].subgraph_id, "b_subgraph");
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[3].name, "conv"));
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[3].name, "relu"));
  EXPECT_EQ(gpu_model.nodes[3].subgraph_id, "a_subgraph");

  // Make sure intermediate tensors were created.
  EXPECT_EQ(gpu_model.nodes[1].outputs[0], gpu_model.nodes[2].inputs[0]);
}


TEST(MergeNodesTest, ExpandSubgraphWithConstTensor) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;

  GpuModelBuilder model_builder(gpu_info, {});

  GpuModelBuilder sub_builder = model_builder.CreateBuilder();
  auto sub_in = sub_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
  TensorDescriptor const_desc = {DataType::FLOAT32, TensorStorageType::BUFFER,
                                 Layout::HWC};
  const_desc.SetBHWCShape(BHWC(1, 32, 32, 16));
  auto const_tensor = sub_builder.AddConstantTensor(std::move(const_desc));
  auto sub_out = sub_builder.Add(sub_in, const_tensor);
  MLD_ASSERT_OK(model_builder.RegisterSubgraph(
      std::move(sub_builder), "add_subgraph", {sub_in}, {sub_out}));

  auto src_th = model_builder.AddTensor(BHWC(1, 32, 32, 16), DataType::FLOAT32);
  MLD_ASSERT_OK_AND_ASSIGN(auto dsts,
                       model_builder.Subgraph("add_subgraph", {src_th}));
  GpuModel gpu_model;
  MLD_ASSERT_OK(model_builder.GetGpuModel(std::vector<unsigned int>{src_th.id},
                                      std::vector<unsigned int>{dsts[0].id},
                                      &gpu_model));
  ASSERT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_TRUE(absl::StrContains(gpu_model.nodes[0].name, "add"));
  EXPECT_EQ(gpu_model.const_tensors.size(), 1);
}

}  // namespace ml_drift
