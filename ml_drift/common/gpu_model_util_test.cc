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

#include "ml_drift/common/gpu_model_util.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace {

using ::testing::ElementsAreArray;

// A GPU for which the local memory fusers are recommended, so that the fuser
// under test actually runs.
GpuInfo GetTestGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.opencl_info.supports_fp16 = true;
  gpu_info.opencl_info.max_allocation_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.buffer_max_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.image2d_max_width = 16384;
  gpu_info.opencl_info.image2d_max_height = 16384;
  gpu_info.opencl_info.image_buffer_max_size = 65536;
  gpu_info.opencl_info.max_work_group_size_x = 1024;
  gpu_info.opencl_info.max_work_group_size_y = 1024;
  gpu_info.opencl_info.max_work_group_size_z = 1024;
  gpu_info.opencl_info.max_work_group_total_size = 1024;
  for (auto type : {DataType::kFloat32, DataType::kFloat16}) {
    gpu_info.opencl_info.supported_images_2d.r_layout.insert(type);
    gpu_info.opencl_info.supported_images_2d.rg_layout.insert(type);
    gpu_info.opencl_info.supported_images_2d.rgb_layout.insert(type);
    gpu_info.opencl_info.supported_images_2d.rgba_layout.insert(type);
  }
  gpu_info.opencl_info.supports_images = true;
  gpu_info.vendor = GpuVendor::kQualcomm;
  gpu_info.adreno_info.generation = AdrenoInfo::Generation::kGen7;
  return gpu_info;
}

// Adds a CONVOLUTION_2D consuming `input`. Its output has no consumers, so it
// becomes the graph output.
void AddConv(GraphFloat32& graph, Value* input) {
  Node* conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::kConvolution2D);
  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(1, 1);
  conv_attr.padding.appended = HW(1, 1);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = OHWI(1, 3, 3, 4);
  // Add 16 elements for XNN_EXTRA_BYTES
  weights.data.resize(1 * 3 * 3 * 4 + 16, 1.0f);
  conv_attr.weights = std::move(weights);
  Tensor<Linear, DataType::kFloat32> bias;
  bias.shape = Linear(1);
  bias.data.resize(1, 0.0f);
  conv_attr.bias = std::move(bias);
  conv_node->operation.attributes = conv_attr;

  graph.AddConsumer(conv_node->id, input->id);
  Value* conv_out = graph.NewValue();
  conv_out->tensor.type = DataType::kFloat32;
  conv_out->tensor.shape = BHWC(1, 4, 4, 1);
  graph.SetProducer(conv_node->id, conv_out->id);
}

TEST(GpuModelUtilTest, ConvertTensorFloat16Constant) {
  GraphFloat32 graph;
  Node* const_node = graph.NewNode();
  const_node->operation.type = ToString(OperationType::kConstant);
  ConstTensorAttributes attr;
  TensorFloat16 tensor;
  tensor.shape = BHWC(1, 1, 1, 4);
  tensor.data = {half(1.0f), half(2.0f), half(3.0f), half(4.0f)};
  attr.tensor = tensor;
  const_node->operation.attributes = attr;

  Value* output = graph.NewValue();
  output->tensor.type = DataType::kFloat16;
  output->tensor.shape = BHWC(1, 1, 1, 4);
  graph.SetProducer(const_node->id, output->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  gpu_info.opencl_info.supports_fp16 = true;
  // Set max_allocation_size to a large enough value to accommodate the tensor.
  gpu_info.opencl_info.max_allocation_size = 256 * 1024 * 1024;
  gpu_info.opencl_info.buffer_max_size = 256 * 1024 * 1024;

  GpuModel gpu_model;
  ABSL_ASSERT_OK(GraphToGpuModel(graph, create_info, gpu_info, &gpu_model));

  // The constant tensor should be part of the GPU model.
  // In GpuModel, constants are often handled by the specific implementation
  // details, but for the purpose of this test, we want to ensure the
  // GraphToGpuModel call succeeded and processed the float16 tensor. The actual
  // data upload happens inside ReserveGraphTensors ->
  // tensor_reserver.AddConstant and then used in GpuModelBuilder.

  // We can verify that the output tensor has the correct type.
  // Since the output value is produced by a constant node, it should be in
  // const_tensors.
  ASSERT_THAT(
      gpu_model.const_tensors,
      ::testing::Contains(::testing::Pair(
          output->id, ::testing::Property(&TensorDescriptor::GetDataType,
                                          DataType::kFloat16))));

  TensorFloat16 downloaded_tensor;
  downloaded_tensor.shape = tensor.shape;
  downloaded_tensor.data.resize(tensor.shape.DimensionsProduct());
  gpu_model.const_tensors.find(output->id)
      ->second.DownloadData(downloaded_tensor.data.data());
  EXPECT_EQ(downloaded_tensor.data, tensor.data);
}

// Regression test for: a RESIZE -> ADD -> CONVOLUTION_2D chain in
// which the ADD has its second operand folded into its attributes, so it only
// has a single runtime input. The fuser used to unconditionally read
// add_inputs[1], running off the end of the vector.
TEST(GpuModelUtilTest, ResizeAddConvLocalMemoryFuserSkipsSingleInputAdd) {
  GraphFloat32 graph;
  Value* input1 = graph.NewValue();
  input1->tensor.type = DataType::kFloat32;
  input1->tensor.shape = BHWC(1, 2, 2, 4);

  Node* resize_node = graph.NewNode();
  resize_node->operation.type = ToString(OperationType::kResize);
  Resize2DAttributes resize_attr;
  resize_attr.new_shape = HW(4, 4);
  resize_attr.type = SamplingType::kBilinear;
  resize_node->operation.attributes = resize_attr;
  graph.AddConsumer(resize_node->id, input1->id);
  Value* resize_out = graph.NewValue();
  resize_out->tensor.type = DataType::kFloat32;
  resize_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(resize_node->id, resize_out->id);

  // The ADD adds a constant scalar, so the resize output is its only input.
  Node* add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::kAdd);
  ElementwiseAttributes add_attr;
  add_attr.param = 1.0f;
  add_node->operation.attributes = add_attr;
  graph.AddConsumer(add_node->id, resize_out->id);
  Value* add_out = graph.NewValue();
  add_out->tensor.type = DataType::kFloat32;
  add_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(add_node->id, add_out->id);

  AddConv(graph, add_out);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;
  // LocalMemory fusers require special kernel hints.
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  // Must not read out of bounds; the fuser simply declines to fuse.
  GpuModel gpu_model;
  ABSL_ASSERT_OK(
      GraphToGpuModel(graph, create_info, GetTestGpuInfo(), &gpu_model));
  for (const auto& node : gpu_model.nodes) {
    EXPECT_THAT(node.name, ::testing::Not(::testing::HasSubstr(
                               "convolution_2d (+ resize input + add)")));
  }
}

// A RESIZE -> ADD -> CONVOLUTION_2D chain in which the ADD output is consumed
// by the CONVOLUTION_2D *and* is a graph output. The fuser collapses all three
// ops into one that only produces the conv output, so the ADD output is dropped
// from the GpuModel even though it is still advertised as a model output.
TEST(GpuModelUtilTest, ResizeAddConvLocalMemoryFuserPreservesGraphOutput) {
  GraphFloat32 graph;
  Value* input1 = graph.NewValue();
  input1->tensor.type = DataType::kFloat32;
  input1->tensor.shape = BHWC(1, 2, 2, 4);
  Value* input2 = graph.NewValue();
  input2->tensor.type = DataType::kFloat32;
  input2->tensor.shape = BHWC(1, 4, 4, 4);

  Node* resize_node = graph.NewNode();
  resize_node->operation.type = ToString(OperationType::kResize);
  Resize2DAttributes resize_attr;
  resize_attr.new_shape = HW(4, 4);
  resize_attr.type = SamplingType::kBilinear;
  resize_node->operation.attributes = resize_attr;
  graph.AddConsumer(resize_node->id, input1->id);
  Value* resize_out = graph.NewValue();
  resize_out->tensor.type = DataType::kFloat32;
  resize_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(resize_node->id, resize_out->id);

  Node* add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::kAdd);
  add_node->operation.attributes = ElementwiseAttributes{};
  graph.AddConsumer(add_node->id, resize_out->id);
  graph.AddConsumer(add_node->id, input2->id);
  Value* add_out = graph.NewValue();
  add_out->tensor.type = DataType::kFloat32;
  add_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(add_node->id, add_out->id);

  AddConv(graph, add_out);

  // The intermediate ADD output is a model output as well as the final CONV
  // output. Unlike conv_out it has a consumer, so it only counts as a graph
  // output once it is registered explicitly.
  graph.AddKnownGraphOutput(add_out);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  GpuModel gpu_model;
  ABSL_ASSERT_OK(
      GraphToGpuModel(graph, create_info, GetTestGpuInfo(), &gpu_model));

  bool advertised_as_output = false;
  for (const auto& [id, ref] : gpu_model.output_ids_and_refs) {
    if (id == add_out->id) advertised_as_output = true;
  }
  ASSERT_TRUE(advertised_as_output)
      << "Precondition: add_out should be advertised as a model output.";

  // A backing tensor must exist, otherwise the delegate dereferences a null
  // GpuSpatialTensor.
  EXPECT_TRUE(gpu_model.tensors.contains(add_out->id))
      << "Graph output " << add_out->id << " has no tensor in the GpuModel.";

  // ...and some node must actually write it, otherwise the caller reads
  // uninitialized GPU memory.
  bool produced_by_some_node = false;
  for (const auto& node : gpu_model.nodes) {
    for (const auto& out_id : node.outputs) {
      if (out_id == add_out->id) produced_by_some_node = true;
    }
  }
  EXPECT_TRUE(produced_by_some_node)
      << "Graph output " << add_out->id
      << " is advertised as a model output but no node produces it; the "
         "fuser collapsed its producer away. nodes="
      << gpu_model.nodes.size();
}

TEST(GpuModelUtilTest, ResizeAddConvLocalMemoryFuserWithQuantizeAndDequantize) {
  GraphFloat32 graph;
  Value* input1 = graph.NewValue();
  input1->tensor.type = DataType::kFloat32;
  input1->tensor.shape = BHWC(1, 2, 2, 4);
  Value* input2 = graph.NewValue();
  input2->tensor.type = DataType::kFloat32;
  input2->tensor.shape = BHWC(1, 4, 4, 4);

  Node* resize_node = graph.NewNode();
  resize_node->operation.type = ToString(OperationType::kResize);
  Resize2DAttributes resize_attr;
  resize_attr.new_shape = HW(4, 4);
  resize_attr.type = SamplingType::kBilinear;
  resize_node->operation.attributes = resize_attr;
  graph.AddConsumer(resize_node->id, input1->id);
  Value* resize_out = graph.NewValue();
  resize_out->tensor.type = DataType::kFloat32;
  resize_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(resize_node->id, resize_out->id);

  QuantizeAndDequantizeAttributes qdq_attr;
  qdq_attr.min = -1.0f;
  qdq_attr.max = 1.0f;
  qdq_attr.scale = 1.0f / 127.5f;

  Node* qdq1_node = graph.NewNode();
  qdq1_node->operation.type = ToString(OperationType::kQuantizeAndDequantize);
  qdq1_node->operation.attributes = qdq_attr;
  graph.AddConsumer(qdq1_node->id, resize_out->id);
  Value* qdq1_out = graph.NewValue();
  qdq1_out->tensor.type = DataType::kFloat32;
  qdq1_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(qdq1_node->id, qdq1_out->id);

  Node* qdq2_node = graph.NewNode();
  qdq2_node->operation.type = ToString(OperationType::kQuantizeAndDequantize);
  qdq2_node->operation.attributes = qdq_attr;
  graph.AddConsumer(qdq2_node->id, qdq1_out->id);
  Value* qdq2_out = graph.NewValue();
  qdq2_out->tensor.type = DataType::kFloat32;
  qdq2_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(qdq2_node->id, qdq2_out->id);

  Node* add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::kAdd);
  add_node->operation.attributes = ElementwiseAttributes{};
  graph.AddConsumer(add_node->id, qdq2_out->id);
  graph.AddConsumer(add_node->id, input2->id);
  Value* add_out = graph.NewValue();
  add_out->tensor.type = DataType::kFloat32;
  add_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(add_node->id, add_out->id);

  Node* qdq3_node = graph.NewNode();
  qdq3_node->operation.type = ToString(OperationType::kQuantizeAndDequantize);
  qdq3_node->operation.attributes = qdq_attr;
  graph.AddConsumer(qdq3_node->id, add_out->id);
  Value* qdq3_out = graph.NewValue();
  qdq3_out->tensor.type = DataType::kFloat32;
  qdq3_out->tensor.shape = BHWC(1, 4, 4, 4);
  graph.SetProducer(qdq3_node->id, qdq3_out->id);

  AddConv(graph, qdq3_out);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kTexture2D;
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  GpuModel gpu_model;
  ABSL_ASSERT_OK(GraphToGpuModel(graph, create_info, GetTestGpuInfo(), &gpu_model));
  ASSERT_EQ(gpu_model.nodes.size(), 1);
  EXPECT_THAT(gpu_model.nodes[0].name,
              ::testing::HasSubstr("convolution_2d (+ resize input + add)"));
}

absl::Status BuildBatchedMatMulModel(const BHWC& left_shape,
                                     const BHWC& right_shape,
                                     bool transpose_right,
                                     GpuModel* gpu_model) {
  GraphFloat32 graph;
  Value* left = graph.NewValue();
  left->tensor.type = DataType::kFloat32;
  left->tensor.shape = left_shape;
  Value* right = graph.NewValue();
  right->tensor.type = DataType::kFloat32;
  right->tensor.shape = right_shape;

  Node* bmm_node = graph.NewNode();
  bmm_node->operation.type = ToString(OperationType::kBatchedMatmul);
  BatchedMatMulAttributes attr;
  attr.transpose_right = transpose_right;
  bmm_node->operation.attributes = attr;
  graph.AddConsumer(bmm_node->id, left->id);
  graph.AddConsumer(bmm_node->id, right->id);

  Value* out = graph.NewValue();
  out->tensor.type = DataType::kFloat32;
  out->tensor.shape =
      BHWC(1, left_shape.h, left_shape.w,
           transpose_right ? right_shape.w : right_shape.c);
  graph.SetProducer(bmm_node->id, out->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;
  return GraphToGpuModel(graph, create_info, GetTestGpuInfo(), gpu_model);
}

TEST(GpuModelUtilTest, BatchedMatMulFullBroadcastDoesNotReplicateConvWeights) {
  GpuModel unbroadcast_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 1, 8, 16), BHWC(1, 1, 16, 32),
                                    /*transpose_right=*/false,
                                    &unbroadcast_model));
  GpuModel full_broadcast_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 4, 8, 16), BHWC(1, 1, 16, 32),
                                    /*transpose_right=*/false,
                                    &full_broadcast_model));

  ASSERT_EQ(unbroadcast_model.nodes.size(), 2);
  ASSERT_EQ(full_broadcast_model.nodes.size(), 2);
  EXPECT_THAT(full_broadcast_model.nodes[0].name,
              ::testing::StartsWith("bhwc_as_hwio"));
  EXPECT_THAT(full_broadcast_model.nodes[1].name,
              ::testing::StartsWith("mat_mul_as_convolution"));

  const ValueId unbroadcast_weights_id = unbroadcast_model.nodes[0].outputs[0];
  const ValueId full_broadcast_weights_id =
      full_broadcast_model.nodes[0].outputs[0];
  EXPECT_EQ(
      full_broadcast_model.tensors.at(full_broadcast_weights_id).GetBHWCShape(),
      unbroadcast_model.tensors.at(unbroadcast_weights_id).GetBHWCShape());
}

TEST(GpuModelUtilTest, BatchedMatMulPartialBroadcastReplicatesConvWeights) {
  GpuModel two_batch_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 2, 8, 16), BHWC(1, 2, 16, 32),
                                    /*transpose_right=*/false,
                                    &two_batch_model));
  GpuModel partial_broadcast_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 4, 8, 16), BHWC(1, 2, 16, 32),
                                    /*transpose_right=*/false,
                                    &partial_broadcast_model));
  GpuModel four_batch_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 4, 8, 16), BHWC(1, 4, 16, 32),
                                    /*transpose_right=*/false,
                                    &four_batch_model));

  ASSERT_EQ(two_batch_model.nodes.size(), 2);
  ASSERT_EQ(partial_broadcast_model.nodes.size(), 2);
  ASSERT_EQ(four_batch_model.nodes.size(), 2);

  const BHWC two_batch_weights_shape =
      two_batch_model.tensors.at(two_batch_model.nodes[0].outputs[0])
          .GetBHWCShape();
  const BHWC partial_broadcast_weights_shape =
      partial_broadcast_model.tensors
          .at(partial_broadcast_model.nodes[0].outputs[0])
          .GetBHWCShape();
  const BHWC four_batch_weights_shape =
      four_batch_model.tensors.at(four_batch_model.nodes[0].outputs[0])
          .GetBHWCShape();

  EXPECT_EQ(partial_broadcast_weights_shape, four_batch_weights_shape);
  EXPECT_EQ(partial_broadcast_weights_shape.DimensionsProduct(),
            2 * two_batch_weights_shape.DimensionsProduct());
}

TEST(GpuModelUtilTest,
     BatchedMatMulTransposeRightUsesFcOnlyForFullBroadcastOrMatchingBatch) {
  // Full broadcast (right_shape.h == 1) can use batched_mat_mul_as_fc.
  GpuModel full_broadcast_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 4, 8, 16), BHWC(1, 1, 32, 16),
                                    /*transpose_right=*/true,
                                    &full_broadcast_model));
  ASSERT_EQ(full_broadcast_model.nodes.size(), 1);
  EXPECT_EQ(full_broadcast_model.nodes[0].name, "batched_mat_mul_as_fc");

  // Matching batch (right_shape.h == dst_shape.h) can use
  // batched_mat_mul_as_fc.
  GpuModel matching_batch_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 4, 8, 16), BHWC(1, 4, 32, 16),
                                    /*transpose_right=*/true,
                                    &matching_batch_model));
  ASSERT_EQ(matching_batch_model.nodes.size(), 1);
  EXPECT_EQ(matching_batch_model.nodes[0].name, "batched_mat_mul_as_fc");

  // Partial broadcast (1 < right_shape.h < dst_shape.h) must fall back to
  // transpose + bhwc_as_hwio + mat_mul_as_convolution.
  GpuModel partial_broadcast_model;
  ABSL_ASSERT_OK(BuildBatchedMatMulModel(BHWC(1, 4, 8, 16), BHWC(1, 2, 32, 16),
                                    /*transpose_right=*/true,
                                    &partial_broadcast_model));
  ASSERT_EQ(partial_broadcast_model.nodes.size(), 3);
  EXPECT_EQ(partial_broadcast_model.nodes[0].name, "transpose");
  EXPECT_THAT(partial_broadcast_model.nodes[1].name,
              ::testing::StartsWith("bhwc_as_hwio"));
  EXPECT_THAT(partial_broadcast_model.nodes[2].name,
              ::testing::StartsWith("mat_mul_as_convolution"));
}

GpuInfo GetTestMetalGpuInfo() {
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription("Apple A17 Pro", GpuApi::kMetal, &gpu_info);
  gpu_info.metal_info.buffer_max_size = 256 * 1024 * 1024;
  gpu_info.metal_info.image2d_max_width = 16384;
  gpu_info.metal_info.image2d_max_height = 16384;
  gpu_info.metal_info.image_array_max_layers = 2048;
  gpu_info.metal_info.image3d_max_width = 2048;
  gpu_info.metal_info.image3d_max_height = 2048;
  gpu_info.metal_info.image3d_max_depth = 2048;
  gpu_info.metal_info.max_work_group_size_x = 1024;
  gpu_info.metal_info.max_work_group_size_y = 1024;
  gpu_info.metal_info.max_work_group_size_z = 1024;
  gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal3_0;
  return gpu_info;
}

absl::Status BuildMetalSplitModel(int num_outputs,
                                  TensorStorageType storage_type,
                                  GpuModel* gpu_model) {
  GraphFloat32 graph;
  Value* input = graph.NewValue();
  input->tensor.type = DataType::kFloat32;
  input->tensor.shape = BHWC(1, 1, 1, num_outputs * 4);

  Node* split_node = graph.NewNode();
  split_node->operation.type = ToString(OperationType::kSplit);
  SplitAttributes attr;
  attr.axis = Axis::kChannels;
  split_node->operation.attributes = attr;
  graph.AddConsumer(split_node->id, input->id);

  for (int i = 0; i < num_outputs; ++i) {
    Value* out = graph.NewValue();
    out->tensor.type = DataType::kFloat32;
    out->tensor.shape = BHWC(1, 1, 1, 4);
    graph.SetProducer(split_node->id, out->id);
  }

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = storage_type;
  return GraphToGpuModel(graph, create_info, GetTestMetalGpuInfo(), gpu_model);
}

TEST(GpuModelUtilTest, MetalSplitChunksWideOutputsWithinResourceLimits) {
  // Metal caps max_total_arguments at min(128, 31) - 4 = 27, so 26 buffer
  // outputs (1 src + 26 dst = 27) fit in a single Split node.
  GpuModel single_chunk_model;
  ABSL_ASSERT_OK(BuildMetalSplitModel(/*num_outputs=*/26, TensorStorageType::kBuffer,
                                 &single_chunk_model));
  ASSERT_EQ(single_chunk_model.nodes.size(), 1);
  EXPECT_EQ(single_chunk_model.nodes[0].inputs.size() +
                single_chunk_model.nodes[0].outputs.size(),
            27);

  // 35 buffer outputs are split across two nodes well below Metal's 31-slot
  // direct [[buffer(0..30)]] table (including uniforms_buffer).
  GpuModel buffer_model;
  ABSL_ASSERT_OK(BuildMetalSplitModel(/*num_outputs=*/35, TensorStorageType::kBuffer,
                                 &buffer_model));
  ASSERT_EQ(buffer_model.nodes.size(), 2);
  EXPECT_EQ(buffer_model.nodes[0].inputs.size() +
                buffer_model.nodes[0].outputs.size(),
            27);
  EXPECT_EQ(buffer_model.nodes[1].inputs.size() +
                buffer_model.nodes[1].outputs.size(),
            11);

  // 35 kTexture2D outputs (which alias write-only outputs to MTLBuffers on
  // Apple2+) also stay well below Metal's 31-slot direct buffer limit.
  GpuModel texture2d_model;
  ABSL_ASSERT_OK(BuildMetalSplitModel(/*num_outputs=*/35,
                                 TensorStorageType::kTexture2D,
                                 &texture2d_model));
  ASSERT_EQ(texture2d_model.nodes.size(), 2);
  EXPECT_EQ(texture2d_model.nodes[0].inputs.size() +
                texture2d_model.nodes[0].outputs.size(),
            28);
  EXPECT_EQ(texture2d_model.nodes[1].inputs.size() +
                texture2d_model.nodes[1].outputs.size(),
            10);
}

// Builds a model with a single FULLY_CONNECTED_INT4 that has 4 inputs and 4
// outputs. Its constant int4 weights are given unpacked, one weight per int8.
absl::Status BuildFullyConnectedInt4Model(const GpuInfo& gpu_info,
                                          GpuModel* gpu_model) {
  GraphFloat32 graph;
  Value* input = graph.NewValue();
  input->tensor.type = DataType::kFloat32;
  input->tensor.shape = BHWC(1, 1, 1, 4);

  Node* fc_node = graph.NewNode();
  fc_node->operation.type = ToString(OperationType::kFullyConnectedInt4);
  FullyConnectedInt4Attributes attr;
  Tensor<OHWI, DataType::kInt8> weights;
  weights.shape = OHWI(4, 1, 1, 4);
  // The weight for output o and input i is 4 * i + o - 8, so that every int4
  // value is used exactly once.
  weights.data = {-8, -4, 0, 4,   // o = 0
                  -7, -3, 1, 5,   // o = 1
                  -6, -2, 2, 6,   // o = 2
                  -5, -1, 3, 7};  // o = 3
  attr.weights = std::move(weights);
  attr.scale.shape = OHWI(1, 1, 1, 1);
  attr.scale.data = {1.0f};
  fc_node->operation.attributes = attr;
  graph.AddConsumer(fc_node->id, input->id);

  Value* out = graph.NewValue();
  out->tensor.type = DataType::kFloat32;
  out->tensor.shape = BHWC(1, 1, 1, 4);
  graph.SetProducer(fc_node->id, out->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;
  return GraphToGpuModel(graph, create_info, gpu_info, gpu_model);
}

// The weights as uploaded to the GPU: biased by 8 into unsigned nibbles and
// stored in I4O4 order, two per byte with the first one in the low nibble.
// The weight for output o and input i thereby becomes nibble 4 * i + o.
constexpr uint8_t kPackedFullyConnectedInt4Weights[] = {0x10, 0x32, 0x54, 0x76,
                                                        0x98, 0xBA, 0xDC, 0xFE};

// The packed weights have to be stored in a buffer.
TEST(GpuModelUtilTest, FullyConnectedInt4PacksInt8WeightsIntoBuffer) {
  GpuInfo gpu_info = GetTestGpuInfo();
  gpu_info.opencl_info.supports_images = false;

  GpuModel gpu_model;
  ABSL_ASSERT_OK(BuildFullyConnectedInt4Model(gpu_info, &gpu_model));

  // The inputs of the node are {src, weights, scale}.
  ASSERT_EQ(gpu_model.nodes.size(), 1);
  ASSERT_EQ(gpu_model.nodes[0].inputs.size(), 3);
  const ValueId weights_id = gpu_model.nodes[0].inputs[1];
  ASSERT_TRUE(gpu_model.const_tensors.contains(weights_id));
  const TensorDescriptor& weights = gpu_model.const_tensors.at(weights_id);
  EXPECT_EQ(weights.GetStorageType(), TensorStorageType::kBuffer);
  EXPECT_THAT(weights.GetData(),
              ElementsAreArray(kPackedFullyConnectedInt4Weights));
}

// The packed weights are stored in a texture which always has the right size,
// so the half of the weights that wasn't packed was left zero-initialized.
TEST(GpuModelUtilTest, FullyConnectedInt4PacksInt8WeightsIntoTexture) {
  GpuModel gpu_model;
  ABSL_ASSERT_OK(BuildFullyConnectedInt4Model(GetTestGpuInfo(), &gpu_model));

  // The inputs of the node are {src, weights, scale}.
  ASSERT_EQ(gpu_model.nodes.size(), 1);
  ASSERT_EQ(gpu_model.nodes[0].inputs.size(), 3);
  const ValueId weights_id = gpu_model.nodes[0].inputs[1];
  ASSERT_TRUE(gpu_model.const_tensors.contains(weights_id));
  const TensorDescriptor& weights = gpu_model.const_tensors.at(weights_id);
  EXPECT_EQ(weights.GetStorageType(), TensorStorageType::kTexture2D);
  EXPECT_THAT(weights.GetData(),
              ElementsAreArray(kPackedFullyConnectedInt4Weights));
}

TEST(GpuModelUtilTest,
     GpuModelBuilderAddTensorExceedingMaxBufferSizeReturnsResourceExhausted) {
  GpuInfo gpu_info = GetTestGpuInfo();
  gpu_info.opencl_info.buffer_max_size = 1 * 1024 * 1024;
  gpu_info.opencl_info.max_allocation_size = 1 * 1024 * 1024;

  {
    GpuModelBuilder builder(gpu_info,
                            {.storage = TensorStorageType::kBuffer});
    auto in = builder.AddTensor(/*b=*/1, /*h=*/1, /*w=*/1024, /*c=*/512,
                                TensorStorageType::kBuffer, DataType::kFloat32);
    auto out = builder.Add(in, 1.0f);
    GpuModel gpu_model;
    EXPECT_THAT(
        builder.GetGpuModel(std::vector<ValueId>{in.id},
                            std::vector<ValueId>{out.id}, &gpu_model),
        ::absl_testing::StatusIs(absl::StatusCode::kResourceExhausted));
  }

  {
    GpuModelBuilder builder(gpu_info,
                            {.storage = TensorStorageType::kBuffer});
    auto in = builder.AddLinearTensor(/*x=*/512 * 1024, DataType::kFloat32);
    GpuModel gpu_model;
    EXPECT_THAT(
        builder.GetGpuModel(std::vector<ValueId>{in.id},
                            std::vector<ValueId>{in.id}, &gpu_model),
        ::absl_testing::StatusIs(absl::StatusCode::kResourceExhausted));
  }
}

TEST(GpuModelUtilTest,
     GpuModelBuilderRegisterSubgraphPropagatesResourceExhausted) {
  GpuInfo gpu_info = GetTestGpuInfo();
  gpu_info.opencl_info.buffer_max_size = 1 * 1024 * 1024;
  gpu_info.opencl_info.max_allocation_size = 1 * 1024 * 1024;

  GpuModelBuilder builder(gpu_info, {.storage = TensorStorageType::kBuffer});
  GpuModelBuilder subgraph_builder = builder.CreateBuilder();
  subgraph_builder.AddTensor(/*b=*/1, /*h=*/1, /*w=*/1024, /*c=*/512,
                             TensorStorageType::kBuffer, DataType::kFloat32);
  EXPECT_THAT(
      builder.RegisterSubgraph(std::move(subgraph_builder), "oversized_sub",
                               /*inputs=*/{}, /*outputs=*/{}),
      ::absl_testing::StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(GpuModelUtilTest,
     GpuModelBuilderConstantExceedingMaxBufferSizeReturnsResourceExhausted) {
  GpuInfo gpu_info = GetTestGpuInfo();
  gpu_info.opencl_info.buffer_max_size = 1 * 1024 * 1024;
  gpu_info.opencl_info.max_allocation_size = 1 * 1024 * 1024;

  TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                        Layout::kLinear);
  desc.SetBHWCShape(BHWC(1, 1, 1, 512 * 1024));  // 2 MiB
  GpuModelBuilder builder(gpu_info, {.storage = TensorStorageType::kBuffer});
  auto weights = builder.AddConstantTensor(std::move(desc));
  GpuModel gpu_model;
  EXPECT_THAT(
      builder.GetGpuModel(std::vector<ValueId>{weights.id},
                          std::vector<ValueId>{weights.id}, &gpu_model),
      ::absl_testing::StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(GpuModelUtilTest,
     GraphToGpuModelExceedingMaxBufferSizeReturnsResourceExhausted) {
  GraphFloat32 graph;
  Value* input = graph.NewValue();
  input->tensor.type = DataType::kFloat32;
  input->tensor.shape = BHWC(1, 1, 1024, 512);  // 2 MiB

  Node* add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::kAdd);
  ElementwiseAttributes add_attr;
  add_attr.param = 1.0f;
  add_node->operation.attributes = add_attr;
  graph.AddConsumer(add_node->id, input->id);
  Value* output = graph.NewValue();
  output->tensor.type = DataType::kFloat32;
  output->tensor.shape = BHWC(1, 1, 1024, 512);
  graph.SetProducer(add_node->id, output->id);

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;

  GpuInfo gpu_info = GetTestGpuInfo();
  gpu_info.opencl_info.buffer_max_size = 1 * 1024 * 1024;
  gpu_info.opencl_info.max_allocation_size = 1 * 1024 * 1024;

  GpuModel gpu_model;
  EXPECT_THAT(
      GraphToGpuModel(graph, create_info, gpu_info, &gpu_model),
      ::absl_testing::StatusIs(absl::StatusCode::kResourceExhausted));
}

}  // namespace
}  // namespace ml_drift
