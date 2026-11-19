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

#include "ml_drift/webgpu/inference_context.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/absl_check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift::webgpu {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;

class InferenceContextTest : public testing::Test {
 public:
  void SetUp() override {
    ABSL_CHECK_OK(env_.Initialize());
    ABSL_CHECK(env_.GetInfo().IsApiWebGpu());
  }

  GpuModelBuilder::TensorHandle InitOptionalModel() {
    CreateGpuModelInfo create_info{
        .precision = CalculationsPrecision::kF32,
        .storage_type = TensorStorageType::kBuffer,
    };
    const DataType float_type = DataType::kFloat32;

    GpuModelBuilder model_builder(env_.GetInfo(), create_info.hints,
                                  create_info.precision,
                                  create_info.storage_type);

    GpuModelBuilder::TensorHandle in =
        model_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);
    GpuModelBuilder::TensorHandle out =
        model_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);

    // Reset the output to zero.
    out = model_builder.Multiplication(in, 0);

    GpuModelBuilder::OptionalNodeContext context;
    GpuModelBuilder::TensorHandle tmp;

    // Add 42. (optional)
    context = model_builder.BeginOptionalNodes(13, out);
    tmp = model_builder.Add(out, 42);
    ABSL_CHECK_OK(model_builder.EndOptionalNodes(std::move(context), tmp));

    // Multiply 3. (optional)
    context = model_builder.BeginOptionalNodes(19, out);
    tmp = model_builder.Multiplication(out, 3);
    ABSL_CHECK_OK(model_builder.EndOptionalNodes(std::move(context), tmp));

    // Add 17. (optional)
    context = model_builder.BeginOptionalNodes(29, out);
    tmp = model_builder.Add(out, 17);
    ABSL_CHECK_OK(model_builder.EndOptionalNodes(std::move(context), tmp));

    // Add 5, (optional on both 30 and 31)
    context = model_builder.BeginOptionalNodes(30, out);
    auto context2 = model_builder.BeginOptionalNodes(31, out);
    tmp = model_builder.Add(out, 5);
    ABSL_CHECK_OK(model_builder.EndOptionalNodes(std::move(context), tmp));
    ABSL_CHECK_OK(model_builder.EndOptionalNodes(std::move(context2), tmp));

    GpuModel gpu_model;
    ABSL_CHECK_OK(model_builder.GetGpuModel(std::vector<uint32_t>{in.id},
                                            std::vector<uint32_t>{out.id},
                                            &gpu_model));
    ABSL_CHECK_OK(context_.InitFromGpuModel(env_, create_info, &gpu_model));
    return out;
  }

  absl::StatusOr<GpuModelBuilder::TensorHandle> InitSubgraphModel() {
    CreateGpuModelInfo create_info{
        .precision = CalculationsPrecision::kF32,
        .storage_type = TensorStorageType::kBuffer,
    };
    const DataType float_type = DataType::kFloat32;

    GpuModelBuilder model_builder(env_.GetInfo(), create_info.hints,
                                  create_info.precision,
                                  create_info.storage_type);

    GpuModelBuilder::TensorHandle in =
        model_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);
    GpuModelBuilder::TensorHandle out =
        model_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);

    // Reset the output to zero.
    out = model_builder.Multiplication(in, 0);
    out = model_builder.Add(out, 2);

    // Subgraph for adding 2.
    {
      GpuModelBuilder subgraph_builder = model_builder.CreateBuilder();
      GpuModelBuilder::TensorHandle sub_in =
          subgraph_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);
      auto sub_out = subgraph_builder.Add(sub_in, 2);
      ABSL_RETURN_IF_ERROR(model_builder.RegisterSubgraph(
          std::move(subgraph_builder), "a", {sub_in}, {sub_out}));
    }
    ABSL_ASSIGN_OR_RETURN(auto outs, model_builder.Subgraph("a", {out}));

    // Subgraph for multiplying by 5.
    {
      GpuModelBuilder subgraph_builder = model_builder.CreateBuilder();
      GpuModelBuilder::TensorHandle sub_in =
          subgraph_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);
      auto sub_out = subgraph_builder.Multiplication(sub_in, 5);
      ABSL_RETURN_IF_ERROR(model_builder.RegisterSubgraph(
          std::move(subgraph_builder), "b", {sub_in}, {sub_out}));
    }
    ABSL_ASSIGN_OR_RETURN(outs, model_builder.Subgraph("b", outs));
    // Use each subgraph again.
    ABSL_ASSIGN_OR_RETURN(outs, model_builder.Subgraph("a", outs));
    ABSL_ASSIGN_OR_RETURN(outs, model_builder.Subgraph("b", outs));
    out = outs[0];

    GpuModel gpu_model;
    ABSL_CHECK_OK(model_builder.GetGpuModel(std::vector<uint32_t>{in.id},
                                            std::vector<uint32_t>{out.id},
                                            &gpu_model));
    ABSL_CHECK_OK(context_.InitFromGpuModel(env_, create_info, &gpu_model));
    return out;
  }

  // Builds `out = in * 3 + 2` into `gpu_model` and returns the input and
  // output handles. The GpuModel is returned to the caller so that the same
  // graph can be used to initialize more than one InferenceContext.
  std::pair<GpuModelBuilder::TensorHandle, GpuModelBuilder::TensorHandle>
  BuildLinearModel(const CreateGpuModelInfo& create_info, GpuModel* gpu_model) {
    GpuModelBuilder model_builder(env_.GetInfo(), create_info.hints,
                                  create_info.precision,
                                  create_info.storage_type);

    GpuModelBuilder::TensorHandle in =
        model_builder.AddTensor(BHWC{1, 1, 1, 1}, DataType::kFloat32);
    GpuModelBuilder::TensorHandle out = model_builder.Multiplication(in, 3);
    out = model_builder.Add(out, 2);

    ABSL_CHECK_OK(model_builder.GetGpuModel(std::vector<uint32_t>{in.id},
                                            std::vector<uint32_t>{out.id},
                                            gpu_model));
    return {in, out};
  }

  float GetModelResult(GpuModelBuilder::TensorHandle out) {
    return GetModelResult(context_, out);
  }

  float GetModelResult(InferenceContext& context,
                       GpuModelBuilder::TensorHandle out) {
    ABSL_CHECK_OK(context.AddToQueue(env_));

    SpatialTensor* tensor = context.GetTensor(out.id);
    ABSL_CHECK(tensor);

    TensorDescriptor desc;
    ABSL_CHECK_OK(tensor->ToDescriptor(env_.device(), &desc));

    TensorFloat32 cpu_tensor;
    desc.DownloadData(&cpu_tensor);

    return cpu_tensor.Get(0);
  }

 protected:
#ifdef __APPLE__
  Environment env_{wgpu::BackendType::Metal};
#else
  Environment env_{wgpu::BackendType::Vulkan};
#endif  // __APPLE__
  InferenceContext context_;
};

TEST_F(InferenceContextTest, OptionalNodes) {
  GpuModelBuilder::TensorHandle out = InitOptionalModel();

  // 0 = 0
  EXPECT_EQ(GetModelResult(out), 0);

  // 0 + 42 = 42
  context_.EnableOptionalNodes({13});
  EXPECT_EQ(GetModelResult(out), 42);

  // 0 * 3 = 0
  context_.EnableOptionalNodes({19});
  EXPECT_EQ(GetModelResult(out), 0);

  // 0 + 17 = 17
  context_.EnableOptionalNodes({29});
  EXPECT_EQ(GetModelResult(out), 17);

  // (0 + 42) * 3 = 126
  context_.EnableOptionalNodes({13, 19});
  EXPECT_EQ(GetModelResult(out), 126);

  // (0 + 42) + 17 = 59
  context_.EnableOptionalNodes({13, 29});
  EXPECT_EQ(GetModelResult(out), 59);

  // (0 * 3) + 17 = 17
  context_.EnableOptionalNodes({19, 29});
  EXPECT_EQ(GetModelResult(out), 17);

  // ((0 + 42) * 3 ) + 17 = 143
  context_.EnableOptionalNodes({13, 19, 29});
  EXPECT_EQ(GetModelResult(out), 143);

  // ((0 + 42) * 3 ) + 17 = 143
  context_.EnableOptionalNodes({13, 19, 29, 30});
  EXPECT_EQ(GetModelResult(out), 143);

  // ((0 + 42) * 3 ) + 17 = 143
  context_.EnableOptionalNodes({13, 19, 29, 31});
  EXPECT_EQ(GetModelResult(out), 143);

  // ((0 + 42) * 3 ) + 17 + 5 = 148
  context_.EnableOptionalNodes({13, 19, 29, 30, 31});
  EXPECT_EQ(GetModelResult(out), 148);
}

TEST_F(InferenceContextTest, Subgraph) {
  auto out_or = InitSubgraphModel();
  ABSL_ASSERT_OK(out_or);
  auto out = std::move(out_or.value());
  // (((2 + 2) * 5) + 2) * 5 = 20
  EXPECT_EQ(GetModelResult(out), 110);
}

TEST_F(InferenceContextTest, F16ModelWithF32PredefinedInput) {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kTexture2D;

  const BHWC input_shape = BHWC(1, 32, 32, 4);
  const BHWC output_shape = BHWC(1, 32, 32, 16);
  Convolution2DAttributes conv_attr;
  auto& attr_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::kFloat32;
  input->tensor.shape = input_shape;

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::kConvolution2D);

  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  attr_weights.shape = OHWI(output_shape.c, 1, 1, input_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.shape.DimensionsProduct(); ++i) {
    attr_weights.data[i] = 1.0f;
  }
  conv_attr.bias.shape = Linear(output_shape.c);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < conv_attr.bias.data.size(); ++i) {
    conv_attr.bias.data[i] = 0.0f;
  }
  conv_node->operation.attributes = conv_attr;
  graph.AddConsumer(conv_node->id, input->id);
  Value* conv_output = nullptr;
  ABSL_CHECK_OK(AddOutput(&graph, conv_node, &conv_output));
  conv_output->tensor.type = DataType::kFloat32;
  conv_output->tensor.shape = output_shape;

  create_info.predefined[input->id] = TensorDescriptor{
      DataType::kFloat32,
      TensorStorageType::kTexture2D,
      Layout::kHWC,
  };
  GpuModel gpu_model;
  ABSL_CHECK_OK(
      GraphToGpuModel(graph, create_info, env_.GetInfo(), &gpu_model));
  ABSL_CHECK_OK(context_.InitFromGpuModel(env_, create_info, &gpu_model));

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 32, 32, 4);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct(), 1.0f);
  ABSL_CHECK_OK(context_.SetInputTensor(env_, input->id, src_tensor));
  ABSL_CHECK_OK(context_.AddToQueue(env_));
  TensorFloat32 dst_tensor;
  ABSL_CHECK_OK(context_.GetOutputTensor(env_, conv_output->id, &dst_tensor));

  EXPECT_EQ(dst_tensor.data[0], 4.0f);
}

// Serializing a model and restoring it must produce a context that computes
// exactly the same results. `RestoreDeserialized` installs the WGSL stored in
// the flatbuffer but rebuilds the host-side bindings from the current
// operation code, so this also guards against the two drifting apart.
TEST_F(InferenceContextTest, RestoreDeserializedMatchesOriginalContext) {
  CreateGpuModelInfo create_info = {
      .precision = CalculationsPrecision::kF32,
      .storage_type = TensorStorageType::kBuffer,
  };
  GpuModel gpu_model;
  auto [in, out] = BuildLinearModel(create_info, &gpu_model);

  std::vector<uint8_t> serialized_model;
  ABSL_ASSERT_OK(context_.InitFromGpuModel(env_, create_info, &gpu_model,
                                      &serialized_model));
  ASSERT_THAT(serialized_model, Not(IsEmpty()));

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 1);
  src_tensor.data = {5.0f};

  // 5 * 3 + 2 = 17
  ABSL_ASSERT_OK(context_.SetInputTensor(env_, in.id, src_tensor));
  EXPECT_EQ(GetModelResult(context_, out), 17.0f);

  InferenceContext restored_context;
  ABSL_ASSERT_OK(restored_context.RestoreDeserialized(serialized_model, env_,
                                                 &create_info));
  ABSL_ASSERT_OK(restored_context.SetInputTensor(env_, in.id, src_tensor));
  EXPECT_EQ(GetModelResult(restored_context, out), 17.0f);
}

}  // namespace
}  // namespace ml_drift::webgpu
