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

#include "ml_drift/cl/inference_context.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/log/absl_check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/cl/cl_test.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift::cl {
namespace {

class InferenceContextTest : public OpenCLTest {
 public:
  void SetUp() override { OpenCLTest::SetUp(); }

  GpuModelBuilder::TensorHandle InitOptionalModel() {
    CreateGpuModelInfo create_info{
        .precision = CalculationsPrecision::F32,
        .storage_type = TensorStorageType::BUFFER,
    };
    const DataType float_type = DataType::FLOAT32;

    GpuModelBuilder model_builder(env_.GetDevicePtr()->GetInfo(),
                                  create_info.hints, create_info.precision,
                                  create_info.storage_type);

    GpuModelBuilder::TensorHandle in =
        model_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);
    GpuModelBuilder::TensorHandle out =
        model_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);

    // Reset the output to zero.
    out = model_builder.Multiplication(in, 0.0f);

    GpuModelBuilder::OptionalNodeContext context;
    GpuModelBuilder::TensorHandle tmp;

    // Add 42. (optional)
    context = model_builder.BeginOptionalNodes(13, out);
    tmp = model_builder.Add(out, 42);
    ABSL_CHECK_OK(model_builder.EndOptionalNodes(std::move(context), tmp));

    // Multiply 3. (optional)
    context = model_builder.BeginOptionalNodes(19, out);
    tmp = model_builder.Multiplication(out, 3.0f);
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
    ABSL_CHECK_OK(model_builder.GetGpuModel(std::vector<ValueId>{in.id},
                                            std::vector<ValueId>{out.id},
                                            &gpu_model));
    ABSL_CHECK_OK(
        context_.InitFromGpuModel(create_info, &gpu_model, &env_, nullptr));
    return out;
  }

  absl::StatusOr<GpuModelBuilder::TensorHandle> InitSubgraphModel() {
    CreateGpuModelInfo create_info{
        .precision = CalculationsPrecision::F32,
        .storage_type = TensorStorageType::BUFFER,
    };
    const DataType float_type = DataType::FLOAT32;

    GpuModelBuilder model_builder(env_.GetDevicePtr()->GetInfo(),
                                  create_info.hints, create_info.precision,
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
    out = outs[0];

    // Subgraph for multiplying by 5.
    {
      GpuModelBuilder subgraph_builder = model_builder.CreateBuilder();
      GpuModelBuilder::TensorHandle sub_in =
          subgraph_builder.AddTensor(BHWC{1, 1, 1, 1}, float_type);
      auto sub_out = subgraph_builder.Multiplication(sub_in, 5);
      ABSL_RETURN_IF_ERROR(model_builder.RegisterSubgraph(
          std::move(subgraph_builder), "b", {sub_in}, {sub_out}));
    }
    ABSL_ASSIGN_OR_RETURN(outs, model_builder.Subgraph("b", {out}));
    out = outs[0];

    GpuModel gpu_model;
    ABSL_CHECK_OK(model_builder.GetGpuModel(std::vector<ValueId>{in.id},
                                            std::vector<ValueId>{out.id},
                                            &gpu_model));
    ABSL_CHECK_OK(
        context_.InitFromGpuModel(create_info, &gpu_model, &env_, nullptr));
    return out;
  }

  float GetModelResult(GpuModelBuilder::TensorHandle out) {
    ABSL_CHECK_OK(context_.AddToQueue(env_.queue()));
    ABSL_CHECK_OK(env_.queue()->WaitForCompletion());

    Tensor* tensor = context_.GetTensor(out.id);
    ABSL_CHECK(tensor);

    TensorDescriptor desc;
    ABSL_CHECK_OK(tensor->ToDescriptor(&desc, env_.queue()));
    ABSL_CHECK_OK(env_.queue()->WaitForCompletion());

    TensorFloat32 cpu_tensor;
    desc.DownloadData(&cpu_tensor);

    return cpu_tensor.Get(0);
  }

 protected:
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
  // (2 + 2) * 4 = 20
  EXPECT_EQ(GetModelResult(out), 20);
}

}  // namespace
}  // namespace ml_drift::cl
