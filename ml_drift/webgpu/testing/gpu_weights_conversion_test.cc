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

#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/gpu_weights_conversion_test_util.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/execution_environment.h"
#include "ml_drift/webgpu/inference_context.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/testing/webgpu_test.h"
#include "ml_drift/webgpu/webgpu_api_util.h"

namespace ml_drift {

using ::testing::AssertionFailure;
using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

absl::StatusOr<TensorFloat32> GetOutput(
    const webgpu::ExecutionEnvironment& exec_env,
    webgpu::InferenceContext& context,
    ml_drift::Tensor<BHWC, DataType::kFloat32>& input_tensor) {
  ABSL_CHECK_EQ(context.GetInputIds().size(), 1);
  ABSL_CHECK_EQ(context.GetOutputIds().size(), 1);
  ValueId input_id = context.GetInputIds()[0];
  ValueId output_id = context.GetOutputIds()[0];
  ABSL_RETURN_IF_ERROR(
      context.SetInputTensor(exec_env, input_id, input_tensor));
  ABSL_RETURN_IF_ERROR(context.AddToQueue(exec_env));
  ABSL_RETURN_IF_ERROR(webgpu::WaitUntilCompleted(
      exec_env.queue(), exec_env.device(), absl::Seconds(10)));
  TensorFloat32 output;
  ABSL_RETURN_IF_ERROR(context.GetOutputTensor(exec_env, output_id, &output));
  return output;
}

absl::Status InitContexts(const webgpu::ExecutionEnvironment* env,
                          const GraphFloat32& graph,
                          const CreateGpuModelInfo& create_info,
                          webgpu::InferenceContext* main_context,
                          webgpu::InferenceContext* conversion_context) {
  CreateGpuModelInfo create_info_main = create_info;

  GpuModel gpu_model_with_external_weights;
  GpuModel gpu_weights_conversion_model;
  absl::flat_hash_map<ValueId, ValueId> weights_mapping;
  std::vector<WeightsManager::UploadWeightsInfo> upload_weights_info;

  ABSL_RETURN_IF_ERROR(GraphToGpuModelWithWeightsConversion(
      graph, create_info, env->GetInfo(), &gpu_model_with_external_weights,
      &gpu_weights_conversion_model, &weights_mapping, &upload_weights_info));
  ABSL_RETURN_IF_ERROR(conversion_context->InitFromGpuModel(
      *env, create_info, &gpu_weights_conversion_model));

  for (const auto& upload_info : upload_weights_info) {
    auto tensor = conversion_context->GetTensor(upload_info.input_id);
    ABSL_RETURN_IF_ERROR(tensor->WriteData(env->queue(), upload_info.data));
  }

  for (const auto& [converted_weight_id, main_model_weight_id] :
       weights_mapping) {
    create_info_main.external_immutable_tensors.insert(
        {main_model_weight_id,
         conversion_context->GetTensor(converted_weight_id)});
  }

  ABSL_RETURN_IF_ERROR(main_context->InitFromGpuModel(
      *env, create_info_main, &gpu_model_with_external_weights));
  return absl::OkStatus();
}

absl::Status ConvFloatTest(const webgpu::ExecutionEnvironment& exec_env,
                           CalculationsPrecision precision,
                           const BHWC input_shape, const BHWC output_shape,
                           const int kernel_size) {
  // cl::Environment* env = exec_env;
  CreateGpuModelInfo create_info;
  create_info.storage_type = TensorStorageType::kBuffer;
  create_info.precision = precision;

  ABSL_ASSIGN_OR_RETURN(
      GraphFloat32 graph,
      CreateConvGraph(input_shape, output_shape, kernel_size));

  // Initialize input tensor.
  ml_drift::Tensor<BHWC, DataType::kFloat32> input_tensor;
  input_tensor.shape = input_shape;
  input_tensor.data.resize(input_shape.DimensionsProduct());
  InitWithSinValues(input_tensor.data);

  GpuModel reference_gpu_model;
  ABSL_EXPECT_OK(GraphToGpuModel(graph, create_info, exec_env.GetInfo(),
                            &reference_gpu_model));
  webgpu::InferenceContext reference_context;
  ABSL_EXPECT_OK(reference_context.InitFromGpuModel(exec_env, create_info,
                                               &reference_gpu_model));
  ABSL_ASSIGN_OR_RETURN(auto reference,
                        GetOutput(exec_env, reference_context, input_tensor));

  webgpu::InferenceContext conversion_context;
  webgpu::InferenceContext context_with_external_weights;
  ABSL_EXPECT_OK(InitContexts(&exec_env, graph, create_info,
                         &context_with_external_weights, &conversion_context));
  ABSL_EXPECT_OK(conversion_context.AddToQueue(exec_env));
  ABSL_ASSIGN_OR_RETURN(
      auto actual,
      GetOutput(exec_env, context_with_external_weights, input_tensor));
  EXPECT_THAT(reference.data, Pointwise(FloatNear(2e-5), actual.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8Test(
    const webgpu::ExecutionEnvironment& exec_env, const BHWC input_shape,
    const BHWC output_shape) {
  CreateGpuModelInfo create_info;
  create_info.storage_type = TensorStorageType::kBuffer;
  create_info.precision = CalculationsPrecision::kF16;

  // Initialize input tensor.
  ml_drift::Tensor<BHWC, DataType::kFloat32> input_tensor;
  input_tensor.shape = input_shape;
  input_tensor.data.resize(input_shape.DimensionsProduct());
  InitWithSinValues(input_tensor.data);

  FCInt8TestGraph fc_int8_test_graph(input_shape, output_shape);
  ABSL_ASSIGN_OR_RETURN(GraphFloat32 graph_int8,
                        fc_int8_test_graph.CreateFCInt8Graph());

  // Compute the results of Int8 FullyConnected without WeightsManager, as the
  // reference.
  GpuModel reference_gpu_model;
  ABSL_EXPECT_OK(GraphToGpuModel(graph_int8, create_info, exec_env.GetInfo(),
                            &reference_gpu_model));
  webgpu::InferenceContext target_context;
  ABSL_EXPECT_OK(target_context.InitFromGpuModel(exec_env, create_info,
                                            &reference_gpu_model));
  ABSL_ASSIGN_OR_RETURN(auto reference,
                        GetOutput(exec_env, target_context, input_tensor));

  webgpu::InferenceContext conversion_context;
  webgpu::InferenceContext context_with_external_weights;
  ABSL_EXPECT_OK(InitContexts(&exec_env, graph_int8, create_info,
                         &context_with_external_weights, &conversion_context));
  ABSL_EXPECT_OK(conversion_context.AddToQueue(exec_env));
  ABSL_ASSIGN_OR_RETURN(
      auto actual,
      GetOutput(exec_env, context_with_external_weights, input_tensor));
  EXPECT_THAT(reference.data, Pointwise(FloatNear(2e-3), actual.data))
      << "input_shape: " << input_shape << ", output_shape: " << output_shape;
  return absl::OkStatus();
}

absl::Status FullyConnectedFloat32VSInt8Test(
    const webgpu::ExecutionEnvironment& exec_env, const BHWC input_shape,
    const BHWC output_shape) {
  CreateGpuModelInfo create_info;
  create_info.storage_type = TensorStorageType::kBuffer;
  create_info.precision = CalculationsPrecision::kF16;

  FCInt8TestGraph fc_int8_test_graph(input_shape, output_shape);

  // Initialize input tensor.
  ml_drift::Tensor<BHWC, DataType::kFloat32> input_tensor;
  input_tensor.shape = input_shape;
  input_tensor.data.resize(input_shape.DimensionsProduct());
  InitWithSinValues(input_tensor.data);

  // Compute the results of F32 FullyConnected, as the reference.
  ABSL_ASSIGN_OR_RETURN(GraphFloat32 graph_f32,
                        fc_int8_test_graph.CreateFCFloat32Graph());
  GpuModel reference_gpu_model;
  ABSL_EXPECT_OK(GraphToGpuModel(graph_f32, create_info, exec_env.GetInfo(),
                            &reference_gpu_model));
  webgpu::InferenceContext reference_context;
  ABSL_EXPECT_OK(reference_context.InitFromGpuModel(exec_env, create_info,
                                               &reference_gpu_model));
  ABSL_ASSIGN_OR_RETURN(auto reference_output,
                        GetOutput(exec_env, reference_context, input_tensor));

  // Compute the results of Int8 FullyConnected, as the target.
  ABSL_ASSIGN_OR_RETURN(GraphFloat32 graph_int8,
                        fc_int8_test_graph.CreateFCInt8Graph());
  GpuModel target_gpu_model;
  ABSL_EXPECT_OK(GraphToGpuModel(graph_int8, create_info, exec_env.GetInfo(),
                            &target_gpu_model));
  webgpu::InferenceContext target_context;
  ABSL_EXPECT_OK(target_context.InitFromGpuModel(exec_env, create_info,
                                            &target_gpu_model));
  ABSL_ASSIGN_OR_RETURN(auto actual,
                        GetOutput(exec_env, target_context, input_tensor));

  EXPECT_THAT(reference_output.data, Pointwise(FloatNear(2e-3), actual.data))
      << "input_shape: " << input_shape << ", output_shape: " << output_shape;
  return absl::OkStatus();
}

template <DataType QuantizedT>
absl::Status QuantizedConvTest(const webgpu::ExecutionEnvironment& exec_env,
                               CalculationsPrecision precision,
                               const BHWC input_shape, const BHWC output_shape,
                               const int kernel_size) {
  CreateGpuModelInfo create_info;
  create_info.storage_type = TensorStorageType::kBuffer;
  create_info.precision = precision;

  ABSL_ASSIGN_OR_RETURN(GraphFloat32 graph,
                        CreateQuantizedConvGraph<QuantizedT>(
                            input_shape, output_shape, kernel_size));

  // Initialize input tensor.
  ml_drift::Tensor<BHWC, DataType::kFloat32> input_tensor;
  input_tensor.shape = input_shape;
  input_tensor.data.resize(input_shape.DimensionsProduct());
  InitWithSinValues(input_tensor.data);

  GpuModel reference_gpu_model;
  ABSL_EXPECT_OK(GraphToGpuModel(graph, create_info, exec_env.GetInfo(),
                            &reference_gpu_model));
  webgpu::InferenceContext reference_context;
  ABSL_EXPECT_OK(reference_context.InitFromGpuModel(exec_env, create_info,
                                               &reference_gpu_model));
  ABSL_ASSIGN_OR_RETURN(auto reference,
                        GetOutput(exec_env, reference_context, input_tensor));

  webgpu::InferenceContext conversion_context;
  webgpu::InferenceContext context_with_external_weights;
  ABSL_EXPECT_OK(InitContexts(&exec_env, graph, create_info,
                         &context_with_external_weights, &conversion_context));
  ABSL_EXPECT_OK(conversion_context.AddToQueue(exec_env));
  ABSL_ASSIGN_OR_RETURN(
      auto actual,
      GetOutput(exec_env, context_with_external_weights, input_tensor));
  EXPECT_THAT(reference.data, Pointwise(FloatNear(2e-5), actual.data));
  return absl::OkStatus();
}

class WeightsManagerConvTest
    : public webgpu::WebGpuOperationTest,
      public WithParamInterface<
          std::tuple<CalculationsPrecision, BHWC, BHWC, int>> {};

TEST_P(WeightsManagerConvTest, Conv1x1Float32) {
  auto [precision, input_shape, output_shape, kernel_size] = GetParam();
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const TensorStorageType storage_type = TensorStorageType::kBuffer;
  if (!exec_env_.IsStorageSupported(storage_type, data_type)) {
    GTEST_SKIP() << "Unsupported storage " << ToString(storage_type)
                 << " with data type " << ToString(data_type);
  }
  ABSL_EXPECT_OK(ConvFloatTest(exec_env_.GetEnv(), precision, input_shape,
                          output_shape, kernel_size));
}

TEST_P(WeightsManagerConvTest, Conv3x3Int4) {
  auto [precision, input_shape, output_shape, kernel_size] = GetParam();
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const TensorStorageType storage_type = TensorStorageType::kBuffer;
  if (!exec_env_.IsStorageSupported(storage_type, data_type)) {
    GTEST_SKIP() << "Unsupported storage " << ToString(storage_type)
                 << " with data type " << ToString(data_type);
  }
  ABSL_EXPECT_OK(QuantizedConvTest<DataType::kInt4>(
      exec_env_.GetEnv(), precision, input_shape, output_shape, kernel_size));
}

TEST_P(WeightsManagerConvTest, Conv3x3Int8) {
  auto [precision, input_shape, output_shape, kernel_size] = GetParam();
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const TensorStorageType storage_type = TensorStorageType::kBuffer;
  if (!exec_env_.IsStorageSupported(storage_type, data_type)) {
    GTEST_SKIP() << "Unsupported storage " << ToString(storage_type)
                 << " with data type " << ToString(data_type);
  }
  ABSL_EXPECT_OK(QuantizedConvTest<DataType::kInt8>(
      exec_env_.GetEnv(), precision, input_shape, output_shape, kernel_size));
}

INSTANTIATE_TEST_SUITE_P(
    WeightsManagerConvTestSuite, WeightsManagerConvTest,
    Combine(ValuesIn({CalculationsPrecision::kF32,
                      CalculationsPrecision::kF16}),
            ValuesIn({BHWC(1, 128, 72, 8)}), ValuesIn({BHWC(1, 128, 72, 224)}),
            ValuesIn({3})),
    [](const TestParamInfo<WeightsManagerConvTest::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_input_shape_",
                       GetShapeName(std::get<1>(info.param)), "_output_shape_",
                       GetShapeName(std::get<2>(info.param)), "_kernel_size_",
                       std::get<3>(info.param)),
          {{":", ""}});
    });

class WeightsManagerFCTest : public webgpu::WebGpuOperationTest,
                             public WithParamInterface<std::tuple<BHWC, int>> {
};

// TODO: b/410044840 - FullyConnectedInt8 test fails on Pixel 9 (Mali
// GPU chip) because of the failures of FullyConnectedFloat32VSInt8 test.
TEST_P(WeightsManagerFCTest, FullyConnectedInt8) {
  auto [input_shape, output_channels] = GetParam();
  BHWC output_shape = input_shape;
  output_shape.c = output_channels;
  ABSL_EXPECT_OK(
      FullyConnectedInt8Test(exec_env_.GetEnv(), input_shape, output_shape));
}
// TODO: b/410044840 - FullyConnectedFloat32VSInt8 test fails on Pixel 9 (Mali
// GPU chip), through it's passed with 'requires-gpu-nvidia' tag.
// The following test is not in the scope of this test file, as it's comparing
// results of F32-FullyConnected and Int8-FullyConnected. It's located here,
// because the above FullyConnectedInt8 test fails because of this issue.
TEST_P(WeightsManagerFCTest, FullyConnectedFloat32VSInt8) {
  auto [input_shape, output_channels] = GetParam();
  BHWC output_shape = input_shape;
  output_shape.c = output_channels;
  ABSL_EXPECT_OK(FullyConnectedFloat32VSInt8Test(exec_env_.GetEnv(), input_shape,
                                            output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    WeightsManagerFCTestSuite, WeightsManagerFCTest,
    Combine(ValuesIn({BHWC(1, 1, 4, 8), BHWC(1, 1, 32, 8), BHWC(1, 1, 32, 17)}),
            ValuesIn({8, 17})),
    [](const TestParamInfo<WeightsManagerFCTest::ParamType>& info) {
      return absl::StrCat("input_shape_", GetShapeName(std::get<0>(info.param)),
                          "_output_channels_", std::get<1>(info.param));
    });

class WeightsManagerWeightsSumITest
    : public webgpu::WebGpuOperationTest,
      public WithParamInterface<std::tuple<OHWI, DataType>> {};

TEST_P(WeightsManagerWeightsSumITest, WeightsSumI) {
  auto& [weights_shape, input_data_type] = GetParam();
  if (!exec_env_.IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kInt8)) {
    GTEST_SKIP() << "Buffer storage with Int8 data type is not supported.";
  }

  CreateGpuModelInfo create_info;
  create_info.storage_type = TensorStorageType::kBuffer;
  create_info.precision = CalculationsPrecision::kF32;

  // Initialize weight data and prepare reference output.
  std::vector<int8_t> weights_data;
  std::vector<int32_t> reference_output;
  InitializeWeightsSumIData(weights_data, reference_output, weights_shape,
                            input_data_type);

  // Use WeightsManager to get weights_sum_i via Gpu computation.
  WeightsManager weights_manager;
  absl::flat_hash_map<ValueId, ValueId> weights_mapping;
  std::vector<WeightsManager::UploadWeightsInfo> upload_weights_info;
  GpuModel conversion_gpu_model;
  ValueId fake_main_model_weight_id = 0;
  std::vector<ValueId> main_model_weights_ids = {fake_main_model_weight_id};
  weights_manager.RegisterWeightsSumIConversion(main_model_weights_ids,
                                                weights_shape, input_data_type,
                                                weights_data.data());
  ABSL_ASSERT_OK(weights_manager.CreateConversionGpuModel(
      exec_env_.GetEnv().GetInfo(), &conversion_gpu_model, &weights_mapping,
      &upload_weights_info));
  ValueId converted_output_id = 0;
  for (auto [converted_id, main_id] : weights_mapping) {
    if (main_id == fake_main_model_weight_id) {
      converted_output_id = converted_id;
      break;
    }
  }
  webgpu::InferenceContext conversion_context;
  ABSL_ASSERT_OK(conversion_context.InitFromGpuModel(exec_env_.GetEnv(), create_info,
                                                &conversion_gpu_model));
  for (const auto& upload_info : upload_weights_info) {
    auto tensor = conversion_context.GetTensor(upload_info.input_id);
    ABSL_ASSERT_OK(tensor->WriteData(exec_env_.GetEnv().queue(), upload_info.data));
  }
  ABSL_EXPECT_OK(conversion_context.AddToQueue(exec_env_.GetEnv()));
  ABSL_ASSERT_OK(webgpu::WaitUntilCompleted(exec_env_.GetEnv().queue(),
                                       exec_env_.GetEnv().device(),
                                       absl::Seconds(10)));
  TensorInt32 output;
  ABSL_ASSERT_OK(conversion_context.GetOutputTensor(exec_env_.GetEnv(),
                                               converted_output_id, &output));

  EXPECT_THAT(output.data, Pointwise(testing::Eq(), reference_output))
      << "weights_shape: " << weights_shape
      << ", input_data_type: " << ToString(input_data_type);
}

INSTANTIATE_TEST_SUITE_P(
    WeightsManagerWeightsSumITestSuite, WeightsManagerWeightsSumITest,
    Combine(ValuesIn({OHWI(16, 1, 1, 16)}),
            ValuesIn({DataType::kInt4, DataType::kInt8})),
    [](const TestParamInfo<WeightsManagerWeightsSumITest::ParamType>& info) {
      return absl::StrCat(
          "weights_shape_", GetShapeName(std::get<0>(info.param)),
          "_input_data_type_", ToString(std::get<1>(info.param)));
    });

class WeightsManagerGpuMemoryUsageTest
    : public webgpu::WebGpuOperationTest,
      public WithParamInterface<
          std::tuple<OHWI, WeightsDescription, DataType>> {};

TEST_P(WeightsManagerGpuMemoryUsageTest, Int4WeightsConversion) {
  auto [weights_shape, weights_desc, input_data_type] = GetParam();
  if (input_data_type == DataType::kInt4) {
    weights_desc.type = DataType::kUint4;
  } else if (input_data_type == DataType::kInt8) {
    weights_desc.type = DataType::kUint8;
  } else {
    AssertionFailure() << absl::StrCat("Unsupported input data type: ",
                                       ToString(input_data_type));
  }
  if (!exec_env_.IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kInt8)) {
    GTEST_SKIP() << "Buffer storage with Int8 data type is not supported.";
  }

  size_t raw_weights_size, expected_input_gpu_memory_size,
      expected_output_gpu_memory_size;
  const size_t output_elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights_shape);
  if (input_data_type == DataType::kInt4) {
    raw_weights_size = DivideRoundUp(weights_shape.DimensionsProduct(), 2);
    const size_t input_elements_count =
        DivideRoundUp(weights_shape.DimensionsProduct(), 2);
    expected_input_gpu_memory_size = DivideRoundUp(input_elements_count, 4) * 4;
    expected_output_gpu_memory_size = DivideRoundUp(output_elements_count, 2);
  } else {
    raw_weights_size =
        weights_shape.DimensionsProduct() * SizeOf(input_data_type);
    expected_input_gpu_memory_size =
        DivideRoundUp(weights_shape.DimensionsProduct(), 4) * 4 *
        SizeOf(input_data_type);
    expected_output_gpu_memory_size =
        GetTotalElementsCountForLayout(weights_desc, weights_shape) *
        SizeOf(input_data_type);
  }

  // Initialize weight data and prepare reference output.
  std::vector<int8_t> raw_weights_data(raw_weights_size);

  // Use WeightsManager to get weights_sum_i via Gpu computation.
  WeightsManager weights_manager;
  absl::flat_hash_map<ValueId, ValueId> weights_mapping;
  std::vector<WeightsManager::UploadWeightsInfo> upload_weights_info;
  GpuModel conversion_gpu_model;
  ValueId fake_main_model_weight_id = 0;
  std::vector<ValueId> main_model_weights_ids = {fake_main_model_weight_id};
  weights_manager.RegisterWeightsConversion(
      main_model_weights_ids, weights_desc, weights_shape, input_data_type,
      raw_weights_data.data());
  ABSL_ASSERT_OK(weights_manager.CreateConversionGpuModel(
      exec_env_.GetEnv().GetInfo(), &conversion_gpu_model, &weights_mapping,
      &upload_weights_info));
  ASSERT_EQ(conversion_gpu_model.input_ids_and_refs.size(), 1);
  ASSERT_EQ(conversion_gpu_model.output_ids_and_refs.size(), 1);

  const size_t input_gpu_memory_size =
      conversion_gpu_model
          .tensors[conversion_gpu_model.input_ids_and_refs[0].first]
          .GetMemorySizeInBytes();
  EXPECT_EQ(input_gpu_memory_size, expected_input_gpu_memory_size);

  const size_t output_gpu_memory_size =
      conversion_gpu_model
          .tensors[conversion_gpu_model.output_ids_and_refs[0].first]
          .GetMemorySizeInBytes();
  EXPECT_EQ(output_gpu_memory_size, expected_output_gpu_memory_size);
}

// Dst Weight Layouts to test
std::array<WeightsDescription, 3> DstWeightsDescsToTest() {
  return {
      {{.layout = WeightsLayout::kOSpatialIOGroupI4O4, .output_group_size = 16},
       {.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4,
        .output_group_size = 16},
       {.layout = WeightsLayout::kOSpatialIOGroupO4I4,
        .output_group_size = 16}}};
}

INSTANTIATE_TEST_SUITE_P(
    WeightsManagerGpuMemoryUsageTestSuite, WeightsManagerGpuMemoryUsageTest,
    Combine(ValuesIn({OHWI(16, 1, 1, 16), OHWI(3, 1, 1, 8)}),
            ValuesIn(DstWeightsDescsToTest()),
            ValuesIn({DataType::kInt4, DataType::kInt8})),
    [](const TestParamInfo<WeightsManagerGpuMemoryUsageTest::ParamType>& info) {
      return absl::StrCat(
          "weights_shape_", GetShapeName(std::get<0>(info.param)),
          "_weights_desc_", ToString(std::get<1>(info.param)),
          "_input_data_type_", ToString(std::get<2>(info.param)));
    });

}  // namespace ml_drift
