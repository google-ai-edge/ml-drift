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

#include <algorithm>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/winograd.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

namespace {
absl::Status ConvGenericWinograd4x4To6x6Test(TestExecutionEnvironment& exec_env,
                                             Convolution2DAttributes& attr,
                                             const TensorFloat32& src_tensor,
                                             const OperationDef& op_def,
                                             CalculationsPrecision precision) {
  auto& weights = GetFloatWeights(attr);
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  auto wino_up =
      std::make_unique<Winograd3x3TiledXForward>(CreateWinograd3x3TiledXForward(
          exec_env.GetGpuInfo(), op_def, attr.padding, 6));
  auto wino_down = std::make_unique<Winograd3x3TiledXBackward>(
      CreateWinograd3x3TiledXBackward(exec_env.GetGpuInfo(), op_def, attr.bias,
                                      6));

  const int tiles_x = DivideRoundUp(dst_ref_tensor.shape.w, 4);
  const int tiles_y = DivideRoundUp(dst_ref_tensor.shape.h, 4);
  TensorFloat32 wino_up_result;
  TensorFloat32 conv_result;
  wino_up_result.shape =
      BHWC(src_tensor.shape.b, 36, tiles_x * tiles_y, src_tensor.shape.c);
  conv_result.shape =
      BHWC(src_tensor.shape.b, 36, tiles_x * tiles_y, dst_ref_tensor.shape.c);

  Tensor<OHWI, DataType::FLOAT32> wino_weights;
  RearrangeWeightsToWinograd3x3TileNxN(weights, &wino_weights, 6);

  Convolution2DAttributes wino_attr;
  wino_attr.padding.prepended = HW(0, 0);
  wino_attr.padding.appended = HW(0, 0);
  wino_attr.strides = HW(1, 1);
  wino_attr.dilations = HW(1, 1);
  auto& wino_attr_weights =
      wino_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  wino_attr_weights.shape = wino_weights.shape;

  auto convolution = CreateConvGenericExternalWeights(
      exec_env.GetGpuInfo(), op_def, precision, wino_attr,
      /*bias=*/nullptr, &conv_result.shape,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true);
  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(wino_weights,
                                           convolution.GetWeightsDescription());

  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::move(wino_up), BHWC(wino_up_result.shape),
      &wino_up_result));
  {
    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(wino_up_result);

    std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
    srcs_td[0] = &src_td;
    for (int i = 0; i < weights_gpu.size(); ++i) {
      srcs_td[1 + i] = &weights_gpu[i];
    }
    TensorDescriptor dst_td = op_def.dst_tensors[0];
    dst_td.SetBHWCShape(BHWC(conv_result.shape));
    ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
        srcs_td, {&dst_td},
        std::make_unique<ConvGeneric>(std::move(convolution))));
    dst_td.DownloadData(&conv_result);
  }

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 3.0f;
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      conv_result, std::move(wino_down), dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionGenericExternalWeightsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    Convolution2DAttributes& attr, const OperationDef& conv_def,
    CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  auto& weights = GetFloatWeights(attr);
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor =
      ConvolutionReference(attr, src_tensor, aligned_runtime_channels);

  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), DeduceDataTypeFromPrecision(precision), attr.bias);
  auto operation = CreateConvGenericExternalWeights(
      exec_env.GetGpuInfo(), conv_def, precision, attr, &bias_tensor_desc,
      &dst_ref_tensor.shape, /*src_exp=*/nullptr,
      /*different_weights_for_height=*/false,
      runtime_channels.GenerateConvRuntimeCheckDesc());

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 2);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }
  srcs_td[weights_gpu.size() + 1] = &bias_tensor_desc;

  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  TensorInt32 params_tensor = runtime_channels.GenerateTensorInt32();
  if (std::any_of(params_tensor.data.begin(), params_tensor.data.end(),
                  [](int x) { return x != -1; })) {
    params_td.UploadData(params_tensor);
    srcs_td.push_back(&params_td);
  }

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td}, std::make_unique<ConvGeneric>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  const int dst_num_ch = dst_ref_tensor.shape.c;
  const int dst_end_ch = aligned_runtime_channels.dst_end_ch.has_value()
                             ? *aligned_runtime_channels.dst_end_ch
                             : dst_num_ch;
  for (int offset = 0; offset < dst_tensor.data.size(); offset += dst_num_ch) {
    std::vector<float> actual_data =
        std::vector<float>(dst_tensor.data.begin() + offset,
                           dst_tensor.data.begin() + offset + dst_end_ch);
    std::vector<float> expected_data =
        std::vector<float>(dst_ref_tensor.data.begin() + offset,
                           dst_ref_tensor.data.begin() + offset + dst_end_ch);
    EXPECT_THAT(actual_data, Pointwise(FloatNear(eps), expected_data));
  }
  return absl::OkStatus();
}

absl::Status ConvolutionGenericExternalBatchedWeightsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    Convolution2DAttributes& attr, const OperationDef& conv_def,
    CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  auto& weights = GetFloatWeights(attr);
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor = FullyConnectedRefDifferentWeightsForHeight(
      weights, src_tensor, aligned_runtime_channels);

  auto operation = CreateConvGenericExternalWeights(
      exec_env.GetGpuInfo(), conv_def, precision, attr, /*bias=*/nullptr,
      /*dst_shape=*/nullptr, /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true,
      runtime_channels.GenerateConvRuntimeCheckDesc());

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  TensorInt32 params_tensor = runtime_channels.GenerateTensorInt32();
  if (std::any_of(params_tensor.data.begin(), params_tensor.data.end(),
                  [](int x) { return x != -1; })) {
    params_td.UploadData(params_tensor);
    srcs_td.push_back(&params_td);
  }

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td}, std::make_unique<ConvGeneric>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  const int dst_num_ch = dst_ref_tensor.shape.c;
  const int dst_end_ch = aligned_runtime_channels.dst_end_ch.has_value()
                             ? *aligned_runtime_channels.dst_end_ch
                             : dst_num_ch;
  for (int offset = 0; offset < dst_tensor.data.size(); offset += dst_num_ch) {
    std::vector<float> actual_data =
        std::vector<float>(dst_tensor.data.begin() + offset,
                           dst_tensor.data.begin() + offset + dst_end_ch);
    std::vector<float> expected_data =
        std::vector<float>(dst_ref_tensor.data.begin() + offset,
                           dst_ref_tensor.data.begin() + offset + dst_end_ch);
    EXPECT_THAT(actual_data, Pointwise(FloatNear(eps), expected_data));
  }
  return absl::OkStatus();
}

absl::Status ConvolutionGenericBatchedMatMulTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& left_tensor,
    const TensorFloat32& right_tensor, const OperationDef& conv_def,
    CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor = BatchedMatMulReference(
      left_tensor, right_tensor, aligned_runtime_channels);

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape =
      OHWI(right_tensor.shape.c, right_tensor.shape.h, 1, right_tensor.shape.w);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int b = 0; b < right_tensor.shape.b; ++b) {
    for (int h = 0; h < right_tensor.shape.h; ++h) {
      for (int w = 0; w < right_tensor.shape.w; ++w) {
        for (int c = 0; c < right_tensor.shape.c; ++c) {
          int src_index = right_tensor.shape.LinearIndex({b, h, w, c});
          int dst_index = weights.shape.LinearIndex({c, h, b, w});
          weights.data[dst_index] = right_tensor.data[src_index];
        }
      }
    }
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = weights.shape;

  auto operation = CreateConvGenericExternalWeights(
      exec_env.GetGpuInfo(), conv_def, precision, attr, /*bias=*/nullptr,
      /*dst_shape=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
      runtime_channels.GenerateConvRuntimeCheckDesc());

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(left_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  TensorInt32 params_tensor = runtime_channels.GenerateTensorInt32();
  if (std::any_of(params_tensor.data.begin(), params_tensor.data.end(),
                  [](int x) { return x != -1; })) {
    params_td.UploadData(params_tensor);
    srcs_td.push_back(&params_td);
  }

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  int mads_count = left_tensor.shape.c;
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo()) * mads_count;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td}, std::make_unique<ConvGeneric>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  const int dst_num_ch = dst_ref_tensor.shape.c;
  const int dst_end_ch = aligned_runtime_channels.dst_end_ch.has_value()
                             ? *aligned_runtime_channels.dst_end_ch
                             : dst_num_ch;
  for (int offset = 0; offset < dst_tensor.data.size(); offset += dst_num_ch) {
    std::vector<float> actual_data =
        std::vector<float>(dst_tensor.data.begin() + offset,
                           dst_tensor.data.begin() + offset + dst_end_ch);
    std::vector<float> expected_data =
        std::vector<float>(dst_ref_tensor.data.begin() + offset,
                           dst_ref_tensor.data.begin() + offset + dst_end_ch);
    EXPECT_THAT(actual_data, Pointwise(FloatNear(eps), expected_data));
  }
  return absl::OkStatus();
}

}  // namespace

class ConvolutionFloatTest : public FloatTest {};

TEST_P(ConvolutionFloatTest, ConvolutionGenericWinograd4x4To6x6) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const int src_channels = 7;
  const int dst_channels = 13;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(10, 10);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto weights = MakeSyntheticTensor(OHWI(dst_channels, 3, 3, src_channels));
  attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(std::move(weights));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  auto src_shape = BHWC(1, 17, 13, src_channels);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
  MLD_EXPECT_OK(ConvGenericWinograd4x4To6x6Test(*exec_env, attr, src_tensor, op_def,
                                            precision()));
}

TEST_P(ConvolutionFloatTest, ConvolutionGenericWinograd4x4To6x6Batched) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  const int src_channels = 7;
  const int dst_channels = 13;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(10, 10);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 3, 3, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  auto src_shape = BHWC(2, 17, 13, src_channels);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage(), Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage(), Layout::BHWC});
  MLD_EXPECT_OK(ConvGenericWinograd4x4To6x6Test(*exec_env, attr, src_tensor, op_def,
                                            precision()));
}

TEST_P(ConvolutionFloatTest, ConvolutionGenericExternalWeights) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(1, 17, 13, 21);
  auto weights_shape = BHWC(23, 3, 2, 21);
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(2, 1);
  attr.dilations = HW(1, 2);
  attr.weights = MakeSyntheticTensor(
      OHWI(weights_shape.b, weights_shape.h, weights_shape.w, weights_shape.c));
  attr.bias = MakeSyntheticTensor(Linear(weights_shape.b));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef conv_def;
  conv_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
  conv_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  conv_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
  MLD_EXPECT_OK(ConvolutionGenericExternalWeightsTest(*exec_env, src_tensor, attr,
                                                  conv_def, precision(),
                                                  TestingRuntimeChannels()));
}

TEST_P(ConvolutionFloatTest, ConvolutionGenericExternalWeightsRuntimeCh) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(1, 1, 4, 64);
  auto weights_shape = BHWC(64, 2, 2, 64);
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(
      OHWI(weights_shape.b, weights_shape.h, weights_shape.w, weights_shape.c));
  attr.bias = MakeSyntheticTensor(Linear(weights_shape.b));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  std::vector<TestingRuntimeChannels> test_runtime_channels = {
      TestingRuntimeChannels(),
      {.dst_end_ch = 14},
      {.src_end_ch = 16},
      {.src_end_ch = 4, .dst_end_ch = 20}};
  for (auto runtime_channels : test_runtime_channels) {
    OperationDef conv_def;
    conv_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
    conv_def.src_tensors.push_back(
        {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
    conv_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
    MLD_EXPECT_OK(ConvolutionGenericExternalWeightsTest(
        *exec_env, src_tensor, attr, conv_def, precision(), runtime_channels));
  }
}

TEST_P(ConvolutionFloatTest, ConvolutionGenericExternalBatchedWeights) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(1, 6, 12, 128);
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(64, 6, 1, src_shape.c));
  attr.bias = MakeZeroTensor(Linear(
      std::visit([](auto& w) -> auto& { return w.shape; }, attr.weights).o));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  std::vector<TestingRuntimeChannels> test_runtime_channels = {
      TestingRuntimeChannels(),
      {.dst_end_ch = 13},
      {.src_end_ch = 12},
      {.src_end_ch = 18, .dst_end_ch = 19}};
  for (auto runtime_channels : test_runtime_channels) {
    OperationDef conv_def;
    conv_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
    conv_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
    MLD_EXPECT_OK(ConvolutionGenericExternalBatchedWeightsTest(
        *exec_env, src_tensor, attr, conv_def, precision(), runtime_channels));
  }
}

TEST_P(ConvolutionFloatTest, ConvolutionGenericBatchedMatMul) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision());
  if (!exec_env->IsStorageSupported(storage(), data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage());
  }
  auto left_shape = BHWC(1, 12, 128, 64);
  auto right_shape = BHWC(1, 12, 64, 64);

  TensorFloat32 left_tensor = MakeSyntheticTensor(left_shape);
  TensorFloat32 right_tensor = MakeSyntheticTensor(right_shape);

  std::vector<TestingRuntimeChannels> test_runtime_channels = {
      TestingRuntimeChannels(),
      {.dst_end_ch = 12},
      {.src_end_ch = 8},
      {.src_end_ch = 18, .dst_end_ch = 21}};

  for (auto runtime_channels : test_runtime_channels) {
    OperationDef conv_def;
    conv_def.src_tensors.push_back({data_type, storage(), Layout::HWC});
    conv_def.dst_tensors.push_back({data_type, storage(), Layout::HWC});
    MLD_EXPECT_OK(ConvolutionGenericBatchedMatMulTest(
        *exec_env, left_tensor, right_tensor, conv_def, precision(),
        runtime_channels));
  }
}

INSTANTIATE_TEST_SUITE_P(
    Suite, ConvolutionFloatTest,
    Combine(ValuesIn(GetCalculationsPrecisions()),
            ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ConvolutionFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
