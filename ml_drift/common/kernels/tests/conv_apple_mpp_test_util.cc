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

#include "ml_drift/common/kernels/tests/conv_apple_mpp_test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/conv_apple_mpp.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
absl::Status ConvAppleMPPBigTest(TestExecutionEnvironment& env,
                                 TensorStorageType dst_storage,
                                 const BHWC& src_shape, int dst_channels) {
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{
      DataType::FLOAT16, TensorStorageType::BUFFER, layout,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::FLOAT16, dst_storage, layout};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_shape.c));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  const TensorFloat32 dst_ref_tensor =
      FullyConnectedReference(attr, src_tensor);

  std::unique_ptr<GPUOperation> conv =
      std::make_unique<ConvAppleMPP>(CreateConvAppleMPP(
          src_tensor_desc, dst_tensor_desc, attr.weights, attr.bias));

  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(src_tensor, std::move(conv),
                                          dst_ref_tensor.shape, &dst_tensor));
  float eps = GetEpsilon(CalculationsPrecision::F16, env.GetGpuInfo(), attr);
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvAppleMPPExternalWeightsTest(TestExecutionEnvironment& env,
                                             TensorStorageType dst_storage,
                                             const BHWC& src_shape,
                                             int dst_channels) {
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{
      DataType::FLOAT16, TensorStorageType::BUFFER, layout,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::FLOAT16, dst_storage, layout};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_shape.c));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  const TensorFloat32 dst_ref_tensor =
      FullyConnectedReference(attr, src_tensor);

  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      env.GetGpuInfo(), DataType::FLOAT16, attr.bias);

  ConvAppleMPP conv = CreateConvAppleMPPExternalWeights(
      src_tensor_desc, dst_tensor_desc, attr.weights.shape, &bias_tensor_desc);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr.weights,
                                           conv.GetWeightsDescription());

  TensorDescriptor src_td = src_tensor_desc;
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 2);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }
  srcs_td[weights_gpu.size() + 1] = &bias_tensor_desc;

  TensorDescriptor dst_td = dst_tensor_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      srcs_td, {&dst_td}, std::make_unique<ConvAppleMPP>(std::move(conv))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  float eps = GetEpsilon(CalculationsPrecision::F16, env.GetGpuInfo(), attr);
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvAppleMPPExternalBatchedWi4Test(TestExecutionEnvironment& env,
                                                TensorStorageType storage,
                                                const BHWC& src_shape,
                                                int dst_channels) {
  TensorFloat32 src_tensor;
  const int src_channels = src_shape.c;
  src_tensor.shape = src_shape;
  const BHWC dst_shape(1, src_shape.h, src_shape.w, dst_channels);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, src_tensor.shape.h, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 = MakeSyntheticTensor(
      OHWI(dst_channels, src_tensor.shape.h, 1, src_channels));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scales =
      MakeSyntheticTensor(OHWI(dst_channels, src_tensor.shape.h, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 8.0f;
  }
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scales.shape;
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(),
                                 0.0f);

  auto weights =
      MakeWeightsFromInt8(weights_i4, weights_scales, weights_zero_point);
  TensorFloat32 dst_ref_tensor =
      FullyConnectedRefDifferentWeightsForHeight(weights, src_tensor);

  Layout layout = src_tensor.shape.b == 1 ? Layout::HWC : Layout::BHWC;
  DataType float_type = DataType::FLOAT16;
  TensorDescriptor src_tensor_desc{
      float_type, TensorStorageType::BUFFER, layout,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{float_type, storage, layout};

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(env.GetGpuInfo(), weights_i4.shape);

  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(),
                                                   weights_scales, float_type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(),
                                                weights_zero_point, float_type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;
  auto operation = CreateConvAppleMPPExternalWeights(
      src_tensor_desc, dst_tensor_desc, external_weights,
      /*bias=*/nullptr,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true);

  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  TensorDescriptor src_td = src_tensor_desc;
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = dst_tensor_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(CalculationsPrecision::F16, env.GetGpuInfo()) *
              weights_i4.shape.i * weights_i4.shape.o;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i4_td, &scale_desc, &zp_desc}, {&dst_td},
      std::make_unique<ConvAppleMPP>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvAppleMPPBatchedMatMulTest(TestExecutionEnvironment& env,
                                           TensorStorageType dst_storage,
                                           const BHWC& left_shape,
                                           const BHWC& right_shape) {
  TensorFloat32 left_tensor = MakeSyntheticTensor(left_shape);
  TensorFloat32 right_tensor = MakeSyntheticTensor(right_shape);
  TensorFloat32 dst_ref_tensor =
      BatchedMatMulReference(left_tensor, right_tensor);

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

  TensorDescriptor src_tensor_desc{
      DataType::FLOAT16, TensorStorageType::BUFFER, Layout::HWC,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::FLOAT16, dst_storage, Layout::HWC};
  ConvAppleMPP conv = CreateConvAppleMPPExternalWeights(
      src_tensor_desc, dst_tensor_desc, weights.shape, /*bias=*/nullptr,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           conv.GetWeightsDescription());

  TensorDescriptor src_td = src_tensor_desc;
  src_td.UploadData(left_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor dst_td = dst_tensor_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  int mads_count = left_tensor.shape.c;
  float eps =
      GetEpsilon(CalculationsPrecision::F16, env.GetGpuInfo()) * mads_count;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      srcs_td, {&dst_td}, std::make_unique<ConvAppleMPP>(std::move(conv))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryRuntimeChannelsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    const FullyConnectedAttributes& attr, TensorStorageType dst_storage,
    const TestingRuntimeChannels& runtime_channels) {
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor;
  {
    Convolution2DAttributes conv_attr;
    conv_attr.padding.prepended = HW(0, 0);
    conv_attr.padding.appended = HW(0, 0);
    conv_attr.strides = HW(1, 1);
    conv_attr.dilations = HW(1, 1);
    conv_attr.weights = attr.weights;
    conv_attr.bias = attr.bias;
    dst_ref_tensor =
        ConvolutionReference(conv_attr, src_tensor, aligned_runtime_channels);
  }

  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), DataType::FLOAT16, attr.bias);

  Layout layout = src_tensor.shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{
      DataType::FLOAT16, TensorStorageType::BUFFER, layout,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::FLOAT16, dst_storage, layout};
  auto operation = CreateConvAppleMPPExternalWeights(
      src_tensor_desc, dst_tensor_desc, attr.weights.shape, &bias_tensor_desc,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/false,
      runtime_channels.GenerateConvRuntimeCheckDesc());

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr.weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = src_tensor_desc;
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

  TensorDescriptor dst_td = dst_tensor_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps =
      GetEpsilon(CalculationsPrecision::F16, exec_env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvAppleMPP>(std::move(operation))));
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
    EXPECT_THAT(actual_data,
                testing::Pointwise(testing::FloatNear(eps), expected_data));
  }
  return absl::OkStatus();
}

absl::Status ConvAppleMPPRuntimeSrcEndChannelsTest(
    TestExecutionEnvironment& env, TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 32, 128);
  auto weights_shape = OHWI(256, 1, 1, 128);
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(weights_shape);
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(weights_shape.o));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  for (int src_ch = 4; src_ch <= weights_shape.i; src_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.src_end_ch = src_ch;
    MLD_EXPECT_OK(ConvWaveMemoryRuntimeChannelsTest(env, src_tensor, attr, storage,
                                                runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status ConvAppleMPPRuntimeDstEndChannelsTest(
    TestExecutionEnvironment& env, TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 32, 128);
  auto weights_shape = OHWI(256, 1, 1, 128);
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(weights_shape);
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(weights_shape.o));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  for (int dst_ch = 0; dst_ch <= weights_shape.o; dst_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.dst_end_ch = dst_ch;
    MLD_EXPECT_OK(ConvWaveMemoryRuntimeChannelsTest(env, src_tensor, attr, storage,
                                                runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status ConvAppleMPPInt8BigTest(TestExecutionEnvironment& env,
                                     TensorStorageType dst_storage,
                                     const BHWC& src_shape, int dst_channels) {
  const int src_channels = src_shape.c;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  ml_drift::Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{
      DataType::INT8, TensorStorageType::BUFFER, layout,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::INT32, dst_storage, layout};

  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_i8_tensor, weights_i8);

  std::unique_ptr<GPUOperation> conv = std::make_unique<ConvAppleMPP>(
      CreateConvAppleMPPInt8(src_tensor_desc, dst_tensor_desc, weights_i8));

  TensorDescriptor src_td = src_tensor_desc;
  src_td.UploadData(src_i8_tensor);

  TensorDescriptor dst_td = dst_tensor_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src_td}, {&dst_td}, std::move(conv)));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ConvAppleMPPInt8ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType dst_storage,
    const BHWC& src_shape, int dst_channels) {
  const int src_channels = src_shape.c;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  ml_drift::Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{
      DataType::INT8, TensorStorageType::BUFFER, layout,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::INT32, dst_storage, layout};

  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_i8_tensor, weights_i8);

  auto conv = CreateConvAppleMPPInt8(src_tensor_desc, dst_tensor_desc,
                                     weights_i8.shape);

  TensorDescriptor weights_i8_td;
  {
    WeightsDescription weights_desc = conv.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape));
    RearrangeWeights(weights_i8, weights_desc, absl::MakeSpan(data));

    weights_i8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i8_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor src_td = src_tensor_desc;
  src_td.UploadData(src_i8_tensor);

  TensorDescriptor dst_td = dst_tensor_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src_td, &weights_i8_td}, {&dst_td},
                              std::make_unique<ConvAppleMPP>(std::move(conv))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ConvAppleMPPInt8ExternalBatchedWi4Test(
    TestExecutionEnvironment& env, TensorStorageType dst_storage,
    const BHWC& src_shape, int dst_channels) {
  const int src_channels = src_shape.c;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, src_shape.h, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, src_shape.h, 1, src_channels));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  ml_drift::Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{
      DataType::INT8, TensorStorageType::BUFFER, layout,
      TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::INT32, dst_storage, layout};

  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_i8_tensor, weights_i4);

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(env.GetGpuInfo(), weights_i4.shape);
  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  auto conv = CreateConvAppleMPPInt8(src_tensor_desc, dst_tensor_desc,
                                     external_weights);

  TensorDescriptor src_td = src_tensor_desc;
  src_td.UploadData(src_i8_tensor);

  TensorDescriptor dst_td = dst_tensor_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src_td, &weights_i4_td}, {&dst_td},
                              std::make_unique<ConvAppleMPP>(std::move(conv))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}

namespace {

absl::Status ConvAppleMPPInt8WithSrcQuantizationTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>* weights_zero_point,
    const DataType float_type, const TensorStorageType float_storage_type,
    const TensorStorageType int_storage_type) {
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = MakeZeroTensor(Linear(weights_i8.shape.o));
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  TensorDescriptor src_params_td;
  const Layout layout = src_tensor.shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_i8_td{DataType::INT8, TensorStorageType::BUFFER, layout,
                             TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  {
    OperationDef op_def;
    op_def.src_tensors.push_back({float_type, float_storage_type, layout});
    op_def.dst_tensors.push_back(src_i8_td);
    op_def.dst_tensors.push_back({float_type, float_storage_type, layout});
    const bool calculate_sum = weights_zero_point != nullptr;
    auto quantization_op =
        CreateQuantization(op_def, PackedType::kInt8C4, exec_env.GetGpuInfo(),
                           src_tensor.shape, calculate_sum);
    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(src_tensor);
    src_i8_td.SetBHWCShape(src_tensor.shape);
    src_params_td = op_def.dst_tensors[1];
    BHWC params_shape = src_tensor.shape;
    params_shape.c = calculate_sum ? 3 : 2;
    src_params_td.SetBHWCShape(params_shape);

    RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
        {&src_td}, {&src_i8_td, &src_params_td}, std::move(quantization_op)));
  }

  TensorDescriptor weights_scale_td = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), float_type, weights_scale);
  TensorDescriptor weights_zero_point_td;
  if (weights_zero_point != nullptr) {
    weights_zero_point_td = CreateConstantLinearTensorDescriptor(
        exec_env.GetGpuInfo(), float_type, *weights_zero_point);
  }
  TensorDescriptor* weights_zero_point_td_ptr =
      weights_zero_point != nullptr ? &weights_zero_point_td : nullptr;

  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);
  auto weights_sum_i_td = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), weights_sum_i);

  TensorDescriptor dst_tensor_desc{DataType::INT32, int_storage_type, layout};
  auto conv =
      CreateConvAppleMPPInt8(src_i8_td, dst_tensor_desc, weights_i8.shape);

  TensorDescriptor dequant_dst = {float_type, float_storage_type, layout};
  auto dequant_op = CreateDequantization(
      weights_i8.shape, exec_env.GetGpuInfo(), dst_tensor_desc, dequant_dst,
      src_params_td, weights_sum_i_td, weights_scale_td,
      weights_zero_point_td_ptr);
  RETURN_IF_ERROR(conv.AddOperation(exec_env.GetGpuInfo(), &dequant_op));

  TensorDescriptor weights_i8_td;
  {
    WeightsDescription weights_desc = conv.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape));
    RearrangeWeights(weights_i8, weights_desc, absl::MakeSpan(data));

    weights_i8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i8_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor dst_td = dequant_dst;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  std::vector<TensorDescriptor*> src_descs = {&src_i8_td, &weights_i8_td,
                                              &src_params_td, &weights_sum_i_td,
                                              &weights_scale_td};
  if (weights_zero_point != nullptr) {
    src_descs.push_back(&weights_zero_point_td);
  }
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_descs, {&dst_td}, std::make_unique<ConvAppleMPP>(std::move(conv))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  int mads_count = weights_i8.shape.i;
  float eps = 0.01f * mads_count;
  if (weights_zero_point != nullptr) {
    eps *= 2.0f;
  }
  EXPECT_THAT(dst_tensor.data, ::testing::Pointwise(::testing::FloatNear(eps),
                                                    dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvAppleMPPInt8WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType int_storage,
    TensorStorageType float_storage, DataType float_type, const BHWC& src_shape,
    int dst_channels) {
  if (float_type != DataType::FLOAT16 && float_type != DataType::FLOAT32) {
    return absl::InvalidArgumentError("Unsupported float type: " +
                                      ToString(float_type));
  }
  const int src_channels = src_shape.c;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }
  auto weights_scale = MakeSyntheticTensor(Linear(dst_channels));
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] /= 127.0f;
  }
  auto weights_zp = MakeSyntheticTensor(Linear(dst_channels));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 2.0f + 2.5f;
  }

  RETURN_IF_ERROR(ConvAppleMPPInt8WithSrcQuantizationTest(
      env, src_tensor, weights_i8, weights_scale,
      /*weights_zero_point=*/nullptr, float_type, float_storage, int_storage));
  RETURN_IF_ERROR(ConvAppleMPPInt8WithSrcQuantizationTest(
      env, src_tensor, weights_i8, weights_scale, &weights_zp, float_type,
      float_storage, int_storage));
  return absl::OkStatus();
}

}  // namespace ml_drift
