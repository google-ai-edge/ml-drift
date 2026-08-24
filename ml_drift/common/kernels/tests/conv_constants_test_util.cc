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

#include "ml_drift/common/kernels/tests/conv_constants_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/conv_constants.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status ConvConstantsSimpleWeightsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  weights.shape = OHWI(1, 2, 2, 2);
  weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvConstants operation =
      CreateConvConstants(env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvConstants>(std::move(operation)),
      BHWC(1, 2, 2, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {28.0f, 18.0f, 22.0f, 13.0f}));
  return absl::OkStatus();
}

absl::Status ConvConstantsTest(TestExecutionEnvironment& env,
                               CalculationsPrecision precision,
                               TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  weights.shape = OHWI(2, 2, 2, 2);
  weights.data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvConstants operation =
      CreateConvConstants(env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvConstants>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {168.5f, 391.5f, 80.5f, 223.5f, 60.5f,
                                         235.5f, 20.5f, 123.5f}));
  return absl::OkStatus();
}

namespace {
absl::Status ConvConstantsTest(TestExecutionEnvironment& exec_env,
                               const Convolution2DAttributes& attr,
                               const TensorFloat32& src_tensor,
                               const OperationDef& op_def,
                               CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  auto operation =
      CreateConvConstants(exec_env.GetGpuInfo(), op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvConstants>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvConstantsBatchedBigTest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 5;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 2);
  attr.padding.appended = HW(3, 1);
  attr.strides = HW(3, 2);
  attr.dilations = HW(1, 2);
  auto synthetic_tensor =
      MakeSyntheticTensor(OHWI(dst_channels, 3, 2, src_channels));
  auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_tensor));
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(7, 7, 5, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  MLD_EXPECT_OK(ConvConstantsTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvConstantsBigTest(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage, int src_channel,
                                  int dst_channel) {
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_tensor =
      MakeSyntheticTensor(OHWI(dst_channel, 2, 3, src_channel));
  auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_tensor));
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channel));

  auto src_shape = BHWC(1, 17, 23, src_channel);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(ConvConstantsTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

// Big test vs ref implementation.
absl::Status ConvConstantsExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  auto src_shape = BHWC(1, 27, 33, 7);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  int dst_ch = 9;

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(2, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_tensor =
      MakeSyntheticTensor(OHWI(dst_ch, 3, 2, src_tensor.shape.c));
  auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_tensor));
  weights.data.resize(weights.data.size() + XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_ch));

  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      env.GetGpuInfo(), data_type, attr.bias);
  auto operation = CreateConvConstantsExternalWeights(
      env.GetGpuInfo(), conv_def, precision, attr, &bias_tensor_desc);

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

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvConstants>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
