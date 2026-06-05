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

#include "ml_drift/common/kernels/tests/depthwise_conv_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/depthwise_conv.h"
#include "ml_drift/common/kernels/depthwise_conv_3x3.h"
#include "ml_drift/common/kernels/depthwise_conv_tiled.h"
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

namespace {
absl::Status DepthwiseConvTest(TestExecutionEnvironment& exec_env,
                               const DepthwiseConvolution2DAttributes& attr,
                               const TensorFloat32& src_tensor,
                               const OperationDef& op_def,
                               CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      DepthWiseConvolutionReference(attr, src_tensor);

  auto operation = CreateDepthwiseConvolution2D(exec_env.GetGpuInfo(), op_def,
                                                precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status DepthwiseConvExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const DepthwiseConvolution2DAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 weights_tensor;
  auto& attr_weights =
      std::get<ml_drift::Tensor<OHWI, DataType::FLOAT32>>(attr.weights);
  weights_tensor.shape = BHWC(attr_weights.shape.o, attr_weights.shape.h,
                              attr_weights.shape.w, attr_weights.shape.i);
  weights_tensor.data = attr_weights.data;
  TensorFloat32 dst_ref_tensor =
      DepthWiseConvolutionReference(attr, src_tensor);

  auto operation = CreateDepthwiseConvolution2DExternalWeights(
      exec_env.GetGpuInfo(), op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {src_tensor, weights_tensor},
      std::make_unique<DepthwiseConv>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status DepthwiseConvTest(TestExecutionEnvironment& exec_env,
                               const DepthwiseConvolution3DAttributes& attr,
                               const Tensor5DFloat32& src_tensor,
                               const OperationDef& op_def,
                               CalculationsPrecision precision) {
  Tensor5DFloat32 dst_ref_tensor =
      DepthWiseConvolutionReference(attr, src_tensor);

  auto operation = CreateDepthwiseConvolution3D(exec_env.GetGpuInfo(), op_def,
                                                precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);

  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status DepthwiseConv3x3Test(TestExecutionEnvironment& exec_env,
                                  const DepthwiseConvolution2DAttributes& attr,
                                  const TensorFloat32& src_tensor,
                                  const OperationDef& op_def,
                                  CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      DepthWiseConvolutionReference(attr, src_tensor);

  auto operation =
      CreateDepthwiseConv3x3(exec_env.GetGpuInfo(), op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv3x3>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status DepthwiseConvTiledTest(
    TestExecutionEnvironment& exec_env,
    const DepthwiseConvolution2DAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      DepthWiseConvolutionReference(attr, src_tensor);

  auto operation =
      CreateDepthwiseConvTiled(exec_env.GetGpuInfo(), op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(src_tensor, std::move(operation),
                                         dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status DepthwiseConvSimpleWeightsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 3, 1, 2);
  attr_weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  attr.bias.shape = Linear(3);
  attr.bias.data = {0.0f, 0.0f, 0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  DepthwiseConv operation =
      CreateDepthwiseConvolution2D(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {4.0f, 4.0f, 8.0f, 8.0f, 4.0f, 4.0f, 8.0f, 8.0f}));
  return absl::OkStatus();
}

absl::Status DepthwiseConvNoMultiplierTest(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(1, 3, 1, 2);
  attr_weights.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  DepthwiseConv operation =
      CreateDepthwiseConvolution2D(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {16.5f, 27.5f, 28.5f, 43.5f, 8.5f,
                                         15.5f, 12.5f, 23.5f}));
  return absl::OkStatus();
}

absl::Status DepthwiseConvMultiplier2Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 3, 1, 2);
  attr_weights.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                       6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {0.5f, -0.5f, 1.0f, -1.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  DepthwiseConv operation =
      CreateDepthwiseConvolution2D(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv>(std::move(operation)),
      BHWC(1, 2, 2, 4), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps),
                {16.5f, 39.5f, 29.0f, 63.0f, 28.5f, 75.5f, 45.0f, 103.0f, 8.5f,
                 31.5f, 17.0f, 51.0f, 12.5f, 59.5f, 25.0f, 83.0f}));
  return absl::OkStatus();
}

absl::Status DepthwiseConvBigTest(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int channel_multiplier = 3;
  const int dst_channels = src_channels * channel_multiplier;
  const int kernel_x = 5;
  const int kernel_y = 4;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = MakeSyntheticTensor(
      OHWI(channel_multiplier, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 12, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return DepthwiseConvTest(env, attr, src_tensor, op_def, precision);
}

absl::Status DepthwiseConvBatchedBigTest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int channel_multiplier = 3;
  const int dst_channels = src_channels * channel_multiplier;
  const int kernel_x = 5;
  const int kernel_y = 4;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = MakeSyntheticTensor(
      OHWI(channel_multiplier, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(3, 12, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return DepthwiseConvTest(env, attr, src_tensor, op_def, precision);
}

absl::Status DepthwiseConvExternalWeightsTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 5;
  const int kernel_y = 4;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 12, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return DepthwiseConvExternalWeightsTest(env, attr, src_tensor, op_def,
                                          precision);
}

absl::Status DepthwiseConv3DTest(TestExecutionEnvironment& env,
                                 CalculationsPrecision precision,
                                 TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int channel_multiplier = 3;
  const int dst_channels = src_channels * channel_multiplier;
  const int kernel_x = 5;
  const int kernel_y = 4;
  const int kernel_z = 3;
  DepthwiseConvolution3DAttributes attr;
  attr.padding.prepended = HWD(1, 0, 2);
  attr.padding.appended = HWD(2, 1, 1);
  attr.strides = HWD(1, 2, 2);
  attr.dilations = HWD(2, 1, 2);
  attr.weights = MakeSyntheticTensor(
      OHWDI(channel_multiplier, kernel_y, kernel_x, kernel_z, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(1, 12, 9, 7, src_channels);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return DepthwiseConvTest(env, attr, src_tensor, op_def, precision);
}

absl::Status DepthwiseConv3DBatchedTest(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int channel_multiplier = 3;
  const int dst_channels = src_channels * channel_multiplier;
  const int kernel_x = 3;
  const int kernel_y = 2;
  const int kernel_z = 3;
  DepthwiseConvolution3DAttributes attr;
  attr.padding.prepended = HWD(1, 0, 2);
  attr.padding.appended = HWD(2, 1, 1);
  attr.strides = HWD(1, 2, 2);
  attr.dilations = HWD(2, 1, 2);
  attr.weights = MakeSyntheticTensor(
      OHWDI(channel_multiplier, kernel_y, kernel_x, kernel_z, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(3, 7, 6, 5, src_channels);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return DepthwiseConvTest(env, attr, src_tensor, op_def, precision);
}

absl::Status DepthwiseConv3DNoMultiplierTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 5;
  const int kernel_y = 4;
  const int kernel_z = 4;
  DepthwiseConvolution3DAttributes attr;
  attr.padding.prepended = HWD(1, 0, 0);
  attr.padding.appended = HWD(2, 1, 2);
  attr.strides = HWD(1, 2, 1);
  attr.dilations = HWD(2, 1, 1);
  attr.weights =
      MakeSyntheticTensor(OHWDI(1, kernel_y, kernel_x, kernel_z, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(1, 11, 10, 5, src_channels);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return DepthwiseConvTest(env, attr, src_tensor, op_def, precision);
}

// 3x3

absl::Status DepthwiseConv3x3SimpleWeightsTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(1, 3, 3, 2);
  attr_weights.data = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                       1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  DepthwiseConv3x3 operation =
      CreateDepthwiseConv3x3(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv3x3>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {6.0f, 16.0f, 8.0f, 16.0f, 10.0f, 16.0f,
                                         12.0f, 16.0f}));
  return absl::OkStatus();
}

absl::Status DepthwiseConv3x3Test(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(1, 3, 3, 2);
  attr_weights.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f,
                       3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  DepthwiseConv3x3 operation =
      CreateDepthwiseConv3x3(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<DepthwiseConv3x3>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {40.5f, 67.5f, 16.5f, 35.5f, 40.5f,
                                         67.5f, 16.5f, 35.5f}));
  return absl::OkStatus();
}

absl::Status DepthwiseConv3x3BigTest(TestExecutionEnvironment& env,
                                     CalculationsPrecision precision,
                                     TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = src_channels;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, 3, 3, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  auto src_shape = BHWC(1, 14, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return DepthwiseConv3x3Test(env, attr, src_tensor, op_def, precision);
}

absl::Status DepthwiseConv3x3BatchedBigTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = src_channels;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, 3, 3, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  auto src_shape = BHWC(3, 14, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return DepthwiseConv3x3Test(env, attr, src_tensor, op_def, precision);
}

// tiled

absl::Status DepthwiseConvTiledSimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(1, 3, 3, 2);
  attr_weights.data = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                       1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  auto operation =
      CreateDepthwiseConvTiled(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(src_tensor, std::move(operation),
                                          BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {6.0f, 16.0f, 8.0f, 16.0f, 10.0f, 16.0f,
                                         12.0f, 16.0f}));
  return absl::OkStatus();
}

absl::Status DepthwiseConvTiledTest(TestExecutionEnvironment& env,
                                    CalculationsPrecision precision,
                                    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(1, 3, 3, 2);
  attr_weights.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f,
                       3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  auto operation =
      CreateDepthwiseConvTiled(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(src_tensor, std::move(operation),
                                          BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {40.5f, 67.5f, 16.5f, 35.5f, 40.5f,
                                         67.5f, 16.5f, 35.5f}));
  return absl::OkStatus();
}

absl::Status DepthwiseConvTiledBigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = src_channels;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(2, 2);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, 3, 3, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  auto src_shape = BHWC(1, 14, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return DepthwiseConvTiledTest(env, attr, src_tensor, op_def, precision);
}

absl::Status DepthwiseConvTiledBatchedBigTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = src_channels;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(2, 2);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, 5, 3, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  auto src_shape = BHWC(3, 14, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return DepthwiseConvTiledTest(env, attr, src_tensor, op_def, precision);
}

absl::Status DepthwiseConvTiledWithDilationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = src_channels;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, 3, 5, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));
  auto src_shape = BHWC(1, 14, 9, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return DepthwiseConvTiledTest(env, attr, src_tensor, op_def, precision);
}

}  // namespace ml_drift
