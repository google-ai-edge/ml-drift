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

#include "ml_drift/common/kernels/tests/depthwise_conv_wave_memory_test_util.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/depthwise_conv_wave_memory.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

absl::Status DepthwiseConvWaveMemoryTest(
    TestExecutionEnvironment& exec_env,
    const DepthwiseConvolution2DAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      DepthWiseConvolutionReference(attr, src_tensor);

  auto operation = CreateDepthwiseConvWaveMemory(exec_env.GetGpuInfo(), op_def,
                                                 precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<DepthwiseConvWaveMemory>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, ::testing::Pointwise(::testing::FloatNear(eps),
                                                    dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status DepthwiseConvWaveMemory3x3Test(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 3;
  const int kernel_y = 3;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 7, 8, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  ABSL_RETURN_IF_ERROR(
      DepthwiseConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status DepthwiseConvWaveMemory3x3BatchedTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 3;
  const int kernel_y = 3;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(5, 7, 8, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  ABSL_RETURN_IF_ERROR(
      DepthwiseConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status DepthwiseConvWaveMemory5x5Test(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 5;
  const int kernel_y = 5;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 27, 18, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  ABSL_RETURN_IF_ERROR(
      DepthwiseConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status DepthwiseConvWaveMemory2x4Test(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 2;
  const int kernel_y = 4;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(2, 2);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 27, 18, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  ABSL_RETURN_IF_ERROR(
      DepthwiseConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status DepthwiseConvWaveMemory7x3Test(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 7;
  const int kernel_y = 3;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 2);
  attr.weights = MakeSyntheticTensor(OHWI(1, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 27, 18, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  ABSL_RETURN_IF_ERROR(
      DepthwiseConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status DepthwiseConvWaveMemory7x7Test(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = src_channels;
  const int kernel_x = 7;
  const int kernel_y = 7;
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(1, kernel_y, kernel_x, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 27, 18, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  ABSL_RETURN_IF_ERROR(
      DepthwiseConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

}  // namespace ml_drift
