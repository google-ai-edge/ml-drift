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

#include "ml_drift/common/kernels/tests/conv_transpose_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/convolution_transposed.h"
#include "ml_drift/common/kernels/convolution_transposed_2x2.h"
#include "ml_drift/common/kernels/convolution_transposed_3x3.h"
#include "ml_drift/common/kernels/convolution_transposed_4x4.h"
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

namespace {
absl::Status ConvolutionTransposedTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed(exec_env.GetGpuInfo(), op_def,
                                               precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvolutionTransposed>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3DTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposed3DAttributes& attr,
    const Tensor5DFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  Tensor5DFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed3D(exec_env.GetGpuInfo(), op_def,
                                                 precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  Tensor5DFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvolutionTransposed>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposedDynamicWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposedDynamicWeights(
      exec_env.GetGpuInfo(), op_def, precision, attr);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr.weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvolutionTransposed>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvTransposedSimpleWeightsTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(2, 2, 2, 2);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr.weights.shape.DimensionsProduct(); ++i) {
    attr.weights.data[i] = 1.0f;
  }
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvolutionTransposed operation =
      CreateConvolutionTransposed(env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvolutionTransposed>(std::move(operation)),
      BHWC(1, 4, 4, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.0f, 1.0f, 1.0f, 1.0f, 5.0f,  5.0f,  5.0f,  5.0f,
                         1.0f, 1.0f, 1.0f, 1.0f, 5.0f,  5.0f,  5.0f,  5.0f,
                         9.0f, 9.0f, 9.0f, 9.0f, 13.0f, 13.0f, 13.0f, 13.0f,
                         9.0f, 9.0f, 9.0f, 9.0f, 13.0f, 13.0f, 13.0f, 13.0f}));
  return absl::OkStatus();
}

absl::Status ConvTransposedTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(1, 2, 2, 2);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr.weights.shape.DimensionsProduct(); ++i) {
    attr.weights.data[i] = i + 1;
  }
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvolutionTransposed operation =
      CreateConvolutionTransposed(env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvolutionTransposed>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {2.5f, 4.5f, 8.5f, 18.5f, 6.5f, 8.5f,
                                         28.5f, 38.5f, 14.5f, 32.5f, 20.5f,
                                         46.5f, 50.5f, 68.5f, 72.5f, 98.5f}));
  return absl::OkStatus();
}

absl::Status ConvTransposedBigTest(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = 7;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(-2, -1);
  attr.padding.appended = HW(-7, 3);
  attr.stride = HW(4, 5);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 3, 4, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 20, 15, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ConvolutionTransposedTest(env, attr, src_tensor, op_def, precision);
}

absl::Status ConvTransposedBatchedBigTest(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = 7;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, -1);
  attr.stride = HW(5, 1);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 6, 1, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(3, 1, 3, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return ConvolutionTransposedTest(env, attr, src_tensor, op_def, precision);
}

absl::Status ConvTransposed3DBigTest(TestExecutionEnvironment& env,
                                     CalculationsPrecision precision,
                                     TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = 7;
  ConvolutionTransposed3DAttributes attr;
  attr.padding.prepended = HWD(-2, -1, 2);
  attr.padding.appended = HWD(-7, 3, -2);
  attr.stride = HWD(4, 5, 3);
  attr.weights =
      MakeSyntheticTensor(OHWDI(dst_channels, 3, 4, 3, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(1, 20, 15, 13, src_channels);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return ConvolutionTransposed3DTest(env, attr, src_tensor, op_def, precision);
}

absl::Status ConvTransposed3DBatchedBigTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = 7;
  ConvolutionTransposed3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, -1);
  attr.padding.appended = HWD(0, -1, 1);
  attr.stride = HWD(5, 1, 2);
  attr.weights =
      MakeSyntheticTensor(OHWDI(dst_channels, 6, 1, 2, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(3, 1, 3, 4, src_channels);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return ConvolutionTransposed3DTest(env, attr, src_tensor, op_def, precision);
}

absl::Status ConvTransposedExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 5;
  const int dst_channels = 7;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(-2, -1);
  attr.padding.appended = HW(-7, 3);
  attr.stride = HW(4, 5);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 3, 4, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 20, 15, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ConvolutionTransposedDynamicWeightsTest(env, attr, src_tensor, op_def,
                                                 precision);
}

namespace {
absl::Status ConvolutionTransposed2x2Test(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed2x2(exec_env.GetGpuInfo(), op_def,
                                                  precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed2x2>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_ref_tensor.data, Pointwise(FloatNear(eps), dst_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed2x2DynamicWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed2x2DynamicWeights(
      exec_env.GetGpuInfo(), op_def, precision, attr);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr.weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvolutionTransposed2x2>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_ref_tensor.data, Pointwise(FloatNear(eps), dst_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvolutionTransposed2x2Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 13;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 2, 2, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 17, 13, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  ABSL_RETURN_IF_ERROR(
      ConvolutionTransposed2x2Test(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed2x2BatchedTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 13;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 2, 2, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(3, 17, 13, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  ABSL_RETURN_IF_ERROR(
      ConvolutionTransposed2x2Test(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed2x2DynamicWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 13;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 2, 2, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 17, 13, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  ABSL_RETURN_IF_ERROR(ConvolutionTransposed2x2DynamicWeightsTest(
      env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

namespace {
absl::Status ConvolutionTransposed3x3Test(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed3x3(exec_env.GetGpuInfo(), op_def,
                                                  precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed3x3>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_ref_tensor.data, Pointwise(FloatNear(eps), dst_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed3x3DynamicWeights(
      exec_env.GetGpuInfo(), op_def, precision, attr);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr.weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvolutionTransposed3x3>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_ref_tensor.data, Pointwise(FloatNear(eps), dst_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvolutionTransposed3x3SimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(1, 3, 3, 1);
  attr.weights.data.resize(
      attr.weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float),
      1.0f);
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvolutionTransposed3x3 operation =
      CreateConvolutionTransposed3x3(env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed3x3>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 6.0f, 4.0f, 4.0f, 2.0f,
                         5.0f, 3.0f, 3.0f, 2.0f, 5.0f, 3.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = 5;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, -2);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 3, 3, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 3, 3, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  ABSL_RETURN_IF_ERROR(
      ConvolutionTransposed3x3Test(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3BatchedTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = 5;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(2, -1);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 3, 3, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(5, 3, 3, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  ABSL_RETURN_IF_ERROR(
      ConvolutionTransposed3x3Test(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3ExternalWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const int src_channels = 5;
  const int dst_channels = 6;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, -2);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 3, 3, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 13, 7, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  ABSL_RETURN_IF_ERROR(ConvolutionTransposed3x3ExternalWeightsTest(
      env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

// 4x4
namespace {
absl::Status ConvolutionTransposed4x4Test(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed4x4(exec_env.GetGpuInfo(), op_def,
                                                  precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed4x4>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_ref_tensor.data, Pointwise(FloatNear(eps), dst_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed4x4ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed4x4DynamicWeights(
      exec_env.GetGpuInfo(), op_def, precision, attr);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr.weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvolutionTransposed4x4>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_ref_tensor.data, Pointwise(FloatNear(eps), dst_tensor.data));
  return absl::OkStatus();
}

}  // namespace

absl::Status ConvolutionTransposed4x4SimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(1, 4, 4, 1);
  attr.weights.data.resize(
      attr.weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float),
      1.0f);
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvolutionTransposed4x4 operation =
      CreateConvolutionTransposed4x4(env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed4x4>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 6.0f, 6.0f, 4.0f, 2.0f,
                         6.0f, 6.0f, 4.0f, 2.0f, 5.0f, 5.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed4x4Test(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 13;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 4, 4, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 5, 4, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  ABSL_RETURN_IF_ERROR(
      ConvolutionTransposed4x4Test(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed4x4BatchedTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 13;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 4, 4, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(6, 5, 4, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  ABSL_RETURN_IF_ERROR(
      ConvolutionTransposed4x4Test(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed4x4ExternalWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 13;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(dst_channels, 4, 4, src_channels);
  attr.weights = MakeSyntheticTensor(attr.weights.shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 5, 4, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  ABSL_RETURN_IF_ERROR(ConvolutionTransposed4x4ExternalWeightsTest(
      env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

}  // namespace ml_drift
