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

#include "ml_drift/common/kernels/tests/conv_transpose_thin_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/convolution_transposed_3x3_thin.h"
#include "ml_drift/common/kernels/convolution_transposed_thin.h"
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
absl::Status ConvolutionTransposedThinTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposedThin(exec_env.GetGpuInfo(),
                                                   op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposedThin>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3ThinTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed3x3Thin(exec_env.GetGpuInfo(),
                                                      op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed3x3Thin>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3ThinDynamicWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ConvolutionTransposedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      ConvolutionTransposedReference(attr, src_tensor);

  auto operation = CreateConvolutionTransposed3x3ThinDynamicWeights(
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
      std::make_unique<ConvolutionTransposed3x3Thin>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvolutionTransposedThinSimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(2, 2, 2, 2);
  attr.weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvolutionTransposedThin operation = CreateConvolutionTransposedThin(
      env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposedThin>(std::move(operation)),
      BHWC(1, 4, 4, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.0f, 1.0f, 1.0f, 1.0f, 5.0f,  5.0f,  5.0f,  5.0f,
                         1.0f, 1.0f, 1.0f, 1.0f, 5.0f,  5.0f,  5.0f,  5.0f,
                         9.0f, 9.0f, 9.0f, 9.0f, 13.0f, 13.0f, 13.0f, 13.0f,
                         9.0f, 9.0f, 9.0f, 9.0f, 13.0f, 13.0f, 13.0f, 13.0f}));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposedThinTest(TestExecutionEnvironment& env,
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
  attr.weights.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvolutionTransposedThin operation = CreateConvolutionTransposedThin(
      env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposedThin>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {2.5f, 4.5f, 8.5f, 18.5f, 6.5f, 8.5f,
                                         28.5f, 38.5f, 14.5f, 32.5f, 20.5f,
                                         46.5f, 50.5f, 68.5f, 72.5f, 98.5f}));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposedThinBigTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 7;
  const int dst_channels = 1;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 2, 2, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 8, 8, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ConvolutionTransposedThinTest(env, attr, src_tensor, op_def,
                                       precision);
}

absl::Status ConvolutionTransposedThinBatchedBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 7;
  const int dst_channels = 1;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 2, 2, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(3, 8, 8, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return ConvolutionTransposedThinTest(env, attr, src_tensor, op_def,
                                       precision);
}

absl::Status ConvolutionTransposed3x3ThinSimpleWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(1, 3, 3, 1);
  attr.weights.data.resize(
      attr.weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float),
      1.0f);
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvolutionTransposed3x3Thin operation = CreateConvolutionTransposed3x3Thin(
      env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed3x3Thin>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 6.0f, 4.0f, 4.0f, 2.0f,
                         5.0f, 3.0f, 3.0f, 2.0f, 5.0f, 3.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3ThinTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(1, 3, 3, 1);
  attr.weights.data.resize(
      attr.weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float),
      1.0f);
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
  ConvolutionTransposed3x3Thin operation = CreateConvolutionTransposed3x3Thin(
      env.GetGpuInfo(), op_def, precision, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<ConvolutionTransposed3x3Thin>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.5f, 4.5f, 5.5f, 6.5f, 4.5f, 16.5f,
                                         14.5f, 18.5f, 10.5f, 24.5f, 15.5f,
                                         18.5f, 16.5f, 39.5f, 24.5f, 27.5f}));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3ThinBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 3;
  const int dst_channels = 5;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.adjacent = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 3, 3, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 3, 3, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ConvolutionTransposed3x3ThinTest(env, attr, src_tensor, op_def,
                                          precision);
}

absl::Status ConvolutionTransposed3x3ThinBatchedBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 3;
  const int dst_channels = 5;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.adjacent = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 3, 3, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(5, 3, 3, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return ConvolutionTransposed3x3ThinTest(env, attr, src_tensor, op_def,
                                          precision);
}

absl::Status ConvolutionTransposed3x3ThinExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 3;
  const int dst_channels = 5;
  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.adjacent = HW(1, 1);
  attr.stride = HW(2, 2);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 3, 3, src_channels));
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 6, 8, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ConvolutionTransposed3x3ThinDynamicWeightsTest(env, attr, src_tensor,
                                                        op_def, precision);
}

}  // namespace ml_drift
