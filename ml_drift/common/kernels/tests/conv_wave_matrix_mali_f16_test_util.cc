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

#include "ml_drift/common/kernels/tests/conv_wave_matrix_mali_f16_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/conv_wave_matrix_mali_f16.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace {
absl::Status ConvWaveMatrixMaliF16Test(
    TestExecutionEnvironment& exec_env, const Convolution2DAttributes& attr,
    const Tensor<OHWI, DataType::kFloat32>& weights,
    const TensorFloat32& src_tensor, const TensorDescriptor& src_desc,
    const TensorDescriptor& dst_desc) {
  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  auto operation = CreateConvWaveMatrixMaliF16(exec_env.GetGpuInfo(), src_desc,
                                               dst_desc, attr);
  float eps =
      GetEpsilon(CalculationsPrecision::kF16, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvWaveMatrixMaliF16>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixMaliF16ExternalWeightsTest(
    TestExecutionEnvironment& exec_env, const Convolution2DAttributes& attr,
    const Tensor<OHWI, DataType::kFloat32>& weights,
    const TensorFloat32& src_tensor, const TensorDescriptor& src_desc,
    const TensorDescriptor& dst_desc, bool different_weights_for_height) {
  TensorFloat32 dst_ref_tensor =
      different_weights_for_height
          ? FullyConnectedRefDifferentWeightsForHeight(weights, src_tensor)
          : ConvolutionReference(attr, src_tensor);

  auto operation = CreateConvWaveMatrixMaliF16ExternalWeights(
      exec_env.GetGpuInfo(), src_desc, dst_desc, attr, /*bias=*/nullptr,
      different_weights_for_height);
  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = src_desc;
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor dst_td = dst_desc;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps =
      GetEpsilon(CalculationsPrecision::kF16, exec_env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMatrixMaliF16>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvWaveMatrixMaliF161x1BigTest(TestExecutionEnvironment& env,
                                             TensorStorageType storage) {
  const int src_channels = 32;
  const int dst_channels = 64;
  auto weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = weights;
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 3, 17, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  TensorDescriptor src_desc = {DataType::kFloat16, storage, Layout::kHWC};
  TensorDescriptor dst_desc = {DataType::kFloat16, storage, Layout::kHWC};
  return ConvWaveMatrixMaliF16Test(env, attr, weights, src_tensor, src_desc,
                                   dst_desc);
}

absl::Status ConvWaveMatrixMaliF16BigTest(TestExecutionEnvironment& env,
                                          TensorStorageType storage) {
  const int src_channels = 11;
  const int dst_channels = 27;
  auto weights = MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels));
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = weights;
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 9, 17, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  TensorDescriptor src_desc = {DataType::kFloat16, storage, Layout::kHWC};
  TensorDescriptor dst_desc = {DataType::kFloat16, storage, Layout::kHWC};
  return ConvWaveMatrixMaliF16Test(env, attr, weights, src_tensor, src_desc,
                                   dst_desc);
}

absl::Status ConvWaveMatrixMaliF16ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType storage) {
  const int src_channels = 11;
  const int dst_channels = 27;
  auto weights = MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels));
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  attr.weights = weights;
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 9, 17, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  TensorDescriptor src_desc = {DataType::kFloat16, storage, Layout::kHWC};
  TensorDescriptor dst_desc = {DataType::kFloat16, storage, Layout::kHWC};
  return ConvWaveMatrixMaliF16ExternalWeightsTest(
      env, attr, weights, src_tensor, src_desc, dst_desc,
      /*different_weights_for_height=*/false);
}

absl::Status ConvWaveMatrixMaliF16ExternalBatchedWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType storage) {
  const int src_channels = 32;
  const int dst_channels = 64;
  auto src_shape = BHWC(1, 6, 17, src_channels);
  auto weights =
      MakeSyntheticTensor(OHWI(dst_channels, src_shape.h, 1, src_channels));
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = weights;
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  TensorDescriptor src_desc = {DataType::kFloat16, storage, Layout::kHWC};
  TensorDescriptor dst_desc = {DataType::kFloat16, storage, Layout::kHWC};
  return ConvWaveMatrixMaliF16ExternalWeightsTest(
      env, attr, weights, src_tensor, src_desc, dst_desc,
      /*different_weights_for_height=*/true);
}

}  // namespace ml_drift
