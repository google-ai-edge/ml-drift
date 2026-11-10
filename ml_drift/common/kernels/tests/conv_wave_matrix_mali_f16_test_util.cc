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

#include <algorithm>
#include <cstdint>
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
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

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
      /*src_exp=*/nullptr, different_weights_for_height);
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

absl::Status ConvWaveMatrixMaliF16PackedGroupsTest(
    TestExecutionEnvironment& env, TensorStorageType storage) {
  const BHWC src_shape(1, 1, 383, 128);
  const int dst_channels = 64;
  const int weights_batch_size = 16;
  const int num_groups_per_item = 4;

  Tensor<OHWI, DataType::kFloat32> weights = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, src_shape.c));
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorInt32 group_ids = GenerateGroupIds(
      BHWC(1, 1, src_shape.w, num_groups_per_item), weights_batch_size);
  TensorFloat32 dst_ref = ConvolutionWithIds(src_tensor, weights, group_ids);

  auto [groups_map, groups_sizes] =
      GroupsMapReference(group_ids, weights_batch_size);
  auto [packed_groups_map, groups_offsets] =
      PackedGroupsMapReference(groups_map, groups_sizes);

  auto src_packed = RemapToReference(src_tensor, packed_groups_map);

  auto dst_packed_shape = src_packed.shape;
  dst_packed_shape.c = dst_channels;

  TensorDescriptor src_desc = {DataType::kFloat16, storage, Layout::kHWC};
  TensorDescriptor dst_desc = {DataType::kFloat16, storage, Layout::kHWC};

  ConvRuntimeCheckDesc::PackedGroups packed_groups;
  packed_groups.params_offset = 0;
  packed_groups.num_groups = weights_batch_size;
  packed_groups.max_group_size = src_shape.w;
  ConvRuntimeCheckDesc runtime_check;
  runtime_check.packed_groups = packed_groups;

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& conv_attr_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  conv_attr_weights.shape = weights.shape;

  auto convolution = CreateConvWaveMatrixMaliF16ExternalWeights(
      env.GetGpuInfo(), src_desc, dst_desc, conv_attr, /*bias=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
      runtime_check);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           convolution.GetWeightsDescription());

  src_desc.UploadData(src_packed);

  std::vector<int32_t> runtime_params_cpu(weights_batch_size * 2, 0);
  for (int i = 0; i < weights_batch_size; ++i) {
    runtime_params_cpu[i] = groups_sizes.data[i];
    runtime_params_cpu[weights_batch_size + i] = groups_offsets.data[i];
  }
  TensorDescriptor runtime_params_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  runtime_params_td.SetBHWCShape(BHWC(1, 1, 1, weights_batch_size * 2));
  runtime_params_td.UploadData(runtime_params_cpu.data());

  dst_desc.SetBHWCShape(dst_packed_shape);
  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 2);
  int idx = 0;
  srcs_td[idx++] = &src_desc;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[idx++] = &weights_gpu[i];
  }
  srcs_td[idx++] = &runtime_params_td;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {srcs_td}, {&dst_desc},
      std::make_unique<ConvWaveMatrixMaliF16>(std::move(convolution))));

  TensorFloat32 dst_packed;
  dst_desc.DownloadData(&dst_packed);
  auto dst_tensor =
      RemapFromReference(dst_packed, packed_groups_map, num_groups_per_item);
  float eps = GetEpsilon(CalculationsPrecision::kF16, env.GetGpuInfo()) * 2.0f *
              src_shape.c;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

namespace {
absl::Status ConvWaveMatrixMaliF16RuntimeChannelsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    Convolution2DAttributes& attr, TensorStorageType storage,
    const TestingRuntimeChannels& runtime_channels) {
  auto& attr_weights = GetFloatWeights(attr);
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor =
      ConvolutionReference(attr, src_tensor, aligned_runtime_channels);

  TensorDescriptor src_desc = {DataType::kFloat16, storage, Layout::kHWC};
  TensorDescriptor dst_desc = {DataType::kFloat16, storage, Layout::kHWC};

  auto operation = CreateConvWaveMatrixMaliF16ExternalWeights(
      exec_env.GetGpuInfo(), src_desc, dst_desc, attr, /*bias=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
      runtime_channels.GenerateConvRuntimeCheckDesc());

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr_weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = src_desc;
  src_td.UploadData(src_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor params_td = {DataType::kInt32, TensorStorageType::kBuffer,
                                Layout::kHWC};
  TensorInt32 params_tensor = runtime_channels.GenerateTensorInt32();
  if (std::any_of(params_tensor.data.begin(), params_tensor.data.end(),
                  [](int x) { return x != -1; })) {
    params_td.UploadData(params_tensor);
    srcs_td.push_back(&params_td);
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

absl::Status ConvWaveMatrixMaliF16RuntimeSrcEndChannelsTest(
    TestExecutionEnvironment& env, TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 32, 128);
  auto weights_shape = OHWI(256, 1, 1, 128);
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(weights_shape);
  attr.bias = MakeZeroTensor(Linear(weights_shape.o));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  for (int src_ch = 4; src_ch <= weights_shape.i; src_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.src_end_ch = src_ch;
    ABSL_EXPECT_OK(ConvWaveMatrixMaliF16RuntimeChannelsTest(
        env, src_tensor, attr, storage, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixMaliF16RuntimeDstEndChannelsTest(
    TestExecutionEnvironment& env, TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 32, 128);
  auto weights_shape = OHWI(256, 1, 1, 128);
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(weights_shape);
  attr.bias = MakeZeroTensor(Linear(weights_shape.o));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  for (int dst_ch = 0; dst_ch <= weights_shape.o; dst_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.dst_end_ch = dst_ch;
    ABSL_EXPECT_OK(ConvWaveMatrixMaliF16RuntimeChannelsTest(
        env, src_tensor, attr, storage, runtime_channels));
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
