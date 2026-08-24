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

#include "ml_drift/common/kernels/tests//conv_wave_matrix_test_util.h"

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
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/conv_wave_matrix.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/kernels/winograd.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status ConvWaveMatrix1x1Test(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 32, 32, 256);
  const BHWC dst_shape(1, 32, 32, 256);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(dst_shape.c, 1, 1, src_tensor.shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.data.size(); ++i) {
    attr_weights.data[i] = std::sin(0.1f * i);
  }
  attr.bias.shape = Linear(dst_shape.c);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < attr.bias.data.size(); ++i) {
    attr.bias.data[i] = std::sin(0.1f * i);
  }

  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  const float eps = GetEpsilon(precision, env.GetGpuInfo(), attr);
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  TensorFloat32 dst_tensor_gpu;
  ConvWaveMatrix operation = CreateConvWaveMatrix(op_def, precision, dst_shape,
                                                  attr, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvWaveMatrix>(std::move(operation)),
      dst_shape, &dst_tensor_gpu));
  EXPECT_THAT(dst_tensor_gpu.data,
              Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 16, 8, 128);
  int dst_ch = 256;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 2);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(2, 1);
  attr.dilations = HW(3, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(dst_ch, 2, 3, src_tensor.shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.data.size(); ++i) {
    attr_weights.data[i] = sin(0.1f * i);
  }
  attr.bias.shape = Linear(dst_ch);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < attr.bias.data.size(); ++i) {
    attr.bias.data[i] = sin(0.1f * i);
  }

  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  const float eps = GetEpsilon(precision, env.GetGpuInfo(), attr);
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  TensorFloat32 dst_tensor_gpu;
  ConvWaveMatrix operation = CreateConvWaveMatrix(
      op_def, precision, dst_ref_tensor.shape, attr, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvWaveMatrix>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor_gpu));
  EXPECT_THAT(dst_tensor_gpu.data,
              Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrix1x1BatchTest(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(4, 8, 8, 128);
  const BHWC dst_shape(4, 8, 8, 256);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(dst_shape.c, 1, 1, src_tensor.shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.data.size(); ++i) {
    attr_weights.data[i] = sin(0.1f * i);
  }
  attr.bias.shape = Linear(dst_shape.c);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < attr.bias.data.size(); ++i) {
    attr.bias.data[i] = sin(0.1f * i);
  }

  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  const float eps = GetEpsilon(precision, env.GetGpuInfo(), attr);
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  TensorFloat32 dst_tensor_gpu;
  ConvWaveMatrix operation = CreateConvWaveMatrix(op_def, precision, dst_shape,
                                                  attr, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvWaveMatrix>(std::move(operation)),
      dst_shape, &dst_tensor_gpu));
  EXPECT_THAT(dst_tensor_gpu.data,
              Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrix1x1ExternalWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 16, 8, 64);
  int dst_ch = 256;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(dst_ch, 1, 1, src_tensor.shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.data.size(); ++i) {
    attr_weights.data[i] = sin(0.1f * i);
  }
  attr.bias.shape = Linear(dst_ch);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < attr.bias.data.size(); ++i) {
    attr.bias.data[i] = sin(0.1f * i);
  }

  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      env.GetGpuInfo(), DeduceDataTypeFromPrecision(precision), attr.bias);
  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_ref_tensor.shape, attr, env.GetGpuInfo(),
      &bias_tensor_desc,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/false);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr_weights,
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

  float eps = GetEpsilon(precision, env.GetGpuInfo(), attr) * 2.0f;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixExternalWeightsTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 16, 8, 128);
  int dst_ch = 256;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 2);
  attr.padding.appended = HW(2, 1);
  attr.strides = HW(2, 1);
  attr.dilations = HW(3, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(dst_ch, 2, 3, src_tensor.shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.data.size(); ++i) {
    attr_weights.data[i] = sin(0.1f * i);
  }
  attr.bias.shape = Linear(dst_ch);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < attr.bias.data.size(); ++i) {
    attr.bias.data[i] = sin(0.1f * i);
  }

  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      env.GetGpuInfo(), DeduceDataTypeFromPrecision(precision), attr.bias);
  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_ref_tensor.shape, attr, env.GetGpuInfo(),
      &bias_tensor_desc,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/false);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr_weights,
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
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixExternalBatchedWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 6, 12, 64);
  const BHWC dst_shape(1, 6, 12, 128);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape =
      OHWI(dst_shape.c, src_tensor.shape.h, 1, src_tensor.shape.c);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = attr_weights.shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = sin(0.1f * i);
  }

  TensorFloat32 dst_ref_tensor =
      FullyConnectedRefDifferentWeightsForHeight(weights, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_shape, attr, env.GetGpuInfo(), /*bias=*/nullptr,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true);

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

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, env.GetGpuInfo(), attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixPackedGroupsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage,
                                            const BHWC& src_shape,
                                            int dst_channels) {
  const int weights_batch_size = 16;
  const int num_groups_per_item = 4;

  Tensor<OHWI, DataType::FLOAT32> weights = MakeSyntheticTensor(
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

  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

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
      conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  conv_attr_weights.shape = weights.shape;

  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, conv_attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  auto convolution = CreateConvWaveMatrixExternalWeights(
      op_def, precision, dst_packed_shape, conv_attr, env.GetGpuInfo(),
      /*bias=*/nullptr,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true, runtime_check);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           convolution.GetWeightsDescription());

  TensorDescriptor src_desc = TensorDescriptor(data_type, storage, Layout::HWC);
  src_desc.UploadData(src_packed);

  std::vector<int32_t> runtime_params_cpu(weights_batch_size * 2, 0);
  for (int i = 0; i < weights_batch_size; ++i) {
    runtime_params_cpu[i] = groups_sizes.data[i];
    runtime_params_cpu[weights_batch_size + i] = groups_offsets.data[i];
  }
  TensorDescriptor runtime_params_td(DataType::INT32, TensorStorageType::BUFFER,
                                     Layout::LINEAR);
  runtime_params_td.SetBHWCShape(BHWC(1, 1, 1, weights_batch_size * 2));
  runtime_params_td.UploadData(runtime_params_cpu.data());

  TensorDescriptor dst_desc = TensorDescriptor(data_type, storage, Layout::HWC);
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
      std::make_unique<ConvWaveMatrix>(std::move(convolution))));

  TensorFloat32 dst_packed;
  dst_desc.DownloadData(&dst_packed);
  auto dst_tensor =
      RemapFromReference(dst_packed, packed_groups_map, num_groups_per_item);
  float eps = GetEpsilon(precision, env.GetGpuInfo()) * 2.0f * src_shape.c;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixExternalWfloatTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage,
                                              const BHWC& src_shape,
                                              int dst_channels,
                                              bool batched_weights) {
  TensorFloat32 src_tensor;
  const int src_channels = src_shape.c;
  src_tensor.shape = src_shape;
  const BHWC dst_shape(1, src_shape.h, src_shape.w, dst_channels);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  const int weights_batch_size = batched_weights ? src_shape.h : 1;
  auto weights_f32 = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, src_channels));
  weights_f32.data.resize(weights_f32.shape.DimensionsProduct() +
                          XNN_EXTRA_BYTES / sizeof(float));
  TensorFloat32 dst_ref_tensor =
      FullyConnectedRefDifferentWeightsForHeight(weights_f32, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  WeightsDescription weights_desc =
      GetFullyConnectedWeightsDesc(data_type, weights_f32.shape);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_f32.shape;

  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_shape, external_weights, env.GetGpuInfo(),
      /*bias=*/nullptr,
      /*src_exp=*/nullptr, batched_weights);

  TensorDescriptor weights_td =
      GetTensorDescriptorsForWeightsLayout(weights_f32, weights_desc)[0];

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps =
      GetEpsilon(precision, env.GetGpuInfo()) * weights_f32.shape.i * 4.0f;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_td}, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixExternalWi8Test(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape, int dst_channels,
    bool batched_weights, int group_size, int scale_zp_batch) {
  TensorFloat32 src_tensor;
  const int src_channels = src_shape.c;
  group_size = group_size != -1 ? group_size : src_channels;
  const int num_groups = src_channels / group_size;
  src_tensor.shape = src_shape;
  const BHWC dst_shape(1, src_shape.h, src_shape.w, dst_channels);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  const int weights_batch_size = batched_weights ? src_shape.h : 1;
  scale_zp_batch = scale_zp_batch != -1 ? scale_zp_batch : weights_batch_size;
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, weights_batch_size, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  auto weights_f32 = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, src_channels));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }
  auto weights_scales =
      MakeSyntheticTensor(OHWI(dst_channels, scale_zp_batch, 1, num_groups));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 128.0f;
  }
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scales.shape;
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(),
                                 0.0f);

  auto weights =
      MakeWeightsFromInt8(weights_i8, weights_scales, weights_zero_point);
  TensorFloat32 dst_ref_tensor =
      FullyConnectedRefDifferentWeightsForHeight(weights, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(env.GetGpuInfo(), weights_i8.shape);

  DataType type = conv_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToTensorDesc(env.GetGpuInfo(), weights_scales, type);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(env.GetGpuInfo(),
                                                weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;

  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_shape, external_weights, env.GetGpuInfo(),
      /*bias=*/nullptr,
      /*src_exp=*/nullptr, batched_weights);

  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps =
      GetEpsilon(precision, env.GetGpuInfo()) * weights_i8.shape.i * 8.0f;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td, &scale_desc, &zp_desc}, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixExternalWi4Test(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape, int dst_channels,
    bool batched_weights, int group_size) {
  TensorFloat32 src_tensor;
  const int src_channels = src_shape.c;
  group_size = group_size != -1 ? group_size : src_channels;
  const int num_groups = src_channels / group_size;
  src_tensor.shape = src_shape;
  const BHWC dst_shape(1, src_shape.h, src_shape.w, dst_channels);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  const int weights_batch_size = batched_weights ? src_shape.h : 1;
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, weights_batch_size, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, src_channels));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scales = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, num_groups));
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

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(env.GetGpuInfo(), weights_i4.shape);

  DataType type = conv_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToTensorDesc(env.GetGpuInfo(), weights_scales, type);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(env.GetGpuInfo(),
                                                weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;

  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_shape, external_weights, env.GetGpuInfo(),
      /*bias=*/nullptr,
      /*src_exp=*/nullptr, batched_weights);

  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps =
      GetEpsilon(precision, env.GetGpuInfo()) * weights_i4.shape.i * 4.0f;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i4_td, &scale_desc, &zp_desc}, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixExternalWi2Test(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape, int dst_channels,
    bool batched_weights, int group_size) {
  TensorFloat32 src_tensor;
  const int src_channels = src_shape.c;
  group_size = group_size != -1 ? group_size : src_channels;
  const int num_groups = src_channels / group_size;
  src_tensor.shape = src_shape;
  const BHWC dst_shape(1, src_shape.h, src_shape.w, dst_channels);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(0.01f * i);
  }

  const int weights_batch_size = batched_weights ? src_shape.h : 1;
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(dst_channels, weights_batch_size, 1, src_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  auto weights_f32 = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, src_channels));
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 4.0f;
    weights_i2.data[i] = std::max(std::min(val, 3), 0) - 2;
  }
  auto weights_scales = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, num_groups));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 2.0f;
  }
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scales.shape;
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(),
                                 0.0f);

  auto weights =
      MakeWeightsFromInt8(weights_i2, weights_scales, weights_zero_point);
  TensorFloat32 dst_ref_tensor =
      FullyConnectedRefDifferentWeightsForHeight(weights, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  WeightsDescription weights_desc =
      GetFullyConnectedInt2WeightsDesc(env.GetGpuInfo(), weights_i2.shape);

  DataType type = conv_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToTensorDesc(env.GetGpuInfo(), weights_scales, type);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(env.GetGpuInfo(),
                                                weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i2.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;

  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_shape, external_weights, env.GetGpuInfo(),
      /*bias=*/nullptr,
      /*src_exp=*/nullptr, batched_weights);

  TensorDescriptor weights_i2_td =
      GetTensorDescriptorForWeightsLayout(weights_i2, weights_desc);

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps =
      GetEpsilon(precision, env.GetGpuInfo()) * weights_i2.shape.i * 2.0f;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i2_td, &scale_desc, &zp_desc}, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixWinograd4x4To6x6Test(TestExecutionEnvironment& env,
                                                CalculationsPrecision precision,
                                                TensorStorageType storage) {
  const int src_channels = 32;
  const int dst_channels = 128;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(2, 2);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 3, 3, src_channels));
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>(
          std::move(synthetic_weights));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 17, 13, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  auto wino_up =
      std::make_unique<Winograd3x3TiledXForward>(CreateWinograd3x3TiledXForward(
          env.GetGpuInfo(), conv_def, attr.padding, 6));
  auto wino_down = std::make_unique<Winograd3x3TiledXBackward>(
      CreateWinograd3x3TiledXBackward(env.GetGpuInfo(), conv_def, attr.bias,
                                      6));

  const int tiles_x = DivideRoundUp(dst_ref_tensor.shape.w, 4);
  const int tiles_y = DivideRoundUp(dst_ref_tensor.shape.h, 4);
  TensorFloat32 wino_up_result;
  TensorFloat32 conv_result;
  wino_up_result.shape =
      BHWC(src_tensor.shape.b, 36, tiles_x * tiles_y, src_tensor.shape.c);
  conv_result.shape =
      BHWC(src_tensor.shape.b, 36, tiles_x * tiles_y, dst_ref_tensor.shape.c);

  auto convolution = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_ref_tensor.shape, attr, env.GetGpuInfo(),
      /*bias=*/nullptr,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true);

  Tensor<OHWI, DataType::FLOAT32> wino_weights;
  RearrangeWeightsToWinograd3x3TileNxN(attr_weights, &wino_weights, 6);
  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(wino_weights,
                                           convolution.GetWeightsDescription());

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(src_tensor, std::move(wino_up),
                                               BHWC(wino_up_result.shape),
                                               &wino_up_result));

  {
    TensorDescriptor src_td = conv_def.src_tensors[0];
    src_td.UploadData(wino_up_result);

    std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
    srcs_td[0] = &src_td;
    for (int i = 0; i < weights_gpu.size(); ++i) {
      srcs_td[1 + i] = &weights_gpu[i];
    }
    TensorDescriptor dst_td = conv_def.dst_tensors[0];
    dst_td.SetBHWCShape(BHWC(conv_result.shape));
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        srcs_td, {&dst_td},
        std::make_unique<ConvWaveMatrix>(std::move(convolution))));
    dst_td.DownloadData(&conv_result);
  }
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      conv_result, std::move(wino_down), dst_ref_tensor.shape, &dst_tensor));
  float eps = GetEpsilon(precision, env.GetGpuInfo(), attr) * 3.0f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixRuntimeChannelsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    Convolution2DAttributes& attr, const OperationDef& conv_def,
    CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  auto& attr_weights =
      std::get<ml_drift::Tensor<OHWI, DataType::FLOAT32>>(attr.weights);
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor =
      ConvolutionReference(attr, src_tensor, aligned_runtime_channels);

  auto operation = CreateConvWaveMatrixExternalWeights(
      conv_def, precision, dst_ref_tensor.shape, attr, exec_env.GetGpuInfo(),
      /*bias=*/nullptr,
      /*src_exp=*/nullptr,
      /*different_weights_for_height=*/true,
      runtime_channels.GenerateConvRuntimeCheckDesc());

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr_weights,
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
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
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

absl::Status ConvWaveMatrixRuntimeSrcEndChannelsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
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

  OperationDef conv_def;
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  for (int src_ch = 0; src_ch <= weights_shape.i; src_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.src_end_ch = src_ch;
    MLD_EXPECT_OK(ConvWaveMatrixRuntimeChannelsTest(env, src_tensor, attr, conv_def,
                                                precision, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixRuntimeDstEndChannelsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
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

  OperationDef conv_def;
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  if (!SupportsConvWaveMatrix(env.GetGpuInfo(), precision, attr)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  for (int dst_ch = 0; dst_ch <= weights_shape.o; dst_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.dst_end_ch = dst_ch;
    MLD_EXPECT_OK(ConvWaveMatrixRuntimeChannelsTest(env, src_tensor, attr, conv_def,
                                                precision, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixInt8Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i8);

  auto operation = CreateConvWaveMatrixInt8(op_def, dst_ref_tensor.shape,
                                            weights_i8, exec_env.GetGpuInfo());

  TensorDescriptor src_td = op_def.src_tensors[0];
  if (quantized_type == PackedType::kInt8W4C4) {
    src_td.SetBHWCShape(BHWC(src_tensor_i8.shape.b, src_tensor_i8.shape.h,
                             DivideRoundUp(src_tensor_i8.shape.w, 4),
                             src_tensor_i8.shape.c));
    TensorDescriptor src_desc_temp = {DataType::INT8, src_td.GetStorageType(),
                                      src_td.GetLayout()};
    src_desc_temp.UploadData(src_tensor_i8);
    src_td.UploadDataRaw(absl::MakeConstSpan(src_desc_temp.GetData()));
  } else if (quantized_type == PackedType::kInt8C16) {
    const BHWC packed_shape =
        BHWC(src_tensor_i8.shape.b, src_tensor_i8.shape.h,
             src_tensor_i8.shape.w, DivideRoundUp(src_tensor_i8.shape.c, 4));
    ml_drift::Tensor<BHWC, DataType::INT32> src_tensor_i32;
    src_tensor_i32.shape = packed_shape;
    src_tensor_i32.data.resize(packed_shape.DimensionsProduct());
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(src_tensor_i32.data.data());
    const uint8_t* src_ptr =
        reinterpret_cast<const uint8_t*>(src_tensor_i8.data.data());
    for (int b = 0; b < packed_shape.b; ++b) {
      for (int h = 0; h < packed_shape.h; ++h) {
        for (int w = 0; w < packed_shape.w; ++w) {
          for (int c = 0; c < packed_shape.c; ++c) {
            int c0 = std::min(c * 4, src_tensor_i8.shape.c - 1);
            int c1 = std::min(c * 4 + 1, src_tensor_i8.shape.c - 1);
            int c2 = std::min(c * 4 + 2, src_tensor_i8.shape.c - 1);
            int c3 = std::min(c * 4 + 3, src_tensor_i8.shape.c - 1);
            uint32_t v0 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c0})];
            uint32_t v1 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c1})];
            uint32_t v2 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c2})];
            uint32_t v3 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c3})];
            uint32_t value = (v3 << 24) | (v2 << 16) | (v1 << 8) | (v0);
            dst_ptr[packed_shape.LinearIndex({b, h, w, c})] = value;
          }
        }
      }
    }
    src_td.UploadData(src_tensor_i32);
  } else {
    src_td.UploadData(src_tensor_i8);
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {&src_td}, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixInt8Test(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage) {
  const int src_channels = 61;  // src_slices = 16, divisible by 8, for using
                                // with wave_matrix_k = 32(8 slices)(Intel)
  const int dst_channels = 77;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_f32.data.size(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  auto src_shape = BHWC(1, 17, 13, src_channels);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  const PackedType quantized_type = PackedType::kInt8C16;
  const DataType src_data_type = ToSpatialTensorType(quantized_type);
  const DataType dst_data_type = DataType::INT32;
  if (!env.IsStorageSupported(src_storage, src_data_type)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  if (!SupportsConvWaveMatrixInt8(env.GetGpuInfo(), weights_i8.shape)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
  MLD_EXPECT_OK(ConvWaveMatrixInt8Test(env, src_i8_tensor, quantized_type,
                                   weights_i8, op_def));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixInt8ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i8);

  auto operation = CreateConvWaveMatrixInt8ExternalWeights(
      exec_env.GetGpuInfo(), op_def, weights_i8.shape, dst_ref_tensor.shape);

  TensorDescriptor src_td = op_def.src_tensors[0];
  if (quantized_type == PackedType::kInt8W4C4) {
    src_td.SetBHWCShape(BHWC(src_tensor_i8.shape.b, src_tensor_i8.shape.h,
                             DivideRoundUp(src_tensor_i8.shape.w, 4),
                             src_tensor_i8.shape.c));
    TensorDescriptor src_desc_temp = {DataType::INT8, src_td.GetStorageType(),
                                      src_td.GetLayout()};
    src_desc_temp.UploadData(src_tensor_i8);
    src_td.UploadDataRaw(absl::MakeConstSpan(src_desc_temp.GetData()));
  } else if (quantized_type == PackedType::kInt8C16) {
    const BHWC packed_shape =
        BHWC(src_tensor_i8.shape.b, src_tensor_i8.shape.h,
             src_tensor_i8.shape.w, DivideRoundUp(src_tensor_i8.shape.c, 4));
    ml_drift::Tensor<BHWC, DataType::INT32> src_tensor_i32;
    src_tensor_i32.shape = packed_shape;
    src_tensor_i32.data.resize(packed_shape.DimensionsProduct());
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(src_tensor_i32.data.data());
    const uint8_t* src_ptr =
        reinterpret_cast<const uint8_t*>(src_tensor_i8.data.data());
    for (int b = 0; b < packed_shape.b; ++b) {
      for (int h = 0; h < packed_shape.h; ++h) {
        for (int w = 0; w < packed_shape.w; ++w) {
          for (int c = 0; c < packed_shape.c; ++c) {
            int c0 = std::min(c * 4, src_tensor_i8.shape.c - 1);
            int c1 = std::min(c * 4 + 1, src_tensor_i8.shape.c - 1);
            int c2 = std::min(c * 4 + 2, src_tensor_i8.shape.c - 1);
            int c3 = std::min(c * 4 + 3, src_tensor_i8.shape.c - 1);
            uint32_t v0 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c0})];
            uint32_t v1 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c1})];
            uint32_t v2 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c2})];
            uint32_t v3 =
                src_ptr[src_tensor_i8.shape.LinearIndex({b, h, w, c3})];
            uint32_t value = (v3 << 24) | (v2 << 16) | (v1 << 8) | (v0);
            dst_ptr[packed_shape.LinearIndex({b, h, w, c})] = value;
          }
        }
      }
    }
    src_td.UploadData(src_tensor_i32);
  } else {
    src_td.UploadData(src_tensor_i8);
  }

  TensorDescriptor weights_i8_td;
  {
    WeightsDescription weights_desc = operation.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape));
    RearrangeWeights(weights_i8, weights_desc, absl::MakeSpan(data));

    weights_i8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i8_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td}, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixInt8ExternalWeightsTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage) {
  const int src_channels = 61;  // src_slices = 16, divisible by 8, for using
                                // with wave_matrix_k = 32(8 slices)(Intel)
  const int dst_channels = 77;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_f32.data.size(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  auto src_shape = BHWC(1, 17, 13, src_channels);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  const PackedType quantized_type = PackedType::kInt8C16;
  const DataType src_data_type = ToSpatialTensorType(quantized_type);
  const DataType dst_data_type = DataType::INT32;
  if (!env.IsStorageSupported(src_storage, src_data_type)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  if (!SupportsConvWaveMatrixInt8(env.GetGpuInfo(), weights_i8.shape)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
  MLD_EXPECT_OK(ConvWaveMatrixInt8ExternalWeightsTest(
      env, src_i8_tensor, quantized_type, weights_i8, op_def));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixInt8WithSrcQuantizationTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>* weights_zero_point,
    const PackedType quantized_type,
    const TensorStorageType quantized_storage_type, const DataType float_type,
    const TensorStorageType float_storage_type) {
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = MakeZeroTensor(Linear(weights_i8.shape.o));
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  BHWC quantized_shape =
      GetShapeForPackedType(src_tensor.shape, quantized_type);
  TensorDescriptor src_params_td;
  TensorDescriptor src_quantized_td;
  const bool need_src_sum = weights_zero_point != nullptr;
  {
    OperationDef op_def;
    op_def.src_tensors.push_back({float_type, float_storage_type, Layout::HWC});
    op_def.dst_tensors.push_back({ToSpatialTensorType(quantized_type),
                                  quantized_storage_type, Layout::HWC});
    op_def.dst_tensors.push_back({float_type, float_storage_type, Layout::HWC});
    auto quantization_op =
        CreateQuantization(op_def, quantized_type, exec_env.GetGpuInfo(),
                           src_tensor.shape, need_src_sum);
    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(src_tensor);
    src_quantized_td = op_def.dst_tensors[0];
    src_quantized_td.SetBHWCShape(quantized_shape);
    src_params_td = op_def.dst_tensors[1];
    BHWC params_shape = src_tensor.shape;
    params_shape.c = need_src_sum ? 3 : 2;
    src_params_td.SetBHWCShape(params_shape);

    MLD_EXPECT_OK(exec_env.ExecuteGPUOperation({&src_td},
                                           {&src_quantized_td, &src_params_td},
                                           std::move(quantization_op)));
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

  OperationDef conv_def;
  conv_def.src_tensors.push_back({ToSpatialTensorType(quantized_type),
                                  quantized_storage_type, Layout::HWC});
  const DataType dst_conv_type = DataType::INT32;
  conv_def.dst_tensors.push_back(
      {dst_conv_type, float_storage_type, Layout::HWC});
  auto conv_op = CreateConvWaveMatrixInt8ExternalWeights(
      exec_env.GetGpuInfo(), conv_def, weights_i8.shape, dst_ref_tensor.shape);

  TensorDescriptor dequant_dst = {float_type, float_storage_type, Layout::HWC};
  auto dequant_op = CreateDequantization(
      weights_i8.shape, exec_env.GetGpuInfo(), conv_def.dst_tensors[0],
      dequant_dst, src_params_td, weights_sum_i_td, weights_scale_td,
      weights_zero_point_td_ptr);
  MLD_EXPECT_OK(conv_op.AddOperation(exec_env.GetGpuInfo(), &dequant_op));

  TensorDescriptor weights_i8_td;
  {
    WeightsDescription weights_desc = conv_op.GetWeightsDescription();
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
  std::vector<TensorDescriptor*> src_descs = {&src_quantized_td, &weights_i8_td,
                                              &src_params_td, &weights_sum_i_td,
                                              &weights_scale_td};
  if (weights_zero_point != nullptr) {
    src_descs.push_back(&weights_zero_point_td);
  }
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_descs, {&dst_td},
      std::make_unique<ConvWaveMatrix>(std::move(conv_op))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  int mads_count = weights_i8.shape.i;
  float eps = 0.01f * mads_count;
  if (weights_zero_point != nullptr) {
    eps *= 2.0f;
  }
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMatrixInt8WithSrcQuantizationTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage) {
  const int src_channels = 61;  // src_slices = 16, divisible by 8, for using
                                // with wave_matrix_k = 32(8 slices)(Intel)
  const int dst_channels = 77;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_f32.data.size(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }
  auto weights_scale = MakeSyntheticTensor(Linear(dst_channels));
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] /= 127.0f;
  }
  auto weights_zp = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 17, 13, src_channels);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  const PackedType quantized_type = PackedType::kInt8C16;
  for (const DataType float_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    if (!env.IsStorageSupported(float_storage, float_type)) {
      return absl::UnimplementedError(env.SkipTestMessage());
    }
    if (!SupportsConvWaveMatrixInt8(env.GetGpuInfo(), weights_i8.shape)) {
      return absl::UnimplementedError(env.SkipTestMessage());
    }
    MLD_EXPECT_OK(ConvWaveMatrixInt8WithSrcQuantizationTest(
        env, src_tensor, weights_i8, weights_scale,
        /*weights_zero_point=*/nullptr, quantized_type, quantized_storage,
        float_type, float_storage));
    MLD_EXPECT_OK(ConvWaveMatrixInt8WithSrcQuantizationTest(
        env, src_tensor, weights_i8, weights_scale, &weights_zp, quantized_type,
        quantized_storage, float_type, float_storage));
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
