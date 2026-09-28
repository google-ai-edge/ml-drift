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

#include "ml_drift/common/kernels/tests/fully_connected_oi_test_util.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/fully_connected_oi.h"
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
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

absl::Status FullyConnectedOITest(TestExecutionEnvironment* exec_env,
                                  const FullyConnectedAttributes& attr,
                                  const TensorFloat32& src_tensor,
                                  const OperationDef& op_def,
                                  WeightsLayout weights_layout,
                                  CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  ExternalWeights external_weights;
  external_weights.desc.type = DeduceDataTypeFromPrecision(precision);
  external_weights.desc.layout = weights_layout;
  external_weights.desc.output_group_size = 1;
  external_weights.shape = attr.weights.shape;

  auto operation = CreateFullyConnectedOI(
      exec_env->GetGpuInfo(), precision, op_def.src_tensors[0],
      op_def.dst_tensors[0], external_weights, /*bias=*/nullptr,
      &dst_ref_tensor.shape);

  auto weights_descs =
      GetTensorDescriptorsForWeightsLayout(attr.weights, external_weights.desc);

  TensorDescriptor src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  TensorDescriptor dst_desc = op_def.dst_tensors[0];
  dst_desc.SetBHWCShape(dst_ref_tensor.shape);

  ABSL_RETURN_IF_ERROR(exec_env->ExecuteGPUOperation(
      {&src_desc, &weights_descs[0]}, {&dst_desc},
      std::make_unique<FullyConnectedOI>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  const float eps = GetEpsilon(precision, exec_env->GetGpuInfo(), attr);
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIO4Int8Test(
    TestExecutionEnvironment* exec_env, const TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::kInt8>& weights_i8,
    const ml_drift::Tensor<OHWI, DataType::kFloat32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::kFloat32>& biases,
    const OperationDef& op_def, WeightsLayout weights_layout,
    CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);

  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToTensorDesc(exec_env->GetGpuInfo(), weights_scale, type);
  auto zero_point_desc = ScaleOrZeroPointToTensorDesc(exec_env->GetGpuInfo(),
                                                      weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc.type = DataType::kUint8;
  external_weights.desc.layout = weights_layout;
  external_weights.desc.output_group_size = 1;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zero_point_desc;

  auto operation = CreateFullyConnectedOI(
      exec_env->GetGpuInfo(), precision, op_def.src_tensors[0],
      op_def.dst_tensors[0], external_weights);

  TensorDescriptor weights_desc =
      GetTensorDescriptorForWeightsLayout(weights_i8, external_weights.desc);

  TensorDescriptor src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  TensorDescriptor dst_desc = op_def.dst_tensors[0];
  dst_desc.SetBHWCShape(dst_ref_tensor.shape);

  ABSL_RETURN_IF_ERROR(exec_env->ExecuteGPUOperation(
      {&src_desc, &weights_desc, &scale_desc, &zero_point_desc}, {&dst_desc},
      std::make_unique<FullyConnectedOI>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  const float eps = GetEpsilon(precision, exec_env->GetGpuInfo(), attr);
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIO4Int4Test(
    TestExecutionEnvironment* exec_env, const TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::kInt8>& weights_i4,
    const ml_drift::Tensor<OHWI, DataType::kFloat32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::kFloat32>& biases,
    const OperationDef& op_def, CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);

  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i4, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToTensorDesc(exec_env->GetGpuInfo(), weights_scale, type);
  auto zero_point_desc = ScaleOrZeroPointToTensorDesc(exec_env->GetGpuInfo(),
                                                      weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc.type = DataType::kUint8;
  external_weights.desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  external_weights.desc.output_group_size = 1;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zero_point_desc;

  auto operation = CreateFullyConnectedOI(
      exec_env->GetGpuInfo(), precision, op_def.src_tensors[0],
      op_def.dst_tensors[0], external_weights);

  TensorDescriptor weights_desc =
      GetTensorDescriptorForWeightsLayout(weights_i4, external_weights.desc);

  TensorDescriptor src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  TensorDescriptor dst_desc = op_def.dst_tensors[0];
  dst_desc.SetBHWCShape(dst_ref_tensor.shape);

  ABSL_RETURN_IF_ERROR(exec_env->ExecuteGPUOperation(
      {&src_desc, &weights_desc, &scale_desc, &zero_point_desc}, {&dst_desc},
      std::make_unique<FullyConnectedOI>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  const float eps = GetEpsilon(precision, exec_env->GetGpuInfo(), attr) * 4.0f;
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIO4Int4Sparse2x4Test(
    TestExecutionEnvironment* exec_env, const TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::kInt8>& weights_i4,
    const ml_drift::Tensor<OHWI, DataType::kUint8>& weights_indices,
    const OperationDef& op_def, CalculationsPrecision precision) {
  auto weights_scales = MakeSyntheticTensor(OHWI(weights_i4.shape.o, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 8.0f;
  }

  TensorFloat32 dst_ref_tensor;
  dst_ref_tensor.shape = src_tensor.shape;
  dst_ref_tensor.shape.c = weights_i4.shape.o;
  dst_ref_tensor.data.resize(dst_ref_tensor.shape.DimensionsProduct(), 0.0f);
  for (int y = 0; y < src_tensor.shape.h; ++y) {
    for (int x = 0; x < src_tensor.shape.w; ++x) {
      for (int dst_ch = 0; dst_ch < weights_i4.shape.o; ++dst_ch) {
        float dst_val = 0.0f;
        float scale =
            weights_scales
                .data[weights_scales.shape.LinearIndex({dst_ch, 0, 0, 0})];
        for (int src_t0 = 0; src_t0 < src_tensor.shape.c / 4; ++src_t0) {
          float src_vals[4];
          src_vals[0] = src_tensor.data[src_tensor.shape.LinearIndex(
              {0, y, x, src_t0 * 4 + 0})];
          src_vals[1] = src_tensor.data[src_tensor.shape.LinearIndex(
              {0, y, x, src_t0 * 4 + 1})];
          src_vals[2] = src_tensor.data[src_tensor.shape.LinearIndex(
              {0, y, x, src_t0 * 4 + 2})];
          src_vals[3] = src_tensor.data[src_tensor.shape.LinearIndex(
              {0, y, x, src_t0 * 4 + 3})];
          int weights_v0_i32 = weights_i4.data[weights_i4.shape.LinearIndex(
              {dst_ch, 0, 0, src_t0 * 2 + 0})];
          int weights_v1_i32 = weights_i4.data[weights_i4.shape.LinearIndex(
              {dst_ch, 0, 0, src_t0 * 2 + 1})];
          int ind0 = weights_indices.data[weights_indices.shape.LinearIndex(
              {dst_ch, 0, 0, src_t0 * 2 + 0})];
          int ind1 = weights_indices.data[weights_indices.shape.LinearIndex(
              {dst_ch, 0, 0, src_t0 * 2 + 1})];
          dst_val += src_vals[ind0] * weights_v0_i32 * scale +
                     src_vals[ind1] * weights_v1_i32 * scale;
        }
        dst_ref_tensor
            .data[dst_ref_tensor.shape.LinearIndex({0, y, x, dst_ch})] =
            dst_val;
      }
    }
  }
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zero_point;
  weights_zero_point.shape = weights_scales.shape;
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(),
                                 0.0f);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToTensorDesc(exec_env->GetGpuInfo(),
                                                 weights_scales, type);
  auto zero_point_desc = ScaleOrZeroPointToTensorDesc(exec_env->GetGpuInfo(),
                                                      weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc.type = DataType::kUint4;
  external_weights.desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  external_weights.desc.output_group_size = 1;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zero_point_desc;

  const int elements_count =
      GetTotalElementsCountForLayout(external_weights.desc, weights_i4.shape);
  TensorDescriptor weights_desc;
  {
    std::vector<uint8_t> weights_data(elements_count / 2);
    RearrangeWeightsInt8AsUint4(weights_i4, external_weights.desc,
                                absl::MakeSpan(weights_data),
                                /*shift_value=*/8, /*pad_value=*/8u);

    weights_desc = TensorDescriptor(
        DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
    weights_desc.SetBHWCShape(BHWC(1, 1, 1, weights_data.size()));
    weights_desc.UploadDataRaw(absl::MakeConstSpan(weights_data));
  }

  TensorDescriptor weights_indices_desc;
  {
    WeightsDescription indices_desc = external_weights.desc;
    indices_desc.type = DataType::kUint2;
    std::vector<uint8_t> weights_indices_data(elements_count / 4);
    RearrangeWeightsUint2(weights_indices, indices_desc,
                          absl::MakeSpan(weights_indices_data));

    weights_indices_desc = TensorDescriptor(
        DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
    weights_indices_desc.SetBHWCShape(
        BHWC(1, 1, 1, weights_indices_data.size()));
    weights_indices_desc.UploadDataRaw(
        absl::MakeConstSpan(weights_indices_data));
  }

  auto operation = CreateFullyConnectedOIInt4Sparse2x4(
      exec_env->GetGpuInfo(), precision, op_def.src_tensors[0],
      op_def.dst_tensors[0], external_weights);

  TensorDescriptor src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  TensorDescriptor dst_desc = op_def.dst_tensors[0];
  dst_desc.SetBHWCShape(dst_ref_tensor.shape);
  ABSL_RETURN_IF_ERROR(exec_env->ExecuteGPUOperation(
      {&src_desc, &weights_desc, &weights_indices_desc, &scale_desc,
       &zero_point_desc},
      {&dst_desc}, std::make_unique<FullyConnectedOI>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  const float eps =
      GetEpsilon(precision, exec_env->GetGpuInfo()) * weights_i4.shape.i * 4.0f;
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedOITest(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage, bool isI4O4) {
  const int src_channels = 256;
  const int dst_channels = 256;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 2, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  DataType data_type = DeduceDataTypeFromPrecision(precision);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  WeightsLayout weights_layout = isI4O4 ? WeightsLayout::kOSpatialIOGroupI4O4
                                        : WeightsLayout::kOSpatialIOGroupO4I4;
  ABSL_EXPECT_OK(FullyConnectedOITest(&env, attr, src_tensor, op_def, weights_layout,
                                 precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIWi8(TestExecutionEnvironment& env,
                                 CalculationsPrecision precision,
                                 TensorStorageType storage, bool isI4O4) {
  const int src_channels = 256;
  const int dst_channels = 256;
  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 128.0f;
  }
  auto biases = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  DataType data_type = DeduceDataTypeFromPrecision(precision);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  WeightsLayout weights_layout = isI4O4 ? WeightsLayout::kOSpatialIOGroupI4O4
                                        : WeightsLayout::kOSpatialIOGroupO4I4;
  ABSL_EXPECT_OK(FullyConnectedOIO4Int8Test(&env, src_tensor, weights_i8,
                                       weights_scales, biases, op_def,
                                       weights_layout, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIWi4(TestExecutionEnvironment& env,
                                 CalculationsPrecision precision,
                                 TensorStorageType storage) {
  const int src_channels = 256;
  const int dst_channels = 256;
  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 8.0f;
  }
  auto biases = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  DataType data_type = DeduceDataTypeFromPrecision(precision);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  ABSL_EXPECT_OK(FullyConnectedOIO4Int4Test(
      &env, src_tensor, weights_i4, weights_scales, biases, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIWi4Sparse2x4(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  const int src_channels = 256;
  const int dst_channels = 256;
  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_channels / 2);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels / 2));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  ml_drift::Tensor<OHWI, DataType::kUint8> weights_indices;
  weights_indices.shape = OHWI(dst_channels, 1, 1, src_channels / 2);
  weights_indices.data.resize(weights_indices.shape.DimensionsProduct() +
                              XNN_EXTRA_BYTES / sizeof(uint8_t));
  const std::vector<int2> possible_positions = {{0, 1}, {0, 2}, {0, 3},
                                                {1, 2}, {1, 3}, {2, 3}};
  for (int i = 0; i < weights_indices.shape.DimensionsProduct() / 2; ++i) {
    const int2 positions = possible_positions[(i * 13 + 7) % 6];
    weights_indices.data[i * 2] = positions.x;
    weights_indices.data[i * 2 + 1] = positions.y;
  }

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  DataType data_type = DeduceDataTypeFromPrecision(precision);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  ABSL_EXPECT_OK(FullyConnectedOIO4Int4Sparse2x4Test(
      &env, src_tensor, weights_i4, weights_indices, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIRuntimeChannelsTest(
    TestExecutionEnvironment* exec_env, const FullyConnectedAttributes& attr,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor =
      FullyConnectedReference(attr, src_tensor, aligned_runtime_channels);

  ExternalWeights external_weights;
  external_weights.desc.type = DeduceDataTypeFromPrecision(precision);
  external_weights.desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  external_weights.desc.output_group_size = 1;
  external_weights.shape = attr.weights.shape;

  auto operation = CreateFullyConnectedOI(
      exec_env->GetGpuInfo(), precision, op_def.src_tensors[0],
      op_def.dst_tensors[0], external_weights, /*bias=*/nullptr,
      &dst_ref_tensor.shape, /*src_exp=*/nullptr,
      runtime_channels.GenerateConvRuntimeCheckDesc());

  auto weights_descs =
      GetTensorDescriptorsForWeightsLayout(attr.weights, external_weights.desc);

  TensorDescriptor src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  TensorDescriptor dst_desc = op_def.dst_tensors[0];
  dst_desc.SetBHWCShape(dst_ref_tensor.shape);

  std::vector<TensorDescriptor*> src_cpu = {&src_desc, &weights_descs[0]};

  TensorDescriptor params_td = {DataType::kInt32, TensorStorageType::kBuffer,
                                Layout::kHWC};
  TensorInt32 params_tensor = runtime_channels.GenerateTensorInt32();
  if (std::any_of(params_tensor.data.begin(), params_tensor.data.end(),
                  [](int x) { return x != -1; })) {
    params_td.UploadData(params_tensor);
    src_cpu.push_back(&params_td);
  }

  ABSL_RETURN_IF_ERROR(exec_env->ExecuteGPUOperation(
      src_cpu, {&dst_desc},
      std::make_unique<FullyConnectedOI>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);

  const float eps = GetEpsilon(precision, exec_env->GetGpuInfo(), attr);
  const int dst_num_ch = dst_ref_tensor.shape.c;
  const int dst_end_ch = aligned_runtime_channels.dst_end_ch.has_value()
                             ? *aligned_runtime_channels.dst_end_ch
                             : dst_num_ch;
  for (int offset = 0; offset < dst_tensor.data.size(); offset += dst_num_ch) {
    std::vector<float> actual_data(
        dst_tensor.data.begin() + offset,
        dst_tensor.data.begin() + offset + dst_end_ch);
    std::vector<float> expected_data(
        dst_ref_tensor.data.begin() + offset,
        dst_ref_tensor.data.begin() + offset + dst_end_ch);
    EXPECT_THAT(actual_data,
                testing::Pointwise(testing::FloatNear(eps), expected_data));
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedOIRuntimeSrcTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 256;
  const int dst_channels = 128;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 2, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  DataType data_type = DeduceDataTypeFromPrecision(precision);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});

  for (int src_ch = 16; src_ch <= src_channels; src_ch += 16) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.src_end_ch = src_ch;
    ABSL_EXPECT_OK(FullyConnectedOIRuntimeChannelsTest(
        &env, attr, src_tensor, op_def, precision, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedOIRuntimeDstTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 128;
  const int dst_channels = 256;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 2, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  DataType data_type = DeduceDataTypeFromPrecision(precision);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});

  for (int dst_ch = 16; dst_ch <= dst_channels; dst_ch += 16) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.dst_end_ch = dst_ch;
    ABSL_EXPECT_OK(FullyConnectedOIRuntimeChannelsTest(
        &env, attr, src_tensor, op_def, precision, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedOIPackedGroupsTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage) {
  const int src_channels = 64;
  const int width = 64;
  const int dst_channels = 32;
  const int weights_batch_size = 16;
  const int num_groups_per_item = 4;

  Tensor<OHWI, DataType::kFloat32> weights = MakeSyntheticTensor(
      OHWI(dst_channels, weights_batch_size, 1, src_channels));
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));

  TensorFloat32 src_tensor =
      MakeSyntheticTensor(BHWC(1, 1, width, src_channels));
  TensorInt32 group_ids = GenerateGroupIds(
      BHWC(1, 1, width, num_groups_per_item), weights_batch_size);
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
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});

  ConvRuntimeCheckDesc::PackedGroups packed_groups;
  packed_groups.params_offset = 0;
  packed_groups.num_groups = weights_batch_size;
  packed_groups.max_group_size = width;
  ConvRuntimeCheckDesc runtime_check;
  runtime_check.packed_groups = packed_groups;

  TensorDescriptor src_desc =
      TensorDescriptor(data_type, storage, Layout::kHWC);
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

  ExternalWeights external_weights;
  external_weights.desc.type = DeduceDataTypeFromPrecision(precision);
  {
    WeightsDescription& desc = external_weights.desc;
    desc.layout = WeightsLayout::kCustomGroups;
    desc.group_sizes.push_back({Axis::kInputChannels, 4});
    desc.group_sizes.push_back({Axis::kOutputChannels, 4});
    desc.group_sizes.push_back({Axis::kInputChannels, 0});
    desc.group_sizes.push_back({Axis::kOutputChannels, 0});
    desc.group_sizes.push_back({Axis::kHeight, 0});
  }
  external_weights.desc.output_group_size = 1;
  external_weights.shape = weights.shape;

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights, external_weights.desc);

  auto conv_fc = CreateFullyConnectedOI(
      env.GetGpuInfo(), precision, op_def.src_tensors[0], op_def.dst_tensors[0],
      external_weights,
      /*bias=*/nullptr, &dst_packed_shape, /*src_exp=*/nullptr, runtime_check);

  TensorDescriptor dst_desc =
      TensorDescriptor(data_type, storage, Layout::kHWC);
  dst_desc.SetBHWCShape(dst_packed_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc, &weights_gpu[0], &runtime_params_td}, {&dst_desc},
      std::make_unique<FullyConnectedOI>(std::move(conv_fc))));

  TensorFloat32 dst_packed;
  dst_desc.DownloadData(&dst_packed);
  auto dst_tensor =
      RemapFromReference(dst_packed, packed_groups_map, num_groups_per_item);
  float eps = GetEpsilon(precision, env.GetGpuInfo()) * 2.0f * src_channels;
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedOIRingedOTest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 72;
  const int dst_channels = 20;
  const int weights_o_size = 56;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(weights_o_size, 1, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src = MakeSyntheticTensor(src_shape);

  auto weights_layout = WeightsLayout::kOSpatialIOGroupO4I4;
  for (int ring_offset : {0, 17, 20, 33, 49}) {
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});

    const BHWC dst_shape =
        BHWC(src_shape.b, src_shape.h, src_shape.w, dst_channels);
    TensorFloat32 output_ref = MakeZeroTensor(dst_shape);
    for (int b = 0; b < src.shape.b; ++b) {
      for (int y = 0; y < src.shape.h; ++y) {
        for (int x = 0; x < src.shape.w; ++x) {
          for (int o = 0; o < dst_channels; o++) {
            float sum = attr.bias.data.empty() ? 0 : attr.bias.data[o];
            for (int i = 0; i < src_channels; i++) {
              const int src_index = src.shape.LinearIndex({b, y, x, i});
              const int w_o = (o + ring_offset) % weights_o_size;
              const int f_index =
                  attr.weights.shape.LinearIndex({w_o, 0, 0, i});
              sum += src.data[src_index] * attr.weights.data[f_index];
            }
            const int dst_index = output_ref.shape.LinearIndex({b, y, x, o});
            output_ref.data[dst_index] = sum;
          }
        }
      }
    }

    TensorDescriptor bias_td = CreateConstantLinearTensorDescriptor(
        env.GetGpuInfo(), data_type, attr.bias);

    WeightsDescription weights_desc;
    weights_desc.type = data_type;
    weights_desc.layout = weights_layout;
    weights_desc.output_group_size = 1;

    std::vector<TensorDescriptor> weights_gpu =
        GetTensorDescriptorsForWeightsLayout(attr.weights, weights_desc);

    ConvRuntimeCheckDesc runtime_check;
    runtime_check.ring_o_offset_index = 0;
    runtime_check.ring_size = weights_o_size;

    ExternalWeights external_weights;
    external_weights.desc = weights_desc;
    external_weights.shape = attr.weights.shape;
    auto operation = CreateFullyConnectedOI(
        env.GetGpuInfo(), precision, op_def.src_tensors[0],
        op_def.dst_tensors[0], external_weights, &bias_td, &dst_shape,
        /*src_exp=*/nullptr, runtime_check);

    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(src);

    TensorDescriptor dst_td = op_def.dst_tensors[0];
    dst_td.SetBHWCShape(dst_shape);

    std::vector<TensorDescriptor*> src_cpu;
    src_cpu.push_back(&src_td);
    for (int i = 0; i < weights_gpu.size(); ++i) {
      src_cpu.push_back(&weights_gpu[i]);
    }
    src_cpu.push_back(&bias_td);

    TensorInt32 params;
    params.shape = BHWC(1, 1, 1, 1);
    params.data = std::vector<int32_t>(1, ring_offset);
    TensorDescriptor params_td = {DataType::kInt32, TensorStorageType::kBuffer,
                                  Layout::kHWC};
    params_td.UploadData(params);
    src_cpu.push_back(&params_td);

    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        src_cpu, {&dst_td},
        std::make_unique<FullyConnectedOI>(std::move(operation))));
    TensorFloat32 dst_tensor;
    dst_td.DownloadData(&dst_tensor);

    const float eps = GetEpsilon(precision, env.GetGpuInfo(), attr) * 2.0f;
    EXPECT_THAT(dst_tensor.data,
                testing::Pointwise(testing::FloatNear(eps), output_ref.data));
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedOIRingedITest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 20;
  const int dst_channels = 72;
  const int weights_i_size = 56;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, weights_i_size));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src = MakeSyntheticTensor(src_shape);

  auto weights_layout = WeightsLayout::kOSpatialIOGroupI4O4;
  for (int ring_offset : {0, 17, 20, 33, 49}) {
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});

    const BHWC dst_shape =
        BHWC(src_shape.b, src_shape.h, src_shape.w, dst_channels);
    TensorFloat32 output_ref = MakeZeroTensor(dst_shape);
    for (int b = 0; b < src.shape.b; ++b) {
      for (int y = 0; y < src.shape.h; ++y) {
        for (int x = 0; x < src.shape.w; ++x) {
          for (int o = 0; o < dst_channels; o++) {
            float sum = attr.bias.data.empty() ? 0 : attr.bias.data[o];
            for (int i = 0; i < src_channels; i++) {
              const int src_index = src.shape.LinearIndex({b, y, x, i});
              const int w_i = (i + ring_offset) % weights_i_size;
              const int f_index =
                  attr.weights.shape.LinearIndex({o, 0, 0, w_i});
              sum += src.data[src_index] * attr.weights.data[f_index];
            }
            const int dst_index = output_ref.shape.LinearIndex({b, y, x, o});
            output_ref.data[dst_index] = sum;
          }
        }
      }
    }

    TensorDescriptor bias_td = CreateConstantLinearTensorDescriptor(
        env.GetGpuInfo(), data_type, attr.bias);

    WeightsDescription weights_desc;
    weights_desc.type = data_type;
    weights_desc.layout = weights_layout;
    weights_desc.output_group_size = 1;

    std::vector<TensorDescriptor> weights_gpu =
        GetTensorDescriptorsForWeightsLayout(attr.weights, weights_desc);

    ConvRuntimeCheckDesc runtime_check;
    runtime_check.ring_i_offset_index = 0;
    runtime_check.ring_size = weights_i_size;

    ExternalWeights external_weights;
    external_weights.desc = weights_desc;
    external_weights.shape = attr.weights.shape;
    auto operation = CreateFullyConnectedOI(
        env.GetGpuInfo(), precision, op_def.src_tensors[0],
        op_def.dst_tensors[0], external_weights, &bias_td, &dst_shape,
        /*src_exp=*/nullptr, runtime_check);

    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(src);

    TensorDescriptor dst_td = op_def.dst_tensors[0];
    dst_td.SetBHWCShape(dst_shape);

    std::vector<TensorDescriptor*> src_cpu;
    src_cpu.push_back(&src_td);
    for (int i = 0; i < weights_gpu.size(); ++i) {
      src_cpu.push_back(&weights_gpu[i]);
    }
    src_cpu.push_back(&bias_td);

    TensorInt32 params;
    params.shape = BHWC(1, 1, 1, 1);
    params.data = std::vector<int32_t>(1, ring_offset);
    TensorDescriptor params_td = {DataType::kInt32, TensorStorageType::kBuffer,
                                  Layout::kHWC};
    params_td.UploadData(params);
    src_cpu.push_back(&params_td);

    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        src_cpu, {&dst_td},
        std::make_unique<FullyConnectedOI>(std::move(operation))));
    TensorFloat32 dst_tensor;
    dst_td.DownloadData(&dst_tensor);

    const float eps = GetEpsilon(precision, env.GetGpuInfo(), attr) * 2.0f;
    EXPECT_THAT(dst_tensor.data,
                testing::Pointwise(testing::FloatNear(eps), output_ref.data));
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
