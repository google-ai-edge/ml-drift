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

#include "ml_drift/common/kernels/tests/fully_connected_test_util.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/fully_connected.h"
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
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status FullyConnectedTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 4);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr.weights.shape.DimensionsProduct(); ++i) {
    attr.weights.data[i] = i;
  }
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  FullyConnected operation =
      CreateFullyConnected(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
      BHWC(1, 1, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {14.5f, 37.5f}));
  return absl::OkStatus();
}

absl::Status FullyConnectedLargeTest(TestExecutionEnvironment& env,
                                     CalculationsPrecision precision,
                                     TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 8);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(12, 1, 1, 8);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr.weights.shape.DimensionsProduct(); ++i) {
    attr.weights.data[i] = i;
  }
  attr.bias.shape = Linear(12);
  attr.bias.data = {-0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f,
                    0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f};

  const float eps = precision == CalculationsPrecision::F32 ? 0.0f : 1.0f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  FullyConnected operation =
      CreateFullyConnected(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
      BHWC(1, 1, 1, 12), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {139.4f, 363.5f, 587.6f, 811.7f,
                                         1035.8f, 1259.9f, 1484.1f, 1708.2f,
                                         1932.3f, 2156.4f, 2380.5f, 2604.6f}));
  return absl::OkStatus();
}

absl::Status FullyConnectedExtraLargeTest(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  static const int kInputSize = 1024;
  static const int kOutputSize = 1024;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, kInputSize);
  src_tensor.data.assign(kInputSize, 1.1f);

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(kOutputSize, 1, 1, kInputSize);
  attr.weights.data.resize(
      attr.weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float),
      2.2f);
  attr.bias.shape = Linear(kOutputSize);
  attr.bias.data.assign(kOutputSize, 3.3f);

  std::vector<float> expected(kOutputSize, 2481.38f);

  float eps;
  switch (precision) {
    case CalculationsPrecision::F32:
      eps = 2.45e-3f;
      break;
    case CalculationsPrecision::F32_F16:
      eps = 1.38f;
      break;
    case CalculationsPrecision::F16:
      eps = 39.0f;
      break;
  }
  if (precision == CalculationsPrecision::F32_F16 &&
      env.GetGpuInfo().IsApiMetal() && env.GetGpuInfo().IsIntel()) {
    eps = 3.5f;
  }
  if (precision == CalculationsPrecision::F32_F16 &&
      env.GetGpuInfo().IsGlsl()) {
    eps = 3.5f;
  }
  if (precision == CalculationsPrecision::F32_F16 &&
      env.GetGpuInfo().IsApiWebGpu()) {
    eps = 3.5f;
  }
  if (!env.GetGpuInfo().IsRoundToNearestSupported()) {
    eps *= 4.0f;
  }
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  FullyConnected operation =
      CreateFullyConnected(env.GetGpuInfo(), op_def, precision, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
      BHWC(1, 1, 1, kOutputSize), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8Test(TestExecutionEnvironment& env,
                                    CalculationsPrecision precision,
                                    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(2, 1, 1, 4);
  weights_i8.data = {2, 4, 6, 8, 10, 12, -14, 16};
  Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(2);
  biases.data = {0.5f, -0.5f};

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  DataType type = op_def.src_tensors[0].GetDataType();
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(env.GetGpuInfo(), weights_i8.shape);
  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = OHWI(1, 1, 1, 1);
  external_weights.scalar_scale = 0.5f;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       /*dst_shape_ptr=*/nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(BHWC(1, 1, 1, 2));
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {20.5f, 15.5f}));

  return absl::OkStatus();
}

absl::Status FullyConnectedInt8BlockwiseAttributesTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 8);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(2, 1, 1, 8);
  weights_i8.data = {1, 1, 1, 1, 2, 2, 2, 2,   // output 0
                     3, 3, 3, 3, 4, 4, 4, 4};  // output 1
  Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(2);
  biases.data = {0.0f, 0.0f};

  Tensor<OHWI, DataType::FLOAT32> scale;
  // 2 blocks per output channel, scale shape is (2, 1, 1, 2)
  scale.shape = OHWI(2, 1, 1, 2);
  scale.data = {0.5f, 2.0f,   // output 0: block 0 scale=0.5, block 1 scale=2.0
                1.0f, 0.1f};  // output 1: block 0 scale=1.0, block 1 scale=0.1

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), scale, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(env.GetGpuInfo(), weights_i8.shape);
  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = scale.shape;
  external_weights.scale = &scale_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       /*dst_shape_ptr=*/nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(BHWC(1, 1, 1, 2));
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td, &scale_desc, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  // Output 0, Block 0: (0*1 + 1*1 + 2*1 + 3*1) * 0.5 = 6 * 0.5 = 3
  // Output 0, Block 1: (4*2 + 5*2 + 6*2 + 7*2) * 2.0 = 44 * 2.0 = 88
  // Output 0 total: 3 + 88 = 91
  // Output 1, Block 0: (0*3 + 1*3 + 2*3 + 3*3) * 1.0 = 18 * 1.0 = 18
  // Output 1, Block 1: (4*4 + 5*4 + 6*4 + 7*4) * 0.1 = 88 * 0.1 = 8.8
  // Output 1 total: 18 + 8.8 = 26.8

  const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 0.05f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {91.0f, 26.8f}));

  return absl::OkStatus();
}

absl::Status FullyConnectedInt8BlockwiseAttributesWithZeroPointsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 8);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(2, 1, 1, 8);
  weights_i8.data = {1,  2,  3,  4,  5,  6,  7,  8,    // output 0
                     -8, -7, -6, -5, -4, -3, -2, -1};  // output 1
  Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(2);
  biases.data = {0.0f, 0.0f};

  Tensor<OHWI, DataType::FLOAT32> scale;
  // 2 blocks per output channel, scale shape is (2, 1, 1, 2)
  scale.shape = OHWI(2, 1, 1, 2);
  scale.data = {0.5f, 2.0f,   // output 0: block 0 scale=0.5, block 1 scale=2.0
                1.5f, 0.1f};  // output 1: block 0 scale=1.5, block 1 scale=0.1

  Tensor<OHWI, DataType::FLOAT32> zero_point;
  // 2 blocks per output channel, zero_point shape is (2, 1, 1, 2)
  zero_point.shape = OHWI(2, 1, 1, 2);
  zero_point.data = {2, -3,  // output 0: block 0 zp=2, block 1 zp=-3
                     1, 4};  // output 1: block 0 zp=1, block 1 zp=4

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), scale, type);
  auto zero_point_desc =
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), zero_point, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(env.GetGpuInfo(), weights_i8.shape);
  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zero_point_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       /*dst_shape_ptr=*/nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(BHWC(1, 1, 1, 2));
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td, &scale_desc, &zero_point_desc, &bias_td},
      {&dst_td}, std::make_unique<FullyConnected>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 0.1f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {509.0f, -121.4f}));

  return absl::OkStatus();
}

absl::Status FullyConnectedWeightsAsSpatialTensorTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const OHWI& weights_shape,
    const BHWC& src_shape) {
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(weights_shape);
  attr.bias.shape = Linear(weights_shape.o);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct(), 0.0f);

  TensorFloat32 weights_tensor;
  weights_tensor.shape =
      BHWC(weights_shape.o, weights_shape.h, weights_shape.w, weights_shape.i);
  weights_tensor.data = attr.weights.data;

  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ASSIGN_OR_RETURN(FullyConnected operation,
                   CreateFullyConnectedWeightsAreSpatialTensor(
                       env.GetGpuInfo(), op_def, precision, attr.weights.shape,
                       /*bias=*/nullptr, &dst_ref_tensor.shape));
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, weights_tensor},
      std::make_unique<FullyConnected>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  const float eps = GetEpsilon(precision, env.GetGpuInfo(), attr);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

namespace {
absl::Status FullyConnectedInt4Sparse2x4Test(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const ml_drift::Tensor<OHWI, DataType::UINT8>& weights_indices,
    const OperationDef& op_def, CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor;
  dst_ref_tensor.shape = src_tensor.shape;
  dst_ref_tensor.shape.c = weights_i4.shape.o;
  dst_ref_tensor.data.resize(dst_ref_tensor.shape.DimensionsProduct(), 0.0f);
  for (int y = 0; y < src_tensor.shape.h; ++y) {
    for (int x = 0; x < src_tensor.shape.w; ++x) {
      for (int dst_ch = 0; dst_ch < weights_i4.shape.o; ++dst_ch) {
        float dst_val = 0.0f;
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
          dst_val +=
              src_vals[ind0] * weights_v0_i32 + src_vals[ind1] * weights_v1_i32;
        }
        dst_ref_tensor
            .data[dst_ref_tensor.shape.LinearIndex({0, y, x, dst_ch})] =
            dst_val;
      }
    }
  }

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scale;
  weights_scale.shape = OHWI(weights_i4.shape.o, 1, 1, 1);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = OHWI(weights_i4.shape.o, 1, 1, 1);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct(),
                                 0.0f);

  auto operation = CreateFullyConnectedInt4Sparse2x4(
      exec_env.GetGpuInfo(), op_def, precision, weights_i4, weights_indices,
      weights_scale, weights_zero_point, {});
  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status FullyConnectedInt4Sparse2x4Test(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage,
                                             const BHWC& src_shape,
                                             int dst_channels) {
  const int src_channels = src_shape.c;
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_channels / 2);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i4.shape.DimensionsProduct(); ++i) {
    weights_i4.data[i] = ((i % 1331) % 131) % 16 - 8;
  }
  ml_drift::Tensor<OHWI, DataType::UINT8> weights_indices;
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

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = static_cast<int>(src_tensor.data[i] * 16.0f);
  }

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(FullyConnectedInt4Sparse2x4Test(
      env, src_tensor, weights_i4, weights_indices, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedBigTest(TestExecutionEnvironment& exec_env,
                                   const FullyConnectedAttributes& attr,
                                   const TensorFloat32& src_tensor,
                                   const OperationDef& op_def,
                                   CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  auto operation = CreateFullyConnected(exec_env.GetGpuInfo(), op_def,
                                        precision, attr, &dst_ref_tensor.shape);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedBigTest(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 351;
  const int dst_channels = 172;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedBigTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedWidth3Height2BigTest(TestExecutionEnvironment& env,
                                                CalculationsPrecision precision,
                                                TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 179;
  const int dst_channels = 93;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 2, 3, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedBigTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedExternalTest(
    TestExecutionEnvironment& exec_env, const FullyConnectedAttributes attr,
    WeightsLayout weights_layout, const TensorFloat32& src_tensor,
    const OperationDef& op_def, CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  TensorFloat32 dst_ref_tensor;
  if (attr.weights.shape.h != 1 || attr.weights.shape.w != 1) {
    dst_ref_tensor = FullyConnectedRefDifferentWeightsForHeight(
        attr.weights, src_tensor, runtime_channels);
  } else {
    dst_ref_tensor =
        FullyConnectedReference(attr, src_tensor, runtime_channels);
  }

  DataType type = op_def.src_tensors[0].GetDataType();
  TensorDescriptor bias_td = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), type, attr.bias);

  WeightsDescription weights_desc;
  weights_desc.type = DeduceDataTypeFromPrecision(precision);
  weights_desc.layout = weights_layout;
  weights_desc.output_group_size = DivideRoundUp(attr.weights.shape.o, 4);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(attr.weights, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = attr.weights.shape;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       &dst_ref_tensor.shape,
                       /*src_exp=*/nullptr,
                       runtime_channels.GenerateConvRuntimeCheckDesc()));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  std::vector<TensorDescriptor*> src_cpu;
  src_cpu.push_back(&src_td);
  for (int i = 0; i < weights_gpu.size(); ++i) {
    src_cpu.push_back(&weights_gpu[i]);
  }
  src_cpu.push_back(&bias_td);

  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  TensorInt32 params_tensor = runtime_channels.GenerateTensorInt32();
  if (std::any_of(params_tensor.data.begin(), params_tensor.data.end(),
                  [](int x) { return x != -1; })) {
    params_td.UploadData(params_tensor);
    src_cpu.push_back(&params_td);
  }

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 2.0f;

  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_cpu, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  const int dst_num_ch = dst_ref_tensor.shape.c;
  const int dst_end_ch = runtime_channels.dst_end_ch.has_value()
                             ? *runtime_channels.dst_end_ch
                             : dst_num_ch;
  // `dst_tensor.shape.h` and `dst_tensor.shape.w` might not equal 1
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

absl::Status FullyConnectedExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 53;
  const int dst_channels = 17;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  std::vector<TestingRuntimeChannels> test_runtime_channels = {
      TestingRuntimeChannels(),
      {.src_end_ch = 4, .dst_end_ch = 8},
      {.src_end_ch = 8},
      {.dst_end_ch = 14}};
  for (auto weights_layout : {WeightsLayout::kOSpatialIOGroupI4O4,
                              WeightsLayout::kOSpatialIOGroupO4I4}) {
    for (auto runtime_channels : test_runtime_channels) {
      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      MLD_EXPECT_OK(FullyConnectedExternalTest(env, attr, weights_layout,
                                           src_tensor, op_def, precision,
                                           runtime_channels));
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedBatchedWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 53;
  const int dst_channels = 17;
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 4, 1, src_channels));
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 4, 2, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  std::vector<TestingRuntimeChannels> test_runtime_channels = {
      TestingRuntimeChannels(),
      {.src_end_ch = 4, .dst_end_ch = 9},
      {.src_end_ch = 8},
      {.dst_end_ch = 15}};
  for (auto weights_layout :
       {WeightsLayout::kOSpatialIOGroupI4O4,
        WeightsLayout::kOSpatialIOGroupO4I4,
        WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4}) {
    for (auto runtime_channels : test_runtime_channels) {
      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      MLD_EXPECT_OK(FullyConnectedExternalTest(env, attr, weights_layout,
                                           src_tensor, op_def, precision,
                                           runtime_channels));
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& biases,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  auto operation = CreateFullyConnectedInt8(
      exec_env.GetGpuInfo(), op_def, precision, weights_i8, weights_scale,
      weights_zero_point, biases, &dst_ref_tensor.shape);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<FullyConnected>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 171;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
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
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt8Test(env, weights_i8, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8Width2Batch2BigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 171;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
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
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(2, 1, 2, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  MLD_EXPECT_OK(FullyConnectedInt8Test(env, weights_i8, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8GroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 7 * 13 * 4;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, 13));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 128.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt8Test(env, weights_i8, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8ExternalTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = MakeZeroTensor(Linear(weights_i8.shape.o));
  TensorFloat32 dst_ref_tensor;
  if (attr.weights.shape.h != 1) {
    dst_ref_tensor =
        FullyConnectedRefDifferentWeightsForHeight(attr.weights, src_tensor);
  } else {
    dst_ref_tensor = FullyConnectedReference(attr, src_tensor);
  }

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                   weights_scale, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                weights_zero_point, type);

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(exec_env.GetGpuInfo(), weights_i8.shape);
  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights,
                       /*bias=*/nullptr, &dst_ref_tensor.shape));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  std::vector<TensorDescriptor*> src_cpu = {&src_td, &weights_i8_td,
                                            &scale_desc, &zp_desc};

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 2.0f;

  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_cpu, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8BatchedWeightsIdsTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const TensorFloat32& src_tensor, const TensorInt32& weights_ids,
    const OperationDef& op_def, CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = MakeZeroTensor(Linear(weights_i8.shape.o));
  TensorFloat32 dst_ref_tensor = FullyConnectedWeightsBatchIdsReference(
      attr.weights, src_tensor, weights_ids);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                   weights_scale, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                weights_zero_point, type);

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(exec_env.GetGpuInfo(), weights_i8.shape);
  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;
  ASSIGN_OR_RETURN(
      auto operation,
      CreateFullyConnectedWeightsBatchIds(
          exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
          op_def.src_tensors[1], op_def.dst_tensors[0], external_weights,
          /*bias=*/nullptr, &dst_ref_tensor.shape));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor weights_ids_td = op_def.src_tensors[1];
  weights_ids_td.UploadData(weights_ids);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  std::vector<TensorDescriptor*> src_cpu = {
      &src_td, &weights_ids_td, &weights_i8_td, &scale_desc, &zp_desc};

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 2.0f;

  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_cpu, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8ExternalTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& biases,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor;
  if (attr.weights.shape.h != 1 || attr.weights.shape.w != 1) {
    dst_ref_tensor = FullyConnectedRefDifferentWeightsForHeight(
        attr.weights, src_tensor, runtime_channels);
  } else {
    dst_ref_tensor =
        FullyConnectedReference(attr, src_tensor, runtime_channels);
  }

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                   weights_scale, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                weights_zero_point, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(exec_env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(exec_env.GetGpuInfo(), weights_i8.shape);
  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       &dst_ref_tensor.shape,
                       /*src_exp=*/nullptr,
                       runtime_channels.GenerateConvRuntimeCheckDesc()));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  std::vector<TensorDescriptor*> src_cpu = {&src_td, &weights_i8_td,
                                            &scale_desc, &zp_desc, &bias_td};
  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  TensorInt32 params_tensor = runtime_channels.GenerateTensorInt32();
  if (std::any_of(params_tensor.data.begin(), params_tensor.data.end(),
                  [](int x) { return x != -1; })) {
    params_td.UploadData(params_tensor);
    src_cpu.push_back(&params_td);
  }

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 2.0f;

  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_cpu, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  const int dst_num_ch = dst_ref_tensor.shape.c;
  const int dst_end_ch = runtime_channels.dst_end_ch.has_value()
                             ? *runtime_channels.dst_end_ch
                             : dst_num_ch;
  // `dst_tensor.shape.h` and `dst_tensor.shape.w` might not equal 1
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

absl::Status FullyConnectedInt8ExternalBigTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  constexpr int kSrcChannels = 171;
  constexpr int kDstChannels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, kSrcChannels));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 128.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(kDstChannels));

  auto src_shape = BHWC(1, 1, 1, kSrcChannels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  std::vector<TestingRuntimeChannels> test_runtime_channels = {
      TestingRuntimeChannels(),
      {.src_end_ch = 4, .dst_end_ch = 8},
      {.src_end_ch = 8},
      {.dst_end_ch = 14}};

  for (auto runtime_channels : test_runtime_channels) {
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
    MLD_EXPECT_OK(FullyConnectedInt8ExternalTest(env, weights_i8, weights_scales,
                                             biases, src_tensor, op_def,
                                             precision, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8BatchedWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int kSrcChannels = 32;
  const int kDstChannels = 32;
  const int kWeightsBatch = 4;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(kDstChannels, kWeightsBatch, 1, kSrcChannels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, kWeightsBatch, 1, kSrcChannels));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }
  auto weights_scales =
      MakeSyntheticTensor(OHWI(kDstChannels, kWeightsBatch, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 128.0f;
  }

  auto src_shape = BHWC(1, kWeightsBatch, 1, kSrcChannels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt8ExternalTest(env, weights_i8, weights_scales,
                                           src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8BatchedWeightsIdsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, const BHWC& src_shape,
    const BHWC& batch_ids_shape) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int kSrcChannels = src_shape.c;
  const int kDstChannels = 32;
  const int kWeightsBatch = 16;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(kDstChannels, kWeightsBatch, 1, kSrcChannels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, kWeightsBatch, 1, kSrcChannels));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }
  auto weights_scales =
      MakeSyntheticTensor(OHWI(kDstChannels, kWeightsBatch, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 128.0f;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorInt32 batch_ids_tensor;
  batch_ids_tensor.shape = batch_ids_shape;
  batch_ids_tensor.data.resize(batch_ids_tensor.shape.DimensionsProduct());
  unsigned int seed = 42;
  for (int i = 0; i < batch_ids_tensor.data.size(); ++i) {
    batch_ids_tensor.data[i] = rand_r(&seed) % kWeightsBatch;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt8BatchedWeightsIdsTest(
      env, weights_i8, weights_scales, src_tensor, batch_ids_tensor, op_def,
      precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt8ExternalGroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  constexpr int kSrcChannels = 11 * 4 * 7;
  constexpr int kDstChannels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, kSrcChannels));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, 7));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 128.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(kDstChannels));

  auto src_shape = BHWC(1, 1, 1, kSrcChannels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  std::vector<TestingRuntimeChannels> test_runtime_channels = {
      TestingRuntimeChannels(),
  };

  for (auto runtime_channels : test_runtime_channels) {
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
    MLD_EXPECT_OK(FullyConnectedInt8ExternalTest(env, weights_i8, weights_scales,
                                             biases, src_tensor, op_def,
                                             precision, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedSi8Wi8Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i8);

  WeightsDescription weights_desc =
      GetFullyConnectedInt8WeightsDesc(exec_env.GetGpuInfo(), weights_i8.shape);
  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), CalculationsPrecision::F32,
                       op_def.src_tensors[0], op_def.dst_tensors[0],
                       external_weights, nullptr, nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor_i8);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, testing::ContainerEq(dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedSi8Wi8BigTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage) {
  const DataType src_data_type = DataType::INT8;
  const DataType dst_data_type = DataType::INT32;
  const int src_channels = 112;
  const int dst_channels = 152;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 256.0f;
    weights_i8.data[i] = std::max(std::min(val, 255), 0) - 128;
  }

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);  // range [-1, 1]

  Tensor<BHWC, DataType::INT8> src_int8_tensor;
  src_int8_tensor.shape = src_shape;
  src_int8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_int8_tensor.data[i] = src_tensor.data[i] * 128.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedSi8Wi8Test(env, src_int8_tensor, weights_i8, op_def));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt4Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& biases,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i4, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                   weights_scale, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(exec_env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(exec_env.GetGpuInfo(), weights_i4.shape);
  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       &dst_ref_tensor.shape));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i4_td, &scale_desc, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt4BlockwiseTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 16;
  const int dst_channels = 4;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct(), 1);  // all 1s

  // 2 blocks per output channel, scale shape is (4, 1, 1, 2)
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(dst_channels, 1, 1, 2);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct(), 0.5f);

  ml_drift::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(dst_channels);
  biases.data.resize(biases.shape.DimensionsProduct(), 0.0f);

  auto src_shape = BHWC(1, 1, 1, src_channels);
  TensorFloat32 src_tensor;
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct(), 1.0f);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  // Block 0: 8 * 1 * 0.5 = 4
  // Block 1: 8 * 1 * 0.5 = 4
  // Total: 4 + 4 = 8

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;  // empty

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), weights_scales, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(env.GetGpuInfo(), weights_i4.shape);
  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       /*dst_shape_ptr=*/nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(BHWC(1, 1, 1, dst_channels));
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i4_td, &scale_desc, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);

  float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
  std::vector<float> expected(dst_channels, 8.0f);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected));

  return absl::OkStatus();
}

absl::Status FullyConnectedInt4BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 171;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
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
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt4Test(env, weights_i4, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2BlockwiseTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 16;
  const int dst_channels = 4;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct(), 1);  // all 1s

  // 2 blocks per output channel, scale shape is (4, 1, 1, 2)
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(dst_channels, 1, 1, 2);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct(), 0.5f);

  ml_drift::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(dst_channels);
  biases.data.resize(biases.shape.DimensionsProduct(), 0.0f);

  auto src_shape = BHWC(1, 1, 1, src_channels);
  TensorFloat32 src_tensor;
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct(), 1.0f);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  // Block 0: 8 * 1 * 0.5 = 4
  // Block 1: 8 * 1 * 0.5 = 4
  // Total: 4 + 4 = 8

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), weights_scales, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt2WeightsDesc(env.GetGpuInfo(), weights_i2.shape);
  TensorDescriptor weights_i2_td =
      GetTensorDescriptorForWeightsLayout(weights_i2, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i2.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       /*dst_shape_ptr=*/nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(BHWC(1, 1, 1, dst_channels));
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i2_td, &scale_desc, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));

  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
  std::vector<float> expected(dst_channels, 8.0f);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected));

  return absl::OkStatus();
}

absl::Status FullyConnectedInt4WidthIs4BigTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 171;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
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
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 4, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt4Test(env, weights_i4, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt4GroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 7 * 13 * 4;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, 13));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 8.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt4Test(env, weights_i4, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt4ExternalTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& biases,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i4, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                   weights_scale, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                weights_zero_point, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(exec_env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(exec_env.GetGpuInfo(), weights_i4.shape);
  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       &dst_ref_tensor.shape));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 2.0f;

  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i4_td, &scale_desc, &zp_desc, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt4ExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  constexpr int kSrcChannels = 171;
  constexpr int kDstChannels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, kSrcChannels));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 8.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(kDstChannels));

  auto src_shape = BHWC(1, 1, 1, kSrcChannels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt4ExternalTest(
      env, weights_i4, weights_scales, biases, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt4ExternalGroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  constexpr int kSrcChannels = 7 * 13 * 4;
  constexpr int kDstChannels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, kSrcChannels));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, 7));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 8.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(kDstChannels));

  auto src_shape = BHWC(1, 1, 1, kSrcChannels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt4ExternalTest(
      env, weights_i4, weights_scales, biases, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedSi8Wi4Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i4);

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(exec_env.GetGpuInfo(), weights_i4.shape);
  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), CalculationsPrecision::F32,
                       op_def.src_tensors[0], op_def.dst_tensors[0],
                       external_weights, nullptr, nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor_i8);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i4_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, testing::ContainerEq(dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedSi8Wi4BigTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage) {
  const DataType src_data_type = DataType::INT8;
  const DataType dst_data_type = DataType::INT32;
  const int src_channels = 112;
  const int dst_channels = 152;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);  // range [-1, 1]

  Tensor<BHWC, DataType::INT8> src_int8_tensor;
  src_int8_tensor.shape = src_shape;
  src_int8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_int8_tensor.data[i] = src_tensor.data[i] * 128.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedSi8Wi4Test(env, src_int8_tensor, weights_i4, op_def));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i2,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& biases,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i2, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                   weights_scale, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(exec_env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt2WeightsDesc(exec_env.GetGpuInfo(), weights_i2.shape);
  TensorDescriptor weights_i2_td =
      GetTensorDescriptorForWeightsLayout(weights_i2, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i2.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       &dst_ref_tensor.shape));

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i2_td, &scale_desc, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 171;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 4.0f;
    weights_i2.data[i] = std::max(std::min(val, 3), 0) - 2;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 2.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt2Test(env, weights_i2, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2WidthIs4BigTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 171;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 4.0f;
    weights_i2.data[i] = std::max(std::min(val, 3), 0) - 2;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 2.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 4, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt2Test(env, weights_i2, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2GroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 7 * 13 * 4;
  const int dst_channels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 4.0f;
    weights_i2.data[i] = std::max(std::min(val, 3), 0) - 2;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, 13));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 2.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt2Test(env, weights_i2, weights_scales, biases,
                                   src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2ExternalTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i2,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& biases,
    const TensorFloat32& src_tensor, const OperationDef& op_def,
    CalculationsPrecision precision) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scale.shape;
  weights_zero_point.data.resize(weights_scale.shape.DimensionsProduct(), 0.0f);
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i2, weights_scale, weights_zero_point);
  attr.bias = biases;
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                   weights_scale, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(exec_env.GetGpuInfo(),
                                                weights_zero_point, type);
  TensorDescriptor bias_td =
      CreateConstantLinearTensorDescriptor(exec_env.GetGpuInfo(), type, biases);

  WeightsDescription weights_desc =
      GetFullyConnectedInt2WeightsDesc(exec_env.GetGpuInfo(), weights_i2.shape);
  TensorDescriptor weights_i2_td =
      GetTensorDescriptorForWeightsLayout(weights_i2, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i2.shape;
  external_weights.scale_zp_shape = weights_scale.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights, &bias_td,
                       &dst_ref_tensor.shape));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 2.0f;

  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i2_td, &scale_desc, &zp_desc, &bias_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2ExternalWeightsBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  constexpr int kSrcChannels = 171;
  constexpr int kDstChannels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, kSrcChannels));
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 4.0f;
    weights_i2.data[i] = std::max(std::min(val, 3), 0) - 2;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, 1));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 2.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(kDstChannels));

  auto src_shape = BHWC(1, 1, 1, kSrcChannels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt2ExternalTest(
      env, weights_i2, weights_scales, biases, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedInt2ExternalGroupedQuantizationBigTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  constexpr int kSrcChannels = 7 * 13 * 4;
  constexpr int kDstChannels = 533;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  auto weights_f32 =
      MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, kSrcChannels));
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 4.0f;
    weights_i2.data[i] = std::max(std::min(val, 3), 0) - 2;
  }
  auto weights_scales = MakeSyntheticTensor(OHWI(kDstChannels, 1, 1, 7));
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] /= 2.0f;
  }
  auto biases = MakeSyntheticTensor(Linear(kDstChannels));

  auto src_shape = BHWC(1, 1, 1, kSrcChannels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedInt2ExternalTest(
      env, weights_i2, weights_scales, biases, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status FullyConnectedSi8Wi2Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i2,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i2);

  WeightsDescription weights_desc =
      GetFullyConnectedInt2WeightsDesc(exec_env.GetGpuInfo(), weights_i2.shape);
  TensorDescriptor weights_i2_td =
      GetTensorDescriptorForWeightsLayout(weights_i2, weights_desc);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i2.shape;
  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(
                       exec_env.GetGpuInfo(), CalculationsPrecision::F32,
                       op_def.src_tensors[0], op_def.dst_tensors[0],
                       external_weights, nullptr, nullptr));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor_i8);

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i2_td}, {&dst_td},
      std::make_unique<FullyConnected>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, testing::ContainerEq(dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status FullyConnectedSi8Wi2BigTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage) {
  const DataType src_data_type = DataType::INT8;
  const DataType dst_data_type = DataType::INT32;
  const int src_channels = 112;
  const int dst_channels = 152;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    weights_i2.data[i] = (weights_f32.data[i] + 0.0f) * 1.0f;
  }

  auto src_shape = BHWC(1, 1, 1, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);  // range [-1, 1]

  Tensor<BHWC, DataType::INT8> src_int8_tensor;
  src_int8_tensor.shape = src_shape;
  src_int8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_int8_tensor.data[i] = src_tensor.data[i] * 128.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
  MLD_EXPECT_OK(FullyConnectedSi8Wi2Test(env, src_int8_tensor, weights_i2, op_def));
  return absl::OkStatus();
}

absl::Status FullyConnectedPackedGroupsTest(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  const int src_channels = 64;
  const int width = 64;
  const int dst_channels = 32;
  const int weights_batch_size = 16;
  const int num_groups_per_item = 4;

  Tensor<OHWI, DataType::FLOAT32> weights = MakeSyntheticTensor(
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
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  ConvRuntimeCheckDesc::PackedGroups packed_groups;
  packed_groups.params_offset = 0;
  packed_groups.num_groups = weights_batch_size;
  packed_groups.max_group_size = width;
  ConvRuntimeCheckDesc runtime_check;
  runtime_check.packed_groups = packed_groups;

  WeightsDescription weights_desc =
      GetFullyConnectedWeightsDesc(data_type, weights.shape);

  TensorDescriptor src_desc = TensorDescriptor(data_type, storage, Layout::HWC);
  src_desc.UploadData(src_packed);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights, weights_desc);

  std::vector<int32_t> runtime_params_cpu(weights_batch_size * 2, 0);
  for (int i = 0; i < weights_batch_size; ++i) {
    runtime_params_cpu[i] = groups_sizes.data[i];
    runtime_params_cpu[weights_batch_size + i] = groups_offsets.data[i];
  }
  TensorDescriptor runtime_params_td(DataType::INT32, TensorStorageType::BUFFER,
                                     Layout::LINEAR);
  runtime_params_td.SetBHWCShape(BHWC(1, 1, 1, weights_batch_size * 2));
  runtime_params_td.UploadData(runtime_params_cpu.data());

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights.shape;
  ASSIGN_OR_RETURN(auto conv_fc,
                   CreateFullyConnectedExternalWeights(
                       env.GetGpuInfo(), precision, op_def.src_tensors[0],
                       op_def.dst_tensors[0], external_weights,
                       /*bias=*/nullptr, &dst_packed_shape,
                       /*src_exp=*/nullptr, runtime_check));

  TensorDescriptor dst_desc = TensorDescriptor(data_type, storage, Layout::HWC);
  dst_desc.SetBHWCShape(dst_packed_shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc, &weights_gpu[0], &runtime_params_td}, {&dst_desc},
      std::make_unique<FullyConnected>(std::move(conv_fc))));

  TensorFloat32 dst_packed;
  dst_desc.DownloadData(&dst_packed);
  auto dst_tensor =
      RemapFromReference(dst_packed, packed_groups_map, num_groups_per_item);
  float eps = GetEpsilon(precision, env.GetGpuInfo()) * 2.0f * src_channels;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
