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

#include "ml_drift/common/kernels/tests/conv_generic_test_util.h"

#include <math.h>

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
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_weights_converter.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/kernels/winograd.h"
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
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status ConvGeneric1x1SimpleWeightsTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 1, 2);
  attr_weights.data = {1.0f, 1.0f, 1.0f, 1.0f};
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvGeneric operation =
      CreateConvGeneric(env.GetGpuInfo(), op_def, precision, attr);
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.0f, 1.0f, 5.0f, 5.0f, 9.0f, 9.0f, 13.0f, 13.0f}));
  return absl::OkStatus();
}

absl::Status ConvGeneric1x1Test(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 1, 1, 2);
  attr_weights.data = {1.0f, 2.0f, 3.0f, 4.0f};
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvGeneric operation =
      CreateConvGeneric(env.GetGpuInfo(), op_def, precision, attr);
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {2.5f, 3.5f, 8.5f, 17.5f, 14.5f, 31.5f, 20.5f, 45.5f}));
  return absl::OkStatus();
}

absl::Status ConvGenericSimpleWeightsTest(TestExecutionEnvironment& env,
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
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(1, 2, 2, 2);
  attr_weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvGeneric operation =
      CreateConvGeneric(env.GetGpuInfo(), op_def, precision, attr);
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(operation)),
      BHWC(1, 2, 2, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {28.0f, 18.0f, 22.0f, 13.0f}));
  return absl::OkStatus();
}

absl::Status ConvGenericTest(TestExecutionEnvironment& env,
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
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(2, 2, 2, 2);
  attr_weights.data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvGeneric operation =
      CreateConvGeneric(env.GetGpuInfo(), op_def, precision, attr);
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {168.5f, 391.5f, 80.5f, 223.5f, 60.5f,
                                         235.5f, 20.5f, 123.5f}));
  return absl::OkStatus();
}

absl::Status ConvGenericGroupedTest(TestExecutionEnvironment& env,
                                    CalculationsPrecision precision,
                                    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 8);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Convolution2DAttributes attr;
  attr.groups = 2;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(8, 1, 1, 4);
  attr_weights.data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                       9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                       17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                       25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f};
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(8);
  attr.bias.data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ConvGeneric operation =
      CreateConvGeneric(env.GetGpuInfo(), op_def, precision, attr);
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(operation)),
      BHWC(1, 1, 1, 8), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {20.0f, 44.0f, 68.0f, 92.0f, 412.0f,
                                         500.0f, 588.0f, 676.0f}));
  return absl::OkStatus();
}

absl::Status ConvGenericWinograd3x3TileNxNTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage,
                                               int tile_size) {
  const int src_channels = 7;
  const int dst_channels = 13;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(10, 10);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(dst_channels, 3, 3, src_channels);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < attr_weights.data.size(); ++i) {
    attr_weights.data[i] = sin(i);
  }
  attr.bias.shape = Linear(dst_channels);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < attr.bias.data.size(); ++i) {
    attr.bias.data[i] = sin(i);
  }

  const int tile_size_outer = tile_size;
  const int tile_size_inner = tile_size_outer - 2;
  auto src_shape = BHWC(1, 170, 130, src_channels);
  auto dst_shape = CalculateOutputShape(src_shape, attr);
  BHWC conv_wino_shape;
  conv_wino_shape.b = dst_shape.b;
  conv_wino_shape.h = tile_size_outer * tile_size_outer;
  conv_wino_shape.w = DivideRoundUp(dst_shape.w, tile_size_inner) *
                      DivideRoundUp(dst_shape.h, tile_size_inner);
  conv_wino_shape.c = dst_shape.c;

  TensorFloat32 src_tensor;
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(i);
  }

  TensorFloat32 output_conv_ref = ConvolutionReference(attr, src_tensor);
  TensorFloat32 winograd_forward_ref =
      Winograd3x3ForwardRef(src_tensor, attr.padding, tile_size);

  float eps = 0.0f;
  if (tile_size == 6) {
    eps = precision == CalculationsPrecision::F32 ? 4e-5f : 0.2f;
  } else if (tile_size == 8) {
    eps = precision == CalculationsPrecision::F32 ? 4e-4f : 1.2f;
  } else if (tile_size == 10) {
    eps = precision == CalculationsPrecision::F32 ? 4e-3f : 6.0f;
  }
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  TensorFloat32 output_conv;
  {  // calculating convolution with winograd algorithm
    auto gpu_op1 = CreateWinograd3x3TiledXForward(env.GetGpuInfo(), op_def,
                                                  attr.padding, tile_size);
    std::unique_ptr<GPUOperation> op1_ptr =
        std::make_unique<Winograd3x3TiledXForward>(std::move(gpu_op1));

    Tensor<OHWI, DataType::FLOAT32> wino_weights;
    RearrangeWeightsToWinograd3x3TileNxN(attr_weights, &wino_weights,
                                         tile_size);

    Convolution2DAttributes wino_attr;
    wino_attr.padding.prepended = HW(0, 0);
    wino_attr.padding.appended = HW(0, 0);
    wino_attr.strides = HW(1, 1);
    wino_attr.dilations = HW(1, 1);
    auto& wino_attr_weights =
        wino_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    wino_attr_weights.shape = wino_weights.shape;

    auto convolution = CreateConvGenericExternalWeights(
        env.GetGpuInfo(), op_def, precision, attr, /*bias=*/nullptr,
        &conv_wino_shape,
        /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true);
    std::vector<TensorDescriptor> weights_gpu =
        GetTensorDescriptorsForWeightsLayout(
            wino_weights, convolution.GetWeightsDescription());

    auto gpu_op3 = CreateWinograd3x3TiledXBackward(env.GetGpuInfo(), op_def,
                                                   attr.bias, tile_size);
    std::unique_ptr<GPUOperation> op3_ptr =
        std::make_unique<Winograd3x3TiledXBackward>(std::move(gpu_op3));

    TensorFloat32 winograd_forward;
    BHWC winograd_forward_shape = conv_wino_shape;
    winograd_forward_shape.c = src_shape.c;
    MLD_EXPECT_OK(env.ExecuteGPUOperation(src_tensor, std::move(op1_ptr),
                                      winograd_forward_shape,
                                      &winograd_forward));
    EXPECT_THAT(winograd_forward.data,
                Pointwise(FloatNear(eps), winograd_forward_ref.data));

    TensorFloat32 output_conv_wino;
    {
      TensorDescriptor src_td = op_def.src_tensors[0];
      src_td.UploadData(winograd_forward);

      std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
      srcs_td[0] = &src_td;
      for (int i = 0; i < weights_gpu.size(); ++i) {
        srcs_td[1 + i] = &weights_gpu[i];
      }
      TensorDescriptor dst_td = op_def.dst_tensors[0];
      dst_td.SetBHWCShape(BHWC(conv_wino_shape));
      RETURN_IF_ERROR(env.ExecuteGPUOperation(
          srcs_td, {&dst_td},
          std::make_unique<ConvGeneric>(std::move(convolution))));
      dst_td.DownloadData(&output_conv_wino);
    }

    MLD_EXPECT_OK(env.ExecuteGPUOperation(output_conv_wino, std::move(op3_ptr),
                                      dst_shape, &output_conv));
  }
  EXPECT_THAT(output_conv.data,
              Pointwise(FloatNear(eps), output_conv_ref.data));
  return absl::OkStatus();
}

absl::Status ConvGeneric1x1Int8SymmetricTest(TestExecutionEnvironment& env,
                                             TensorStorageType src_storage,
                                             TensorStorageType dst_storage) {
  ml_drift::Tensor<BHWC, DataType::INT8> src_tensor_i8;
  src_tensor_i8.shape = BHWC(1, 2, 2, 2);
  src_tensor_i8.data = {0, -1, -2, 3, 4, -5, -6, 7};

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(2, 1, 1, 2);
  weights_i8.data = {8, 9, -3, 5};
  weights_i8.data.resize(weights_i8.data.size() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));

  const PackedType quantized_type =
      GetConvGenericInt8SrcType(env.GetGpuInfo(), src_tensor_i8.shape);
  const bool is_signed_src = IsSigned(quantized_type);

  if (is_signed_src) {
    const DataType src_data_type = DataType::INT8;
    const DataType dst_data_type = DataType::INT32;
    OperationDef op_def;
    op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
    op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
    ConvGeneric operation = CreateConvGenericInt8(
        env.GetGpuInfo(), op_def,
        /*src_packed_type=*/PackedType::kInt8C4, weights_i8);

    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(src_tensor_i8);

    TensorDescriptor dst_td = op_def.dst_tensors[0];
    dst_td.SetBHWCShape(BHWC(1, 2, 2, 2));
    MLD_EXPECT_OK(env.ExecuteGPUOperation(
        {&src_td}, {&dst_td},
        std::make_unique<ConvGeneric>(std::move(operation))));
    TensorInt32 dst_tensor;
    dst_td.DownloadData(&dst_tensor);
    std::vector<int32_t> expected_data = {-9, -5, 11, 21, -13, -37, 15, 53};
    EXPECT_THAT(dst_tensor.data, expected_data);
  }
  return absl::OkStatus();
}

namespace {
absl::Status ConvolutionGenericTest(TestExecutionEnvironment& exec_env,
                                    const Convolution2DAttributes& attr,
                                    const TensorFloat32& src_tensor,
                                    const OperationDef& op_def,
                                    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  auto operation =
      CreateConvGeneric(exec_env.GetGpuInfo(), op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGeneric1x1BigTest(TestExecutionEnvironment& env,
                                   CalculationsPrecision precision,
                                   TensorStorageType storage) {
  const int src_channels = 32;
  const int dst_channels = 32;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 12, 16, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(ConvolutionGenericTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvGeneric1x1BatchedBigTest(TestExecutionEnvironment& env,
                                          CalculationsPrecision precision,
                                          TensorStorageType storage) {
  const int src_channels = 32;
  const int dst_channels = 32;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(7, 12, 16, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  MLD_EXPECT_OK(ConvolutionGenericTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvGenericBigTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage) {
  const int src_channels = 13;
  const int dst_channels = 19;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(2, 1);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 15, 12, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(ConvolutionGenericTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvGenericBatchedBigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage) {
  const int src_channels = 13;
  const int dst_channels = 19;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(2, 1);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(3, 15, 12, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  MLD_EXPECT_OK(ConvolutionGenericTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvGenericGroupedBigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage) {
  Convolution2DAttributes attr;
  attr.groups = 7;
  const int src_channels = 8 * attr.groups;
  const int dst_channels = 12 * attr.groups;
  attr.padding.prepended = HW(2, 1);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels / attr.groups));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 15, 12, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  MLD_EXPECT_OK(ConvolutionGenericTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvGenericPackedGroupsTest(TestExecutionEnvironment& env,
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

  auto convolution = CreateConvGenericExternalWeights(
      env.GetGpuInfo(), op_def, precision, conv_attr, /*bias=*/nullptr,
      &dst_packed_shape,
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
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {srcs_td}, {&dst_desc},
      std::make_unique<ConvGeneric>(std::move(convolution))));

  TensorFloat32 dst_packed;
  dst_desc.DownloadData(&dst_packed);
  auto dst_tensor =
      RemapFromReference(dst_packed, packed_groups_map, num_groups_per_item);
  float eps = GetEpsilon(precision, env.GetGpuInfo()) * 2.0f * src_shape.c;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status ConvGenericExternalWfloatTest(TestExecutionEnvironment& env,
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
  weights_f32.data.resize(weights_f32.data.size() +
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

  if (!SupportsConvGeneric(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvGenericExternalWeights(
      env.GetGpuInfo(), conv_def, precision, external_weights, /*bias=*/nullptr,
      &dst_shape,
      /*src_exp=*/nullptr,
      batched_weights);

  std::vector<TensorDescriptor> weights_td =
      GetTensorDescriptorsForWeightsLayout(weights_f32, weights_desc);

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, env.GetGpuInfo()) * src_channels * 4.0f;
  std::vector<TensorDescriptor*> src_cpu = {&src_td};
  for (auto& td : weights_td) {
    src_cpu.push_back(&td);
  }
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_cpu, {&dst_td}, std::make_unique<ConvGeneric>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvGenericExternalWi8Test(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage,
                                        const BHWC& src_shape, int dst_channels,
                                        bool batched_weights, int group_size,
                                        int scale_zp_batch) {
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
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), weights_scales, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(),
                                                weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i8.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;

  if (!SupportsConvGeneric(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvGenericExternalWeights(
      env.GetGpuInfo(), conv_def, precision, external_weights, /*bias=*/nullptr,
      &dst_shape,
      /*src_exp=*/nullptr,
      batched_weights);

  TensorDescriptor weights_i8_td =
      GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, env.GetGpuInfo()) * group_size * 8.0f;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td, &scale_desc, &zp_desc}, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvGenericExternalWi4Test(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage,
                                        const BHWC& src_shape, int dst_channels,
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
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), weights_scales, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(),
                                                weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;

  if (!SupportsConvGeneric(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvGenericExternalWeights(
      env.GetGpuInfo(), conv_def, precision, external_weights, /*bias=*/nullptr,
      &dst_shape,
      /*src_exp=*/nullptr,
      batched_weights);

  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, env.GetGpuInfo()) * group_size * 4.0f;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i4_td, &scale_desc, &zp_desc}, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvGenericExternalWi2Test(TestExecutionEnvironment& env,
                                        CalculationsPrecision precision,
                                        TensorStorageType storage,
                                        const BHWC& src_shape, int dst_channels,
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
      ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(), weights_scales, type);
  auto zp_desc = ScaleOrZeroPointToFCTensorDesc(env.GetGpuInfo(),
                                                weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i2.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  external_weights.zero_point = &zp_desc;

  if (!SupportsConvGeneric(env.GetGpuInfo(), precision, external_weights)) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  auto operation = CreateConvGenericExternalWeights(
      env.GetGpuInfo(), conv_def, precision, external_weights, /*bias=*/nullptr,
      &dst_shape,
      /*src_exp=*/nullptr,
      batched_weights);

  TensorDescriptor weights_i2_td =
      GetTensorDescriptorForWeightsLayout(weights_i2, weights_desc);

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(src_tensor);

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  float eps = GetEpsilon(precision, env.GetGpuInfo()) * group_size * 2.0f;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &weights_i2_td, &scale_desc, &zp_desc}, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

namespace {

absl::Status ConvGeneric3DTest(TestExecutionEnvironment& exec_env,
                               Convolution3DAttributes& attr,
                               const Tensor5DFloat32& src_tensor,
                               const OperationDef& op_def,
                               CalculationsPrecision precision) {
  attr.weights.data.resize(attr.weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  Tensor5DFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  auto operation = CreateConvGeneric3D(exec_env.GetGpuInfo(), op_def, precision,
                                       attr, &dst_ref_tensor.shape);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  Tensor5DFloat32 dst_tensor;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvGeneric>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace

absl::Status ConvGeneric3d1x1x1BigTest(TestExecutionEnvironment& env,
                                       CalculationsPrecision precision,
                                       TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 17;
  const int dst_channels = 21;
  Convolution3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, 0);
  attr.padding.appended = HWD(0, 0, 0);
  attr.strides = HWD(1, 1, 1);
  attr.dilations = HWD(1, 1, 1);
  attr.weights =
      MakeSyntheticTensor(OHWDI(dst_channels, 1, 1, 1, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(1, 15, 12, 13, src_channels);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return ConvGeneric3DTest(env, attr, src_tensor, op_def, precision);
}

absl::Status ConvGeneric3d1x1x1BatchedBigTest(TestExecutionEnvironment& env,
                                              CalculationsPrecision precision,
                                              TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 13;
  const int dst_channels = 19;
  Convolution3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, 0);
  attr.padding.appended = HWD(0, 0, 0);
  attr.strides = HWD(1, 1, 1);
  attr.dilations = HWD(1, 1, 1);
  attr.weights =
      MakeSyntheticTensor(OHWDI(dst_channels, 1, 1, 1, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(3, 15, 12, 13, src_channels);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return ConvGeneric3DTest(env, attr, src_tensor, op_def, precision);
}

absl::Status ConvGeneric3dBigTest(TestExecutionEnvironment& env,
                                  CalculationsPrecision precision,
                                  TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 9;
  const int dst_channels = 23;
  Convolution3DAttributes attr;
  attr.padding.prepended = HWD(-1, 1, 2);
  attr.padding.appended = HWD(2, -1, 0);
  attr.strides = HWD(2, 1, 2);
  attr.dilations = HWD(2, 3, 1);
  attr.weights =
      MakeSyntheticTensor(OHWDI(dst_channels, 3, 2, 4, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(1, 15, 12, 13, src_channels);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return ConvGeneric3DTest(env, attr, src_tensor, op_def, precision);
}

absl::Status ConvGeneric3dBatchedBigTest(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int src_channels = 11;
  const int dst_channels = 17;
  Convolution3DAttributes attr;
  attr.padding.prepended = HWD(1, 0, 1);
  attr.padding.appended = HWD(-2, 0, 1);
  attr.strides = HWD(1, 1, 2);
  attr.dilations = HWD(1, 1, 2);
  attr.weights =
      MakeSyntheticTensor(OHWDI(dst_channels, 2, 1, 2, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWDC(4, 15, 12, 13, src_channels);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return ConvGeneric3DTest(env, attr, src_tensor, op_def, precision);
}

namespace {
absl::Status ConvolutionGenericInt8Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i8);

  auto operation =
      CreateConvGenericInt8(exec_env.GetGpuInfo(), op_def, quantized_type,
                            weights_i8, &dst_ref_tensor.shape);

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
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ConvolutionGenericInt8Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::UINT8>& src_tensor_ui8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const int src_sum_scale =
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo()) ? 128 : 0;
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_ui8, weights_i8, src_sum_scale);

  auto operation =
      CreateConvGenericInt8(exec_env.GetGpuInfo(), op_def, quantized_type,
                            weights_i8, &dst_ref_tensor.shape);

  TensorDescriptor src_td = op_def.src_tensors[0];
  if (quantized_type == PackedType::kUint8W4C4) {
    src_td.SetBHWCShape(BHWC(src_tensor_ui8.shape.b, src_tensor_ui8.shape.h,
                             DivideRoundUp(src_tensor_ui8.shape.w, 4),
                             src_tensor_ui8.shape.c));
    TensorDescriptor src_desc_temp = {DataType::UINT8, src_td.GetStorageType(),
                                      src_td.GetLayout()};
    src_desc_temp.UploadData(src_tensor_ui8);
    src_td.UploadDataRaw(absl::MakeConstSpan(src_desc_temp.GetData()));
  } else if (quantized_type == PackedType::kUint8C16) {
    const BHWC packed_shape =
        BHWC(src_tensor_ui8.shape.b, src_tensor_ui8.shape.h,
             src_tensor_ui8.shape.w, DivideRoundUp(src_tensor_ui8.shape.c, 4));
    ml_drift::Tensor<BHWC, DataType::UINT32> src_tensor_ui32;
    src_tensor_ui32.shape = packed_shape;
    src_tensor_ui32.data.resize(packed_shape.DimensionsProduct());
    uint32_t* dst_ptr = src_tensor_ui32.data.data();
    const uint8_t* src_ptr = src_tensor_ui8.data.data();
    for (int b = 0; b < packed_shape.b; ++b) {
      for (int h = 0; h < packed_shape.h; ++h) {
        for (int w = 0; w < packed_shape.w; ++w) {
          for (int c = 0; c < packed_shape.c; ++c) {
            uint32_t v0 = c * 4 >= src_tensor_ui8.shape.c ? 0 :
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c * 4})];
            uint32_t v1 = c * 4 + 1 >= src_tensor_ui8.shape.c ? 0 :
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c * 4 + 1})];
            uint32_t v2 = c * 4 + 2 >= src_tensor_ui8.shape.c ? 0 :
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c * 4 + 2})];
            uint32_t v3 = c * 4 + 3 >= src_tensor_ui8.shape.c ? 0 :
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c * 4 + 3})];
            uint32_t value = (v3 << 24) | (v2 << 16) | (v1 << 8) | (v0);
            dst_ptr[packed_shape.LinearIndex({b, h, w, c})] = value;
          }
        }
      }
    }
    src_td.UploadData(src_tensor_ui32);
  } else {
    src_td.UploadData(src_tensor_ui8);
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {&src_td}, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGenericInt8BigTest(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage) {
  const int src_channels =
      61;  // src_slices = 16,
           // divisible by 4, for using with kUint8C16/kInt8C16
           // divisible by 8 for triggering Intel wave matmul
  const int dst_channels = 127;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  auto src_shape = BHWC(1, 17, 28, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  const PackedType quantized_type =
      GetConvGenericInt8SrcType(env.GetGpuInfo(), src_shape);
  std::vector<PackedType> quantized_types = {quantized_type};
  if (quantized_type == PackedType::kUint8W4C4) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8W4C4) {
    quantized_types.push_back(PackedType::kInt8C4);
  } else if (quantized_type == PackedType::kUint8C16) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8C16) {
    quantized_types.push_back(PackedType::kInt8C4);
  }

  if (IsSigned(quantized_type)) {
    Tensor<BHWC, DataType::INT8> src_i8_tensor;
    src_i8_tensor.shape = src_shape;
    src_i8_tensor.data.resize(src_shape.DimensionsProduct());
    for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
      src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
    }
    for (const auto quantized_type : quantized_types) {
      const DataType src_data_type = ToSpatialTensorType(quantized_type);
      const DataType dst_data_type = DataType::INT32;
      if (!env.IsStorageSupported(src_storage, src_data_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: " << ToString(src_storage)
                       << " data type: " << ToString(src_data_type);
        continue;
      }
      OperationDef op_def;
      op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
      RETURN_IF_ERROR(ConvolutionGenericInt8Test(
          env, src_i8_tensor, quantized_type, weights_i8, op_def));
    }
  } else {
    Tensor<BHWC, DataType::UINT8> src_ui8_tensor;
    src_ui8_tensor.shape = src_shape;
    src_ui8_tensor.data.resize(src_shape.DimensionsProduct());
    for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
      src_ui8_tensor.data[i] = (src_tensor.data[i] + 1.0f) * 127.0f;
    }
    for (const auto quantized_type : quantized_types) {
      const DataType src_data_type = ToSpatialTensorType(quantized_type);
      if (!env.IsStorageSupported(src_storage, src_data_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: " << ToString(src_storage)
                       << " data type: " << ToString(src_data_type);
        continue;
      }
      const DataType dst_data_type = DataType::INT32;
      OperationDef op_def;
      op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
      RETURN_IF_ERROR(ConvolutionGenericInt8Test(
          env, src_ui8_tensor, quantized_type, weights_i8, op_def));
    }
  }
  return absl::OkStatus();
}

namespace {
absl::Status ConvolutionGenericInt8ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i8);

  auto operation = CreateConvGenericInt8ExternalWeights(
      exec_env.GetGpuInfo(), op_def, quantized_type, weights_i8.shape,
      &dst_ref_tensor.shape);

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
      {&src_td, &weights_i8_td}, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ConvolutionGenericInt8ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::UINT8>& src_tensor_ui8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const int src_sum_scale =
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo()) ? 128 : 0;
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_ui8, weights_i8, src_sum_scale);

  auto operation = CreateConvGenericInt8ExternalWeights(
      exec_env.GetGpuInfo(), op_def, quantized_type, weights_i8.shape,
      &dst_ref_tensor.shape);

  TensorDescriptor weights_i8_td;
  if (UseUint8MathForInt8Weights(exec_env.GetGpuInfo())) {
    WeightsDescription weights_desc = operation.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape));
    RearrangeWeightsInt8AsUint8(weights_i8, weights_desc, absl::MakeSpan(data),
                                128, 0u);
    weights_i8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i8_td.UploadDataRaw(absl::MakeConstSpan(data));
  } else {
    WeightsDescription weights_desc = operation.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape));
    RearrangeWeights(weights_i8, weights_desc, absl::MakeSpan(data));

    weights_i8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i8_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor src_td = op_def.src_tensors[0];
  if (quantized_type == PackedType::kUint8W4C4) {
    src_td.SetBHWCShape(BHWC(src_tensor_ui8.shape.b, src_tensor_ui8.shape.h,
                             DivideRoundUp(src_tensor_ui8.shape.w, 4),
                             src_tensor_ui8.shape.c));
    TensorDescriptor src_desc_temp = {DataType::UINT8, src_td.GetStorageType(),
                                      src_td.GetLayout()};
    src_desc_temp.UploadData(src_tensor_ui8);
    src_td.UploadDataRaw(absl::MakeConstSpan(src_desc_temp.GetData()));
  } else if (quantized_type == PackedType::kUint8C16) {
    const BHWC packed_shape =
        BHWC(src_tensor_ui8.shape.b, src_tensor_ui8.shape.h,
             src_tensor_ui8.shape.w, DivideRoundUp(src_tensor_ui8.shape.c, 4));
    ml_drift::Tensor<BHWC, DataType::UINT32> src_tensor_ui32;
    src_tensor_ui32.shape = packed_shape;
    src_tensor_ui32.data.resize(packed_shape.DimensionsProduct());
    uint32_t* dst_ptr = src_tensor_ui32.data.data();
    const uint8_t* src_ptr = src_tensor_ui8.data.data();
    for (int b = 0; b < packed_shape.b; ++b) {
      for (int h = 0; h < packed_shape.h; ++h) {
        for (int w = 0; w < packed_shape.w; ++w) {
          for (int c = 0; c < packed_shape.c; ++c) {
            int c0 = std::min(c * 4, src_tensor_ui8.shape.c - 1);
            int c1 = std::min(c * 4 + 1, src_tensor_ui8.shape.c - 1);
            int c2 = std::min(c * 4 + 2, src_tensor_ui8.shape.c - 1);
            int c3 = std::min(c * 4 + 3, src_tensor_ui8.shape.c - 1);
            uint32_t v0 =
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c0})];
            uint32_t v1 =
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c1})];
            uint32_t v2 =
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c2})];
            uint32_t v3 =
                src_ptr[src_tensor_ui8.shape.LinearIndex({b, h, w, c3})];
            uint32_t value = (v3 << 24) | (v2 << 16) | (v1 << 8) | (v0);
            dst_ptr[packed_shape.LinearIndex({b, h, w, c})] = value;
          }
        }
      }
    }
    src_td.UploadData(src_tensor_ui32);
  } else {
    src_td.UploadData(src_tensor_ui8);
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td}, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGenericInt8ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage) {
  const int src_channels =
      61;  // src_slices = 16,
           // divisible by 4, for using with kUint8C16/kInt8C16
           // divisible by 8 for triggering Intel wave matmul
  const int dst_channels = 127;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  auto src_shape = BHWC(1, 17, 20, src_channels);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  const PackedType quantized_type =
      GetConvGenericInt8SrcType(env.GetGpuInfo(), src_shape);
  std::vector<PackedType> quantized_types = {quantized_type};
  if (quantized_type == PackedType::kUint8W4C4) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8W4C4) {
    quantized_types.push_back(PackedType::kInt8C4);
  } else if (quantized_type == PackedType::kUint8C16) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8C16) {
    quantized_types.push_back(PackedType::kInt8C4);
  }

  if (IsSigned(quantized_type)) {
    Tensor<BHWC, DataType::INT8> src_i8_tensor;
    src_i8_tensor.shape = src_shape;
    src_i8_tensor.data.resize(src_shape.DimensionsProduct());
    for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
      src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
    }
    for (const auto quantized_type : quantized_types) {
      const DataType src_data_type = ToSpatialTensorType(quantized_type);
      const DataType dst_data_type = DataType::INT32;
      if (!env.IsStorageSupported(src_storage, src_data_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: " << ToString(src_storage)
                       << " data type: " << ToString(src_data_type);
        continue;
      }
      OperationDef op_def;
      op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
      MLD_EXPECT_OK(ConvolutionGenericInt8ExternalWeightsTest(
          env, src_i8_tensor, quantized_type, weights_i8, op_def));
    }
  } else {
    Tensor<BHWC, DataType::UINT8> src_ui8_tensor;
    src_ui8_tensor.shape = src_shape;
    src_ui8_tensor.data.resize(src_shape.DimensionsProduct());
    for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
      src_ui8_tensor.data[i] = (src_tensor.data[i] + 1.0f) * 127.0f;
    }
    for (const auto quantized_type : quantized_types) {
      const DataType src_data_type = ToSpatialTensorType(quantized_type);
      const DataType dst_data_type = DataType::INT32;
      if (!env.IsStorageSupported(src_storage, src_data_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: " << ToString(src_storage)
                       << " data type: " << ToString(src_data_type);
        continue;
      }
      OperationDef op_def;
      op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
      MLD_EXPECT_OK(ConvolutionGenericInt8ExternalWeightsTest(
          env, src_ui8_tensor, quantized_type, weights_i8, op_def));
    }
  }
  return absl::OkStatus();
}

namespace {

absl::Status ConvGenericInt8WithSrcQuantizationTest(
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
  const bool need_src_sum = weights_zero_point != nullptr ||
                            UseUint8MathForInt8Weights(exec_env.GetGpuInfo());
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
  const DataType dst_conv_type =
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo()) ? DataType::UINT32
                                                        : DataType::INT32;
  conv_def.dst_tensors.push_back(
      {dst_conv_type, float_storage_type, Layout::HWC});
  auto conv_op = CreateConvGenericInt8ExternalWeights(
      exec_env.GetGpuInfo(), conv_def, quantized_type, weights_i8.shape,
      &dst_ref_tensor.shape);

  TensorDescriptor dequant_dst = {float_type, float_storage_type, Layout::HWC};
  auto dequant_op = CreateDequantization(
      weights_i8.shape, exec_env.GetGpuInfo(), conv_def.dst_tensors[0],
      dequant_dst, src_params_td, weights_sum_i_td, weights_scale_td,
      weights_zero_point_td_ptr);
  MLD_EXPECT_OK(conv_op.AddOperation(exec_env.GetGpuInfo(), &dequant_op));

  TensorDescriptor weights_i8_td;
  if (UseUint8MathForInt8Weights(exec_env.GetGpuInfo())) {
    WeightsDescription weights_desc = conv_op.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape));
    RearrangeWeightsInt8AsUint8(weights_i8, weights_desc, absl::MakeSpan(data),
                                128, 0u);
    weights_i8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i8_td.UploadDataRaw(absl::MakeConstSpan(data));
  } else {
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
      src_descs, {&dst_td}, std::make_unique<ConvGeneric>(std::move(conv_op))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  int mads_count = weights_i8.shape.i;
  float eps = 0.01f * mads_count;
  if (weights_zero_point != nullptr ||
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo())) {
    eps *= 2.0f;
  }
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGenericInt8WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage) {
  const int src_channels =
      61;  // src_slices = 16,
           // divisible by 4, for using with kUint8C16/kInt8C16
           // divisible by 8 for triggering Intel wave matmul
  const int dst_channels = 127;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }
  auto weights_scale = MakeSyntheticTensor(Linear(dst_channels));
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] /= 127.0f;
  }
  auto weights_zp = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 17, 13, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 2.0f + 2.5f;
  }

  const PackedType quantized_type =
      GetConvGenericInt8SrcType(env.GetGpuInfo(), src_shape);
  std::vector<PackedType> quantized_types = {quantized_type};
  if (quantized_type == PackedType::kUint8W4C4) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8W4C4) {
    quantized_types.push_back(PackedType::kInt8C4);
  } else if (quantized_type == PackedType::kUint8C16) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8C16) {
    quantized_types.push_back(PackedType::kInt8C4);
  }

  for (const PackedType quantized_type : quantized_types) {
    for (const DataType float_type : {DataType::FLOAT32, DataType::FLOAT16}) {
      const DataType quantized_spatial_type =
          ToSpatialTensorType(quantized_type);
      if (!env.IsStorageSupported(quantized_storage, quantized_spatial_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: "
                       << ToString(quantized_storage)
                       << " data type: " << ToString(quantized_spatial_type);
        continue;
      }
      if (!env.IsStorageSupported(float_storage, float_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: "
                       << ToString(float_storage)
                       << " data type: " << ToString(float_type);
        continue;
      }
      MLD_EXPECT_OK(ConvGenericInt8WithSrcQuantizationTest(
          env, src_tensor, weights_i8, weights_scale,
          /*weights_zero_point=*/nullptr, quantized_type, quantized_storage,
          float_type, float_storage));
      MLD_EXPECT_OK(ConvGenericInt8WithSrcQuantizationTest(
          env, src_tensor, weights_i8, weights_scale, &weights_zp,
          quantized_type, quantized_storage, float_type, float_storage));
    }
  }
  return absl::OkStatus();
}

namespace {
absl::Status ConvGenericInt8WeightsInt4WithSrcQuantizationBig(
    TestExecutionEnvironment& exec_env,
    const ml_drift::TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>* weights_zero_point,
    const PackedType quantized_type,
    const TensorStorageType quantized_storage_type, const DataType float_type,
    const TensorStorageType float_storage_type) {
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i4, weights_scale, weights_zero_point);
  attr.bias = MakeZeroTensor(Linear(weights_i4.shape.o));
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  BHWC quantized_shape =
      GetShapeForPackedType(src_tensor.shape, quantized_type);
  TensorDescriptor src_params_td;
  TensorDescriptor src_quantized_td;
  const bool need_src_sum = weights_zero_point != nullptr ||
                            UseUint8MathForInt8Weights(exec_env.GetGpuInfo());
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

  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i4);
  auto weights_sum_i_td = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), weights_sum_i);

  OperationDef conv_def;
  conv_def.src_tensors.push_back({ToSpatialTensorType(quantized_type),
                                  quantized_storage_type, Layout::HWC});
  const DataType dst_conv_type =
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo()) ? DataType::UINT32
                                                        : DataType::INT32;
  conv_def.dst_tensors.push_back(
      {dst_conv_type, float_storage_type, Layout::HWC});
  auto conv_op = CreateConvGenericInt8ExternalWeights(
      exec_env.GetGpuInfo(), conv_def, quantized_type, weights_i4.shape,
      &dst_ref_tensor.shape);

  TensorDescriptor dequant_dst = {float_type, float_storage_type, Layout::HWC};
  auto dequant_op = CreateDequantization(
      weights_i4.shape, exec_env.GetGpuInfo(), conv_def.dst_tensors[0],
      dequant_dst, src_params_td, weights_sum_i_td, weights_scale_td,
      weights_zero_point_td_ptr);
  MLD_EXPECT_OK(conv_op.AddOperation(exec_env.GetGpuInfo(), &dequant_op));

  TensorDescriptor weights_i4_td;
  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT4;
  src_weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  src_weights_desc.output_group_size = DivideRoundUp(weights_i4.shape.o, 4);
  {
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(src_weights_desc, weights_i4.shape) / 2);
    RearrangeWeightsInt8AsUint4(weights_i4, src_weights_desc,
                                absl::MakeSpan(data), 8, 8u);

    weights_i4_td = TensorDescriptor(DataType::UINT8, TensorStorageType::BUFFER,
                                     Layout::LINEAR);
    weights_i4_td.SetBHWCShape(BHWC(1, 1, 1, data.size()));
    weights_i4_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor weights_i8_td;
  {
    WeightsDescription dst_weights_desc = conv_op.GetWeightsDescription();
    OperationDef op_def;
    op_def.src_tensors.push_back(
        {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
    op_def.dst_tensors.push_back(
        {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
    WeightsConverter uint4_converter(exec_env.GetGpuInfo(), op_def,
                                     weights_i4.shape, src_weights_desc,
                                     dst_weights_desc);

    weights_i8_td = TensorDescriptor(dst_weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWCShape(BHWC(
        1, 1, 1,
        GetTotalElementsCountForLayout(dst_weights_desc, weights_i4.shape)));

    MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
        {&weights_i4_td}, {&weights_i8_td},
        std::make_unique<WeightsConverter>(std::move(uint4_converter))));
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
      src_descs, {&dst_td}, std::make_unique<ConvGeneric>(std::move(conv_op))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  int mads_count = weights_i4.shape.i;
  float eps = 0.01f * mads_count;
  if (weights_zero_point != nullptr ||
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo())) {
    eps *= 4.0f;
  }
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGenericInt8WeightsInt4WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage) {
  const int src_channels =
      61;  // src_slices = 16,
           // divisible by 4, for using with kUint8C16/kInt8C16
           // divisible by 8 for triggering Intel wave matmul
  const int dst_channels = 127;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scale = MakeSyntheticTensor(Linear(dst_channels));
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] /= 8.0f;
  }
  auto weights_zp = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 17, 13, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 2.0f + 2.5f;
  }

  const PackedType quantized_type =
      GetConvGenericInt8SrcType(env.GetGpuInfo(), src_shape);
  std::vector<PackedType> quantized_types = {quantized_type};
  if (quantized_type == PackedType::kUint8W4C4) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8W4C4) {
    quantized_types.push_back(PackedType::kInt8C4);
  } else if (quantized_type == PackedType::kUint8C16) {
    quantized_types.push_back(PackedType::kUint8C4);
  } else if (quantized_type == PackedType::kInt8C16) {
    quantized_types.push_back(PackedType::kInt8C4);
  }

  for (const PackedType quantized_type : quantized_types) {
    for (const DataType float_type : {DataType::FLOAT32, DataType::FLOAT16}) {
      const DataType quantized_spatial_type =
          ToSpatialTensorType(quantized_type);
      if (!env.IsStorageSupported(quantized_storage, quantized_spatial_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: "
                       << ToString(quantized_storage)
                       << " data type: " << ToString(quantized_spatial_type);
        continue;
      }
      if (!env.IsStorageSupported(float_storage, float_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: "
                       << ToString(float_storage)
                       << " data type: " << ToString(float_type);
        continue;
      }
      RETURN_IF_ERROR(ConvGenericInt8WeightsInt4WithSrcQuantizationBig(
          env, src_tensor, weights_i4, weights_scale,
          /*weights_zero_point=*/nullptr, quantized_type, quantized_storage,
          float_type, float_storage));
      RETURN_IF_ERROR(ConvGenericInt8WeightsInt4WithSrcQuantizationBig(
          env, src_tensor, weights_i4, weights_scale, &weights_zp,
          quantized_type, quantized_storage, float_type, float_storage));
    }
  }
  return absl::OkStatus();
}

namespace {
absl::Status ConvolutionGenericInt4Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i4,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i4, weights_i4);

  auto operation = CreateConvGenericInt4(exec_env.GetGpuInfo(), op_def,
                                         weights_i4, &dst_ref_tensor.shape);

  TensorDescriptor src_td = op_def.src_tensors[0];
  if (quantized_type == PackedType::kInt4C32) {
    const BHWC packed_shape =
        BHWC(src_tensor_i4.shape.b, src_tensor_i4.shape.h,
             src_tensor_i4.shape.w, DivideRoundUp(src_tensor_i4.shape.c, 8));
    ml_drift::Tensor<BHWC, DataType::INT32> src_tensor_i32;
    src_tensor_i32.shape = packed_shape;
    src_tensor_i32.data.resize(packed_shape.DimensionsProduct());
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(src_tensor_i32.data.data());
    const int8_t* src_ptr =
        reinterpret_cast<const int8_t*>(src_tensor_i4.data.data());
    for (int b = 0; b < packed_shape.b; ++b) {
      for (int h = 0; h < packed_shape.h; ++h) {
        for (int w = 0; w < packed_shape.w; ++w) {
          for (int c = 0; c < packed_shape.c; ++c) {
            int c0 = std::min(c * 8 + 0, src_tensor_i4.shape.c - 1);
            int c1 = std::min(c * 8 + 1, src_tensor_i4.shape.c - 1);
            int c2 = std::min(c * 8 + 2, src_tensor_i4.shape.c - 1);
            int c3 = std::min(c * 8 + 3, src_tensor_i4.shape.c - 1);
            int c4 = std::min(c * 8 + 4, src_tensor_i4.shape.c - 1);
            int c5 = std::min(c * 8 + 5, src_tensor_i4.shape.c - 1);
            int c6 = std::min(c * 8 + 6, src_tensor_i4.shape.c - 1);
            int c7 = std::min(c * 8 + 7, src_tensor_i4.shape.c - 1);
            int32_t iv0 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c0})];
            int32_t iv1 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c1})];
            int32_t iv2 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c2})];
            int32_t iv3 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c3})];
            int32_t iv4 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c4})];
            int32_t iv5 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c5})];
            int32_t iv6 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c6})];
            int32_t iv7 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c7})];
            uint32_t v0 = iv0 >= 0 ? iv0 : 16 + iv0;
            uint32_t v1 = iv1 >= 0 ? iv1 : 16 + iv1;
            uint32_t v2 = iv2 >= 0 ? iv2 : 16 + iv2;
            uint32_t v3 = iv3 >= 0 ? iv3 : 16 + iv3;
            uint32_t v4 = iv4 >= 0 ? iv4 : 16 + iv4;
            uint32_t v5 = iv5 >= 0 ? iv5 : 16 + iv5;
            uint32_t v6 = iv6 >= 0 ? iv6 : 16 + iv6;
            uint32_t v7 = iv7 >= 0 ? iv7 : 16 + iv7;
            int32_t value = (v7 << 28) | (v6 << 24) | (v5 << 20) | (v4 << 16) |
                            (v3 << 12) | (v2 << 8) | (v1 << 4) | (v0);
            dst_ptr[packed_shape.LinearIndex({b, h, w, c})] = value;
          }
        }
      }
    }
    src_td.UploadData(src_tensor_i32);
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {&src_td}, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGenericInt4BigTest(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage,
                                    const BHWC& src_shape) {
  const int dst_channels = 71;
  auto weights_f32 = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_shape.c));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    weights_i4.data[i] = weights_f32.data[i] * 7.0f;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  const PackedType quantized_type =
      GetConvGenericInt4SrcType(env.GetGpuInfo(), src_shape);
  std::vector<PackedType> quantized_types = {quantized_type};

  if (IsSigned(quantized_type)) {
    Tensor<BHWC, DataType::INT8> src_i4_tensor;
    src_i4_tensor.shape = src_shape;
    src_i4_tensor.data.resize(src_shape.DimensionsProduct());
    for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
      src_i4_tensor.data[i] = src_tensor.data[i] * 7.0f;
    }
    for (const auto quantized_type : quantized_types) {
      const DataType src_data_type = ToSpatialTensorType(quantized_type);
      const DataType dst_data_type = DataType::INT32;
      if (!env.IsStorageSupported(src_storage, src_data_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: " << ToString(src_storage)
                       << " data type: " << ToString(src_data_type);
        continue;
      }
      OperationDef op_def;
      op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
      MLD_EXPECT_OK(ConvolutionGenericInt4Test(env, src_i4_tensor, quantized_type,
                                           weights_i4, op_def));
    }
  }
  return absl::OkStatus();
}

namespace {
absl::Status ConvolutionGenericInt4ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i4,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i4, weights_i4);

  auto operation = CreateConvGenericInt4ExternalWeights(
      exec_env.GetGpuInfo(), op_def, weights_i4.shape, &dst_ref_tensor.shape);

  TensorDescriptor weights_i4_td;
  {
    WeightsDescription weights_desc = operation.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i4.shape) / 2);
    RearrangeWeightsInt4(weights_i4, weights_desc, absl::MakeSpan(data));

    weights_i4_td = TensorDescriptor(DataType::INT8, TensorStorageType::BUFFER,
                                     Layout::LINEAR);
    weights_i4_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i4_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor src_td = op_def.src_tensors[0];
  if (quantized_type == PackedType::kInt4C32) {
    const BHWC packed_shape =
        BHWC(src_tensor_i4.shape.b, src_tensor_i4.shape.h,
             src_tensor_i4.shape.w, DivideRoundUp(src_tensor_i4.shape.c, 8));
    ml_drift::Tensor<BHWC, DataType::INT32> src_tensor_i32;
    src_tensor_i32.shape = packed_shape;
    src_tensor_i32.data.resize(packed_shape.DimensionsProduct());
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(src_tensor_i32.data.data());
    const int8_t* src_ptr =
        reinterpret_cast<const int8_t*>(src_tensor_i4.data.data());
    for (int b = 0; b < packed_shape.b; ++b) {
      for (int h = 0; h < packed_shape.h; ++h) {
        for (int w = 0; w < packed_shape.w; ++w) {
          for (int c = 0; c < packed_shape.c; ++c) {
            int c0 = std::min(c * 8 + 0, src_tensor_i4.shape.c - 1);
            int c1 = std::min(c * 8 + 1, src_tensor_i4.shape.c - 1);
            int c2 = std::min(c * 8 + 2, src_tensor_i4.shape.c - 1);
            int c3 = std::min(c * 8 + 3, src_tensor_i4.shape.c - 1);
            int c4 = std::min(c * 8 + 4, src_tensor_i4.shape.c - 1);
            int c5 = std::min(c * 8 + 5, src_tensor_i4.shape.c - 1);
            int c6 = std::min(c * 8 + 6, src_tensor_i4.shape.c - 1);
            int c7 = std::min(c * 8 + 7, src_tensor_i4.shape.c - 1);
            int32_t iv0 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c0})];
            int32_t iv1 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c1})];
            int32_t iv2 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c2})];
            int32_t iv3 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c3})];
            int32_t iv4 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c4})];
            int32_t iv5 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c5})];
            int32_t iv6 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c6})];
            int32_t iv7 =
                src_ptr[src_tensor_i4.shape.LinearIndex({b, h, w, c7})];
            uint32_t v0 = iv0 >= 0 ? iv0 : 16 + iv0;
            uint32_t v1 = iv1 >= 0 ? iv1 : 16 + iv1;
            uint32_t v2 = iv2 >= 0 ? iv2 : 16 + iv2;
            uint32_t v3 = iv3 >= 0 ? iv3 : 16 + iv3;
            uint32_t v4 = iv4 >= 0 ? iv4 : 16 + iv4;
            uint32_t v5 = iv5 >= 0 ? iv5 : 16 + iv5;
            uint32_t v6 = iv6 >= 0 ? iv6 : 16 + iv6;
            uint32_t v7 = iv7 >= 0 ? iv7 : 16 + iv7;
            int32_t value = (v7 << 28) | (v6 << 24) | (v5 << 20) | (v4 << 16) |
                            (v3 << 12) | (v2 << 8) | (v1 << 4) | (v0);
            dst_ptr[packed_shape.LinearIndex({b, h, w, c})] = value;
          }
        }
      }
    }
    src_td.UploadData(src_tensor_i32);
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  std::vector<TensorDescriptor*> src_descs = {&src_td, &weights_i4_td};
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_descs, {&dst_td},
      std::make_unique<ConvGeneric>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGenericInt4ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage, const BHWC& src_shape) {
  const int dst_channels = 71;
  auto weights_f32 = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_shape.c));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    weights_i4.data[i] = weights_f32.data[i] * 7.0f;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  const PackedType quantized_type =
      GetConvGenericInt4SrcType(env.GetGpuInfo(), src_shape);
  std::vector<PackedType> quantized_types = {quantized_type};

  if (IsSigned(quantized_type)) {
    Tensor<BHWC, DataType::INT8> src_i4_tensor;
    src_i4_tensor.shape = src_shape;
    src_i4_tensor.data.resize(src_shape.DimensionsProduct());
    for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
      src_i4_tensor.data[i] = src_tensor.data[i] * 7.0f;
    }
    for (const auto quantized_type : quantized_types) {
      const DataType src_data_type = ToSpatialTensorType(quantized_type);
      const DataType dst_data_type = DataType::INT32;
      if (!env.IsStorageSupported(src_storage, src_data_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: " << ToString(src_storage)
                       << " data type: " << ToString(src_data_type);
        continue;
      }
      OperationDef op_def;
      op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
      op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
      MLD_EXPECT_OK(ConvolutionGenericInt4ExternalWeightsTest(
          env, src_i4_tensor, quantized_type, weights_i4, op_def));
    }
  }
  return absl::OkStatus();
}

namespace {
absl::Status ConvGenericInt4WithSrcQuantizationTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i4,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>* weights_zero_point,
    const PackedType quantized_type,
    const TensorStorageType quantized_storage_type, const DataType float_type,
    const TensorStorageType float_storage_type) {
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i4, weights_scale, weights_zero_point);
  attr.bias = MakeZeroTensor(Linear(weights_i4.shape.o));
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
    op_def.dst_tensors.push_back(
        {DataType::FLOAT32, float_storage_type, Layout::HWC});
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

  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i4);
  auto weights_sum_i_td = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), weights_sum_i);

  OperationDef conv_def;
  conv_def.src_tensors.push_back({ToSpatialTensorType(quantized_type),
                                  quantized_storage_type, Layout::HWC});
  conv_def.dst_tensors.push_back(
      {DataType::INT32, quantized_storage_type, Layout::HWC});
  auto conv_op = CreateConvGenericInt4ExternalWeights(
      exec_env.GetGpuInfo(), conv_def, weights_i4.shape, &dst_ref_tensor.shape);

  TensorDescriptor dequant_dst = {float_type, float_storage_type, Layout::HWC};
  auto dequant_op = CreateDequantization(
      weights_i4.shape, exec_env.GetGpuInfo(), conv_def.dst_tensors[0],
      dequant_dst, src_params_td, weights_sum_i_td, weights_scale_td,
      weights_zero_point_td_ptr);
  MLD_EXPECT_OK(conv_op.AddOperation(exec_env.GetGpuInfo(), &dequant_op));

  TensorDescriptor weights_i4_td;
  {
    WeightsDescription weights_desc = conv_op.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i4.shape) / 2);
    RearrangeWeightsInt4(weights_i4, weights_desc, absl::MakeSpan(data));

    weights_i4_td = TensorDescriptor(DataType::INT8, TensorStorageType::BUFFER,
                                     Layout::LINEAR);
    weights_i4_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i4_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor dst_td = dequant_dst;
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  std::vector<TensorDescriptor*> src_descs = {&src_quantized_td, &weights_i4_td,
                                              &src_params_td, &weights_sum_i_td,
                                              &weights_scale_td};
  if (weights_zero_point != nullptr) {
    src_descs.push_back(&weights_zero_point_td);
  }
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_descs, {&dst_td}, std::make_unique<ConvGeneric>(std::move(conv_op))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  int mads_count = weights_i4.shape.i;
  float eps = 0.04f * mads_count;
  if (weights_zero_point != nullptr) {
    eps *= 2.0f;
  }
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvGenericInt4WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType quantized_storage,
    TensorStorageType float_storage, const BHWC& src_shape) {
  const int dst_channels = 128;

  auto weights_f32 = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_shape.c));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    const int val = (weights_f32.data[i] + 1.0f) * 16.0f;
    weights_i4.data[i] = std::max(std::min(val, 15), 0) - 8;
  }
  auto weights_scale = MakeSyntheticTensor(Linear(dst_channels));
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] /= 8.0f;
  }
  auto weights_zp = MakeSyntheticTensor(Linear(dst_channels));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 2.0f + 0.5f;
  }

  const PackedType quantized_type =
      GetConvGenericInt4SrcType(env.GetGpuInfo(), src_shape);
  std::vector<PackedType> quantized_types = {quantized_type};

  for (const PackedType quantized_type : quantized_types) {
    const DataType quantized_spatial_type = ToSpatialTensorType(quantized_type);
    for (const DataType float_type : {DataType::FLOAT32, DataType::FLOAT16}) {
      if (!env.IsStorageSupported(quantized_storage, quantized_spatial_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: "
                       << ToString(quantized_storage)
                       << " data type: " << ToString(quantized_spatial_type);
        continue;
      }
      if (!env.IsStorageSupported(float_storage, float_type)) {
        ABSL_LOG(INFO) << "Unsupported storage type: "
                       << ToString(float_storage)
                       << " data type: " << ToString(float_type);
        continue;
      }
      MLD_EXPECT_OK(ConvGenericInt4WithSrcQuantizationTest(
          env, src_tensor, weights_i4, weights_scale,
          /*weights_zero_point=*/nullptr, quantized_type, quantized_storage,
          float_type, float_storage));
      MLD_EXPECT_OK(ConvGenericInt4WithSrcQuantizationTest(
          env, src_tensor, weights_i4, weights_scale, &weights_zp,
          quantized_type, quantized_storage, float_type, float_storage));
    }
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
