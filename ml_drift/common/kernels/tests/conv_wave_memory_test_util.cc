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

#include "ml_drift/common/kernels/tests/conv_wave_memory_test_util.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_wave_memory.h"
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
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status ConvWaveMemoryTest(TestExecutionEnvironment& exec_env,
                                const Convolution2DAttributes& attr,
                                const TensorFloat32& src_tensor,
                                const OperationDef& op_def,
                                CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  auto operation =
      CreateConvWaveMemory(exec_env.GetGpuInfo(), op_def, precision, attr);
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvWaveMemory>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryExternalWeightsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    const Convolution2DAttributes& attr, const OperationDef& conv_def,
    CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);

  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      exec_env.GetGpuInfo(), DeduceDataTypeFromPrecision(precision), attr.bias);
  auto operation = CreateConvWaveMemoryExternalWeights(
      exec_env.GetGpuInfo(), conv_def, precision, attr, &bias_tensor_desc,
      /*dst_shape=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/false);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(GetFloatWeights(attr),
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

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMemory>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryExternalBatchedWeightsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    const Convolution2DAttributes& attr, const OperationDef& conv_def,
    CalculationsPrecision precision) {
  const auto& weights = GetFloatWeights(attr);
  TensorFloat32 dst_ref_tensor =
      FullyConnectedRefDifferentWeightsForHeight(weights, src_tensor);

  auto operation = CreateConvWaveMemoryExternalWeights(
      exec_env.GetGpuInfo(), conv_def, precision, attr, /*bias=*/nullptr,
      /*dst_shape=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/true);

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

  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMemory>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryWinograd4x4To6x6(TestExecutionEnvironment& exec_env,
                                            const Convolution2DAttributes& attr,
                                            const TensorFloat32& src_tensor,
                                            const OperationDef& op_def,
                                            CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor = ConvolutionReference(attr, src_tensor);
  auto wino_up =
      std::make_unique<Winograd3x3TiledXForward>(CreateWinograd3x3TiledXForward(
          exec_env.GetGpuInfo(), op_def, attr.padding, 6));
  auto wino_down = std::make_unique<Winograd3x3TiledXBackward>(
      CreateWinograd3x3TiledXBackward(exec_env.GetGpuInfo(), op_def, attr.bias,
                                      6));

  Tensor<OHWI, DataType::FLOAT32> wino_weights;
  RearrangeWeightsToWinograd3x3TileNxN(GetFloatWeights(attr), &wino_weights, 6);

  Convolution2DAttributes wino_attr;
  wino_attr.padding.prepended = HW(0, 0);
  wino_attr.padding.appended = HW(0, 0);
  wino_attr.strides = HW(1, 1);
  wino_attr.dilations = HW(1, 1);
  auto& wino_attr_weights =
      wino_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  wino_attr_weights.shape = wino_weights.shape;

  auto convolution = CreateConvWaveMemoryExternalWeights(
      exec_env.GetGpuInfo(), op_def, precision, wino_attr, /*bias=*/nullptr,
      /*dst_shape=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/true);
  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(wino_weights,
                                           convolution.GetWeightsDescription());

  const int tiles_x = DivideRoundUp(dst_ref_tensor.shape.w, 4);
  const int tiles_y = DivideRoundUp(dst_ref_tensor.shape.h, 4);
  TensorFloat32 wino_up_result;
  TensorFloat32 conv_result;
  wino_up_result.shape =
      BHWC(src_tensor.shape.b, 36, tiles_x * tiles_y, src_tensor.shape.c);
  conv_result.shape =
      BHWC(src_tensor.shape.b, 36, tiles_x * tiles_y, dst_ref_tensor.shape.c);

  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(src_tensor, std::move(wino_up),
                                               BHWC(wino_up_result.shape),
                                               &wino_up_result));
  {
    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(wino_up_result);

    std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
    srcs_td[0] = &src_td;
    for (int i = 0; i < weights_gpu.size(); ++i) {
      srcs_td[1 + i] = &weights_gpu[i];
    }
    TensorDescriptor dst_td = op_def.dst_tensors[0];
    dst_td.SetBHWCShape(BHWC(conv_result.shape));
    RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
        srcs_td, {&dst_td},
        std::make_unique<ConvWaveMemory>(std::move(convolution))));
    dst_td.DownloadData(&conv_result);
  }
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo(), attr) * 3.0f;
  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      conv_result, std::move(wino_down), dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryBatchedMatMulTest(TestExecutionEnvironment& exec_env,
                                             const TensorFloat32& left_tensor,
                                             const TensorFloat32& right_tensor,
                                             const OperationDef& conv_def,
                                             CalculationsPrecision precision) {
  TensorFloat32 dst_ref_tensor =
      BatchedMatMulReference(left_tensor, right_tensor);

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape =
      OHWI(right_tensor.shape.c, right_tensor.shape.h, 1, right_tensor.shape.w);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int b = 0; b < right_tensor.shape.b; ++b) {
    for (int h = 0; h < right_tensor.shape.h; ++h) {
      for (int w = 0; w < right_tensor.shape.w; ++w) {
        for (int c = 0; c < right_tensor.shape.c; ++c) {
          int src_index = right_tensor.shape.LinearIndex({b, h, w, c});
          int dst_index = weights.shape.LinearIndex({c, h, b, w});
          weights.data[dst_index] = right_tensor.data[src_index];
        }
      }
    }
  }

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = weights.shape;

  auto operation = CreateConvWaveMemoryExternalWeights(
      exec_env.GetGpuInfo(), conv_def, precision, attr, /*bias=*/nullptr,
      /*dst_shape=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/true);

  std::vector<TensorDescriptor> weights_gpu =
      GetTensorDescriptorsForWeightsLayout(weights,
                                           operation.GetWeightsDescription());

  TensorDescriptor src_td = conv_def.src_tensors[0];
  src_td.UploadData(left_tensor);

  std::vector<TensorDescriptor*> srcs_td(weights_gpu.size() + 1);
  srcs_td[0] = &src_td;
  for (int i = 0; i < weights_gpu.size(); ++i) {
    srcs_td[1 + i] = &weights_gpu[i];
  }

  TensorDescriptor dst_td = conv_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  int mads_count = left_tensor.shape.c;
  float eps = GetEpsilon(precision, exec_env.GetGpuInfo()) * mads_count;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMemory>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

bool UseUint8MathForInt8Weights(const GpuInfo& gpu_info) {
  return gpu_info.IsApiOpenCl() && gpu_info.IsPowerVR();
}

absl::Status ConvWaveMemoryInt8Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::UINT8>& src_tensor_ui8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const int src_sum_scale =
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo()) ? 128 : 0;
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_ui8, weights_i8, src_sum_scale);

  auto operation = CreateConvWaveMemoryInt8(exec_env.GetGpuInfo(), op_def,
                                            weights_i8, &dst_ref_tensor.shape);

  TensorDescriptor src_td = op_def.src_tensors[0];
  if (quantized_type == PackedType::kUint8C16) {
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
            uint32_t v0 = c * 4 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4})];
            uint32_t v1 = c * 4 + 1 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4 + 1})];
            uint32_t v2 = c * 4 + 2 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4 + 2})];
            uint32_t v3 = c * 4 + 3 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4 + 3})];
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
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td}, {&dst_td},
      std::make_unique<ConvWaveMemory>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  if (dst_tensor.data != dst_ref_tensor.data) {
    return absl::InternalError("not equal");
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryInt8ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::UINT8>& src_tensor_ui8,
    const PackedType quantized_type,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const int src_sum_scale =
      UseUint8MathForInt8Weights(exec_env.GetGpuInfo()) ? 128 : 0;
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_ui8, weights_i8, src_sum_scale);

  auto operation = CreateConvWaveMemoryInt8ExternalWeights(
      exec_env.GetGpuInfo(), op_def, weights_i8.shape, &dst_ref_tensor.shape);

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
  if (quantized_type == PackedType::kUint8C16) {
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
            uint32_t v0 = c * 4 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4})];
            uint32_t v1 = c * 4 + 1 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4 + 1})];
            uint32_t v2 = c * 4 + 2 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4 + 2})];
            uint32_t v3 = c * 4 + 3 >= src_tensor_ui8.shape.c
                              ? 0
                              : src_ptr[src_tensor_ui8.shape.LinearIndex(
                                    {b, h, w, c * 4 + 3})];
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
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td}, {&dst_td},
      std::make_unique<ConvWaveMemory>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  if (dst_tensor.data != dst_ref_tensor.data) {
    return absl::InternalError("not equal");
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryInt8WithSrcQuantizationTest(
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
  {
    OperationDef op_def;
    op_def.src_tensors.push_back({float_type, float_storage_type, Layout::HWC});
    op_def.dst_tensors.push_back({ToSpatialTensorType(quantized_type),
                                  quantized_storage_type, Layout::HWC});
    op_def.dst_tensors.push_back({float_type, float_storage_type, Layout::HWC});
    const bool need_src_sum = weights_zero_point != nullptr ||
                              UseUint8MathForInt8Weights(exec_env.GetGpuInfo());
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

    RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
        {&src_td}, {&src_quantized_td, &src_params_td},
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
  auto conv_op = CreateConvWaveMemoryInt8ExternalWeights(
      exec_env.GetGpuInfo(), conv_def, weights_i8.shape, &dst_ref_tensor.shape);

  TensorDescriptor dequant_dst = {float_type, float_storage_type, Layout::HWC};
  auto dequant_op = CreateDequantization(
      weights_i8.shape, exec_env.GetGpuInfo(), conv_def.dst_tensors[0],
      dequant_dst, src_params_td, weights_sum_i_td, weights_scale_td,
      weights_zero_point_td_ptr);
  RETURN_IF_ERROR(conv_op.AddOperation(exec_env.GetGpuInfo(), &dequant_op));

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
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_descs, {&dst_td},
      std::make_unique<ConvWaveMemory>(std::move(conv_op))));
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

absl::Status ConvWaveMemoryTest(TestExecutionEnvironment& env,
                                CalculationsPrecision precision,
                                TensorStorageType storage, const BHWC& shape,
                                int dst_channels) {
  const int src_channels = shape.c;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  TensorFloat32 src_tensor = MakeSyntheticTensor(shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
  op_def.src_tensors.push_back({data_type, storage, layout});
  op_def.dst_tensors.push_back({data_type, storage, layout});
  RETURN_IF_ERROR(ConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryGroupedX3Test(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage) {
  Convolution2DAttributes attr;
  attr.groups = 3;
  const int src_channels = 8 * attr.groups;
  const int dst_channels = 16 * attr.groups;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels / attr.groups));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 7, 4, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(ConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryGroupedX7Test(TestExecutionEnvironment& env,
                                         CalculationsPrecision precision,
                                         TensorStorageType storage) {
  Convolution2DAttributes attr;
  attr.groups = 7;
  const int src_channels = 4 * attr.groups;
  const int dst_channels = 12 * attr.groups;
  attr.padding.prepended = HW(1, 0);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 2);
  attr.dilations = HW(2, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 2, 3, src_channels / attr.groups));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 7, 4, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(ConvWaveMemoryTest(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryExternalWeightsTest(TestExecutionEnvironment& env,
                                               CalculationsPrecision precision,
                                               TensorStorageType storage) {
  auto src_shape = BHWC(1, 17, 13, 21);
  auto weights_shape = BHWC(23, 3, 2, 21);
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(2, 1);
  attr.dilations = HW(1, 2);
  auto synthetic_weights = MakeSyntheticTensor(
      OHWI(weights_shape.b, weights_shape.h, weights_shape.w, weights_shape.c));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(weights_shape.b));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.src_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(ConvWaveMemoryExternalWeightsTest(env, src_tensor, attr,
                                                    conv_def, precision));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryExternalBatchedWeightsTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  auto src_shape = BHWC(1, 6, 12, 128);
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto synthetic_weights = MakeSyntheticTensor(OHWI(64, 6, 1, src_shape.c));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(attr_weights.shape.o));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(ConvWaveMemoryExternalBatchedWeightsTest(
      env, src_tensor, attr, conv_def, precision));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryWinograd4x4To6x6Test(TestExecutionEnvironment& env,
                                                CalculationsPrecision precision,
                                                TensorStorageType storage) {
  const int src_channels = 7;
  const int dst_channels = 13;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(10, 10);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto synthetic_weights =
      MakeSyntheticTensor(OHWI(dst_channels, 3, 3, src_channels));
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>(
      std::move(synthetic_weights));
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  auto src_shape = BHWC(1, 17, 13, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(
      ConvWaveMemoryWinograd4x4To6x6(env, attr, src_tensor, op_def, precision));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryBatchedMatMulTest(TestExecutionEnvironment& env,
                                             CalculationsPrecision precision,
                                             TensorStorageType storage) {
  auto left_shape = BHWC(1, 12, 128, 32);
  auto right_shape = BHWC(1, 12, 32, 64);

  TensorFloat32 left_tensor = MakeSyntheticTensor(left_shape);
  TensorFloat32 right_tensor = MakeSyntheticTensor(right_shape);

  OperationDef conv_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  conv_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  conv_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(ConvWaveMemoryBatchedMatMulTest(
      env, left_tensor, right_tensor, conv_def, precision));
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryInt8Test(TestExecutionEnvironment& env,
                                    TensorStorageType src_storage,
                                    TensorStorageType dst_storage) {
  const int src_channels =
      45;  // src_slices = 12, divisible by 4, for using with kUint8C16
  const int dst_channels = 127;

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
  Tensor<BHWC, DataType::UINT8> src_ui8_tensor;
  src_ui8_tensor.shape = src_shape;
  src_ui8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_ui8_tensor.data[i] = (src_tensor.data[i] + 1.0f) * 127.0f;
  }

  const DataType dst_data_type = DataType::INT32;
  for (const PackedType quantized_type :
       {PackedType::kUint8C4, PackedType::kUint8C16}) {
    const DataType src_data_type = ToSpatialTensorType(quantized_type);
    if (!env.IsStorageSupported(src_storage, src_data_type)) {
      ABSL_LOG(INFO) << "Skipping unsupported src storage type: "
                     << ToString(src_storage)
                     << " data type: " << ToString(src_data_type);
      continue;
    }
    OperationDef op_def;
    op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
    op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
    RETURN_IF_ERROR(ConvWaveMemoryInt8Test(env, src_ui8_tensor, quantized_type,
                                           weights_i8, op_def));
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryInt8ExternalWeightsTest(
    TestExecutionEnvironment& env, TensorStorageType src_storage,
    TensorStorageType dst_storage) {
  const int src_channels =
      45;  // src_slices = 12, divisible by 4, for using with kUint8C16
  const int dst_channels = 127;

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
  Tensor<BHWC, DataType::UINT8> src_ui8_tensor;
  src_ui8_tensor.shape = src_shape;
  src_ui8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_ui8_tensor.data[i] = (src_tensor.data[i] + 1.0f) * 127.0f;
  }

  const DataType dst_data_type = DataType::INT32;
  for (const PackedType quantized_type :
       {PackedType::kUint8C4, PackedType::kUint8C16}) {
    const DataType src_data_type = ToSpatialTensorType(quantized_type);
    if (!env.IsStorageSupported(src_storage, src_data_type)) {
      ABSL_LOG(INFO) << "Skipping unsupported src storage type: "
                     << ToString(src_storage)
                     << " data type: " << ToString(src_data_type);
      continue;
    }
    OperationDef op_def;
    op_def.src_tensors.push_back({src_data_type, src_storage, Layout::HWC});
    op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
    RETURN_IF_ERROR(ConvWaveMemoryInt8ExternalWeightsTest(
        env, src_ui8_tensor, quantized_type, weights_i8, op_def));
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryInt8WithSrcQuantizationTest(
    TestExecutionEnvironment& env, TensorStorageType int_storage,
    TensorStorageType float_storage, DataType float_type) {
  if (float_type != DataType::FLOAT16 && float_type != DataType::FLOAT32) {
    return absl::InvalidArgumentError("Unsupported float type: " +
                                      ToString(float_type));
  }
  const int src_channels = 47;
  const int dst_channels = 127;

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
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 2.0f + 2.5f;
  }

  for (const PackedType quantized_type :
       {PackedType::kUint8C4, PackedType::kUint8C16}) {
    RETURN_IF_ERROR(ConvWaveMemoryInt8WithSrcQuantizationTest(
        env, src_tensor, weights_i8, weights_scale,
        /*weights_zero_point=*/nullptr, quantized_type, int_storage, float_type,
        float_storage));
    RETURN_IF_ERROR(ConvWaveMemoryInt8WithSrcQuantizationTest(
        env, src_tensor, weights_i8, weights_scale, &weights_zp, quantized_type,
        int_storage, float_type, float_storage));
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryRuntimeChannelsTest(
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    Convolution2DAttributes& attr, const OperationDef& conv_def,
    CalculationsPrecision precision,
    const TestingRuntimeChannels& runtime_channels) {
  auto& attr_weights = GetFloatWeights(attr);
  attr_weights.data.resize(attr_weights.data.size() +
                           XNN_EXTRA_BYTES / sizeof(float));
  TestingRuntimeChannels aligned_runtime_channels =
      runtime_channels.GenerateAlignedRuntimeChannels();
  TensorFloat32 dst_ref_tensor =
      ConvolutionReference(attr, src_tensor, aligned_runtime_channels);

  auto operation = CreateConvWaveMemoryExternalWeights(
      exec_env.GetGpuInfo(), conv_def, precision, attr,
      /*bias=*/nullptr, &dst_ref_tensor.shape,
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
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      srcs_td, {&dst_td},
      std::make_unique<ConvWaveMemory>(std::move(operation))));
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

absl::Status ConvWaveMemoryRuntimeSrcEndChannelsTest(
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

  if (!IsConvWaveMemorySupported(env.GetGpuInfo())) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  for (int src_ch = 4; src_ch <= weights_shape.i; src_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.src_end_ch = src_ch;
    MLD_EXPECT_OK(ConvWaveMemoryRuntimeChannelsTest(env, src_tensor, attr, conv_def,
                                                precision, runtime_channels));
  }
  return absl::OkStatus();
}

absl::Status ConvWaveMemoryRuntimeDstEndChannelsTest(
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

  if (!IsConvWaveMemorySupported(env.GetGpuInfo())) {
    return absl::UnimplementedError(env.SkipTestMessage());
  }

  for (int dst_ch = 0; dst_ch <= weights_shape.o; dst_ch += 4) {
    TestingRuntimeChannels runtime_channels;
    runtime_channels.dst_end_ch = dst_ch;
    MLD_EXPECT_OK(ConvWaveMemoryRuntimeChannelsTest(env, src_tensor, attr, conv_def,
                                                precision, runtime_channels));
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
