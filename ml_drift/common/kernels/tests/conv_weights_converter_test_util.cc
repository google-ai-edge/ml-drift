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

#include "ml_drift/common/kernels/tests/conv_weights_converter_test_util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_weights_converter.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/operations.h"
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

namespace {

absl::Status ConvolutionWeightsConverterTest(
    const Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& weight_desc, TestExecutionEnvironment& env,
    const OperationDef& op_def) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  // reinterpreting weights as HWIO-BHWC tensor
  TensorFloat32 src_tensor_as_hwio;
  src_tensor_as_hwio.shape =
      BHWC(weights.shape.h, weights.shape.w, weights.shape.i, weights.shape.o);
  src_tensor_as_hwio.data.resize(src_tensor_as_hwio.shape.DimensionsProduct(),
                                 2.0);
  // reinterpreting weights as OHWI-BHWC tensor
  TensorFloat32 src_tensor_as_ohwi;
  src_tensor_as_ohwi.shape =
      BHWC(weights.shape.o, weights.shape.h, weights.shape.w, weights.shape.i);
  src_tensor_as_ohwi.data.resize(src_tensor_as_ohwi.shape.DimensionsProduct(),
                                 2.0);
  for (int o = 0; o < weights.shape.o; ++o) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int i = 0; i < weights.shape.i; ++i) {
          const int f_index = weights.shape.LinearIndex({o, y, x, i});
          const int s_index_hwio =
              src_tensor_as_hwio.shape.LinearIndex({y, x, i, o});
          src_tensor_as_hwio.data[s_index_hwio] = weights.data[f_index];
          const int s_index_ohwi =
              src_tensor_as_ohwi.shape.LinearIndex({o, y, x, i});
          src_tensor_as_ohwi.data[s_index_ohwi] = weights.data[f_index];
        }
      }
    }
  }

  WeightsDescription weight_desc_copy = weight_desc;
  weight_desc_copy.type = DataType::FLOAT32;
  const int flt_count =
      GetTotalElementsCountForLayout(weight_desc_copy, weights.shape);
  DataType weights_type = DataType::FLOAT32;

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_type));
  RearrangeWeights(weights, weight_desc_copy, absl::MakeSpan(weights_data));

  std::vector<TensorFloat32> dst_tensors;
  if (weight_desc_copy.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weight_desc_copy.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    dst_tensors.resize(4);
    const int dst_depth = AlignByN(DivideRoundUp(weights.shape.o, 4),
                                   weight_desc_copy.output_group_size);
    const int src_depth = DivideRoundUp(weights.shape.i, 4);
    const int kernel_x = weights.shape.w;
    const int kernel_y = weights.shape.h;
    int texture_width = dst_depth;
    int texture_height = src_depth * kernel_x * kernel_y;
    int sub_size = SizeOf(weights_type) * 4 * texture_width * texture_height;
    for (int i = 0; i < 4; ++i) {
      dst_tensors[i].shape = BHWC(1, texture_height, texture_width, 4);
      dst_tensors[i].data.resize(4 * texture_width * texture_height);
      std::memcpy(dst_tensors[i].data.data(),
                  weights_data.data() + sub_size * i, sub_size);
    }
  } else {
    dst_tensors.resize(1);
    dst_tensors[0].shape = BHWC(1, 1, 1, flt_count);
    dst_tensors[0].data.resize(flt_count);
    std::memcpy(dst_tensors[0].data.data(), weights_data.data(),
                flt_count * SizeOf(weights_type));
  }

  std::vector<TensorFloat32> dst_tensors_gpu(dst_tensors.size());
  std::vector<TensorFloat32*> dst_ptrs;
  std::vector<BHWC> dst_shapes;
  for (int i = 0; i < dst_tensors.size(); ++i) {
    dst_shapes.push_back(dst_tensors[i].shape);
    dst_ptrs.push_back(&dst_tensors_gpu[i]);
  }

  WeightsDescription dst_weights_desc = weight_desc;
  dst_weights_desc.type = op_def.dst_tensors[0].GetDataType();
  auto converter_from_ohwi =
      ConverterToConvWeights(gpu_info, op_def, weights.shape, dst_weights_desc,
                             /*input layout*/ Layout::OHWI);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_as_ohwi},
      std::make_unique<ConverterToConvWeights>(std::move(converter_from_ohwi)),
      dst_shapes, dst_ptrs));
  for (int i = 0; i < dst_tensors.size(); ++i) {
    EXPECT_THAT(dst_tensors[i].data,
                Pointwise(FloatNear(0.0f), dst_tensors_gpu[i].data));
  }

  auto converter_from_hwio =
      ConverterToConvWeights(gpu_info, op_def, weights.shape, dst_weights_desc,
                             /*input layout*/ Layout::HWIO);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_as_hwio},
      std::make_unique<ConverterToConvWeights>(std::move(converter_from_hwio)),
      dst_shapes, dst_ptrs));
  for (int i = 0; i < dst_tensors.size(); ++i) {
    EXPECT_THAT(dst_tensors[i].data,
                Pointwise(FloatNear(0.0f), dst_tensors_gpu[i].data));
  }
  return absl::OkStatus();
}

absl::Status ConvolutionWeightsConverterRawInputTest(
    const Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& weight_desc, TestExecutionEnvironment& env,
    DataType data_type, OperationDef& op_def) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  WeightsDescription weight_desc_copy = weight_desc;
  weight_desc_copy.type = weight_desc.type;
  const int flt_count =
      GetTotalElementsCountForLayout(weight_desc_copy, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(data_type));
  RearrangeWeights(weights, weight_desc_copy, absl::MakeSpan(weights_data));

  TensorFloat32 reference_tensor;
  reference_tensor.shape = BHWC(1, 1, 1, flt_count);
  reference_tensor.data.resize(flt_count);
  if (weight_desc.type == DataType::FLOAT16) {
    // Convert fp16 data to fp32 for comparison.
    const half* weights_data_half =
        reinterpret_cast<const half*>(weights_data.data());
    for (int i = 0; i < flt_count; ++i) {
      reference_tensor.data[i] = static_cast<float>(weights_data_half[i]);
    }
  } else {
    std::memcpy(reference_tensor.data.data(), weights_data.data(),
                flt_count * SizeOf(data_type));
  }

  TensorDescriptor& dst_cpu_desc = op_def.dst_tensors.at(0);
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs{&op_def.src_tensors.at(0)};
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs{&dst_cpu_desc};

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
      std::make_unique<ConverterToConvWeights>(gpu_info, op_def, weights.shape,
                                               weight_desc, Layout::OHWI)));
  TensorFloat32 gpu_output;
  dst_cpu_desc.DownloadData(&gpu_output);
  EXPECT_THAT(reference_tensor.data, gpu_output.data);
  return absl::OkStatus();
}

}  // namespace

absl::Status ConverterToConvWeights1x1OutX4Test(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage,
                                                WeightsLayout weights_layout) {
  const int kSrcChannels = 8;
  const int kDstChannels = 32;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  conv_weight_desc.layout = weights_layout;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  ABSL_RETURN_IF_ERROR(
      ConvolutionWeightsConverterTest(weights, conv_weight_desc, env, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights1x1OutX4UnalignedTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, WeightsLayout weights_layout) {
  const int kSrcChannels = 8;
  const int kDstChannels = 17;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  conv_weight_desc.layout = weights_layout;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  ABSL_RETURN_IF_ERROR(
      ConvolutionWeightsConverterTest(weights, conv_weight_desc, env, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights1x1OutX2Test(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage,
                                                WeightsLayout weights_layout) {
  const int kSrcChannels = 7;
  const int kDstChannels = 37;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 2;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  conv_weight_desc.layout = weights_layout;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  ABSL_RETURN_IF_ERROR(
      ConvolutionWeightsConverterTest(weights, conv_weight_desc, env, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsOutX2Test(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage,
                                             WeightsLayout weights_layout) {
  const int kSrcChannels = 8;
  const int kDstChannels = 38;
  auto weights_shape = OHWI(kDstChannels, 3, 4, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 2;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  conv_weight_desc.layout = weights_layout;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  ABSL_RETURN_IF_ERROR(
      ConvolutionWeightsConverterTest(weights, conv_weight_desc, env, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToConvTransposedWeights4x4Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, WeightsLayout weights_layout) {
  const int kSrcChannels = 7;
  const int kDstChannels = 11;
  auto weights_shape = OHWI(kDstChannels, 4, 4, kSrcChannels);
  WeightsDescription weight_desc;
  weight_desc.spatial_remap = {10, 11, 14, 15, 8, 9, 12, 13,
                               2,  3,  6,  7,  0, 1, 4,  5};

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  weight_desc.layout = weights_layout;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
  ABSL_RETURN_IF_ERROR(
      ConvolutionWeightsConverterTest(weights, weight_desc, env, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights4xTexturesTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, WeightsLayout weights_layout) {
  const int src_channels = 9;
  const int dst_channels = 17;
  auto weights_shape = OHWI(dst_channels, 3, 1, src_channels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  conv_weight_desc.layout = weights_layout;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  ABSL_RETURN_IF_ERROR(
      ConvolutionWeightsConverterTest(weights, conv_weight_desc, env, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsFloat32OHWItoFloat32Test(
    TestExecutionEnvironment& env, DataType data_type,
    const OHWI& weights_shape) {
  auto conv_weight_desc = WeightsDescription{
      .type = data_type,
      .layout = WeightsLayout::kOSpatialIOGroupI4O4,
      .output_group_size = 1,
  };

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.shape.DimensionsProduct(); ++i) {
    weights.data[i] = i;
  }
  TensorDescriptor src_raw_ohwi =
      TensorDescriptor(data_type, TensorStorageType::BUFFER, Layout::LINEAR);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, weights.shape.DimensionsProduct()));
  src_raw_ohwi.UploadDataRaw(absl::MakeConstSpan(
      weights.data.data(), weights.shape.DimensionsProduct()));

  TensorDescriptor dst_descriptor =
      TensorDescriptor(data_type, TensorStorageType::BUFFER, Layout::LINEAR);
  dst_descriptor.SetBHWCShape(
      BHWC(1, 1, 1,
           GetTotalElementsCountForLayout(conv_weight_desc, weights_shape)));
  OperationDef op_def;
  op_def.src_tensors.push_back(src_raw_ohwi);
  op_def.dst_tensors.push_back(dst_descriptor);
  ABSL_RETURN_IF_ERROR(ConvolutionWeightsConverterRawInputTest(
      weights, conv_weight_desc, env, data_type, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsInt8OHWIToUint8Test(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  Tensor<OHWI, DataType::INT8> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights.shape.DimensionsProduct(); ++i) {
    weights.data[i] = static_cast<int8_t>(i);
  }
  TensorDescriptor src_raw_ohwi = TensorDescriptor(
      DataType::INT8, TensorStorageType::BUFFER, Layout::LINEAR);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, weights.shape.DimensionsProduct()));
  src_raw_ohwi.UploadDataRaw(absl::MakeConstSpan(
      weights.data.data(), weights.shape.DimensionsProduct()));

  TensorDescriptor dst_descriptor;
  if (conv_weight_desc.layout ==
      WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    dst_descriptor = TensorDescriptor(
        DataType::UINT32, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(conv_weight_desc, weights_shape);
    tex_size.x /= sizeof(uint32_t) / sizeof(uint8_t);
    dst_descriptor.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 1));
  } else {
    dst_descriptor = TensorDescriptor(
        conv_weight_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    dst_descriptor.SetBHWCShape(
        BHWC(1, 1, 1,
             GetTotalElementsCountForLayout(conv_weight_desc, weights_shape)));
  }

  OperationDef op_def;
  op_def.src_tensors.push_back(src_raw_ohwi);
  op_def.dst_tensors.push_back(dst_descriptor);
  const int elems_count =
      GetTotalElementsCountForLayout(conv_weight_desc, weights.shape);
  std::vector<uint8_t> weights_data(elems_count);
  RearrangeWeightsInt8AsUint8(weights, conv_weight_desc,
                              absl::MakeSpan(weights_data),
                              /*shift_value=*/128, /*pad_value=*/128u);
  TensorFloat32 reference_tensor;
  reference_tensor.shape = BHWC(1, 1, 1, elems_count);
  reference_tensor.data.resize(elems_count);
  for (int i = 0; i < elems_count; ++i) {
    reference_tensor.data[i] = weights_data[i];
  }
  TensorDescriptor& dst_cpu_desc = op_def.dst_tensors.at(0);
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs{&op_def.src_tensors.at(0)};
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs{&dst_cpu_desc};
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
      std::make_unique<ConverterToConvWeights>(
          gpu_info, op_def, weights.shape, conv_weight_desc, Layout::OHWI)));
  auto data = dst_cpu_desc.GetData();
  EXPECT_EQ(data, weights_data);
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsInt4OHWIToUint4Test(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  constexpr int uint4_bits = 4;
  constexpr int uint8_bits = 8;
  constexpr int kUint4ValuesPerUint8Value = uint8_bits / uint4_bits;
  const size_t size_packed =
      (weights_shape.DimensionsProduct() + kUint4ValuesPerUint8Value - 1) /
      kUint4ValuesPerUint8Value;
  // Prepare int4 SRC tensor for Cpu conversion (reference) and Gpu conversion.
  Tensor<OHWI, DataType::INT8> src_int4_weights;
  src_int4_weights.shape = weights_shape;
  src_int4_weights.data.resize(size_packed + XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < size_packed; ++i) {
    int8_t lower_int4 = (i * 2) % 16 - 8;
    int8_t upper_int4 = (i * 2 + 1) % 16 - 8;
    src_int4_weights.data[i] = (upper_int4 << 4) | (lower_int4 & 0x0F);
  }

  // Get reference results by Cpu conversion.
  ml_drift::Tensor<OHWI, DataType::INT8> intermediate_int8_weights;
  intermediate_int8_weights.shape = weights_shape;
  intermediate_int8_weights.data.resize(
      intermediate_int8_weights.shape.DimensionsProduct());
  UnpackPackedIntToInt8(
      src_int4_weights.data.data(), weights_shape.DimensionsProduct(),
      /*bit_width=*/4, intermediate_int8_weights.data.data());
  const int elems_count =
      GetTotalElementsCountForLayout(conv_weight_desc, weights_shape);
  std::vector<uint8_t> reference_dst_data(elems_count /
                                          kUint4ValuesPerUint8Value);
  RearrangeWeightsInt8AsUint4(intermediate_int8_weights, conv_weight_desc,
                              absl::MakeSpan(reference_dst_data),
                              /*shift_value=*/8, /*pad_value=*/8u);

  // Get results to test Gpu conversion.
  TensorDescriptor src_raw_ohwi = TensorDescriptor(
      DataType::INT4, TensorStorageType::BUFFER, Layout::LINEAR);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, size_packed));
  src_raw_ohwi.UploadDataRaw(
      absl::MakeConstSpan(src_int4_weights.data.data(), size_packed));
  TensorDescriptor dst_descriptor;
  if (conv_weight_desc.IsLinearLayout()) {
    dst_descriptor = TensorDescriptor(
        conv_weight_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    dst_descriptor.SetBHWDCShape(
        BHWDC(1, 1, 1, 1, elems_count / kUint4ValuesPerUint8Value));
  } else if (conv_weight_desc.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    dst_descriptor = TensorDescriptor(
        DataType::UINT16, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(conv_weight_desc, weights_shape);
    constexpr int uint4_bits = 4;
    constexpr int uint16_bits = 16;
    constexpr int kUint4ElementsPerUint16Texel = uint16_bits / uint4_bits;
    tex_size.x /= kUint4ElementsPerUint16Texel;
    dst_descriptor.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  } else {
    return absl::InvalidArgumentError(
        "Unsupported weights layout for int4 to uint4 conversion.");
  }
  OperationDef op_def;
  op_def.src_tensors.push_back(src_raw_ohwi);
  op_def.dst_tensors.push_back(dst_descriptor);
  TensorDescriptor& dst_cpu_desc = op_def.dst_tensors.at(0);
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs{&op_def.src_tensors.at(0)};
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs{&dst_cpu_desc};
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation(src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
                              std::make_unique<ConverterToConvWeights>(
                                  gpu_info, op_def, src_int4_weights.shape,
                                  conv_weight_desc, Layout::OHWI)));
  auto dst_data = dst_cpu_desc.GetData();
  EXPECT_EQ(dst_data, reference_dst_data);
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsInt2OHWIToUint2Test(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  const int kUint2ValuesPerUint8Value =
      SizeInBitsOf(DataType::UINT8) / SizeInBitsOf(DataType::UINT2);
  const size_t size_packed = DivideRoundUp(weights_shape.DimensionsProduct(),
                                           kUint2ValuesPerUint8Value);
  // Prepare int2 SRC tensor for Cpu conversion (reference) and Gpu conversion.
  Tensor<OHWI, DataType::INT8> src_int2_weights;
  src_int2_weights.shape = weights_shape;
  src_int2_weights.data.resize(size_packed + XNN_EXTRA_BYTES / sizeof(int8_t));
  // For example, int2 values of [-2, -1, 0, 1] will be stored in one byte as
  // 01 00 11 10 (little endian) in an int8_t value.
  for (int i = 0; i < size_packed; ++i) {
    int8_t int2_val_0 = (i) % 4 - 2;
    int8_t int2_val_1 = (i + 1) % 4 - 2;
    int8_t int2_val_2 = (i + 2) % 4 - 2;
    int8_t int2_val_3 = (i + 3) % 4 - 2;
    src_int2_weights.data[i] =
        (int2_val_3 << 6 & 0xC0) | (int2_val_2 << 4 & 0x30) |
        (int2_val_1 << 2 & 0x0C) | (int2_val_0 & 0x03);
  }

  // Get reference results by CPU weights rearrangement.
  ml_drift::Tensor<OHWI, DataType::INT8> intermediate_int8_weights;
  intermediate_int8_weights.shape = weights_shape;
  intermediate_int8_weights.data.resize(
      intermediate_int8_weights.shape.DimensionsProduct());
  UnpackPackedIntToInt8(
      src_int2_weights.data.data(), weights_shape.DimensionsProduct(),
      /*bit_width=*/2, intermediate_int8_weights.data.data());
  const int elems_count =
      GetTotalElementsCountForLayout(conv_weight_desc, weights_shape);
  std::vector<uint8_t> reference_dst_data(elems_count /
                                          kUint2ValuesPerUint8Value);
  RearrangeWeightsInt8AsUint2(intermediate_int8_weights, conv_weight_desc,
                              absl::MakeSpan(reference_dst_data),
                              /*shift_value=*/2, /*pad_value=*/2u);

  // Get results to test by GPU weights rearrangement.
  TensorDescriptor src_raw_ohwi = TensorDescriptor(
      DataType::INT2, TensorStorageType::BUFFER, Layout::LINEAR);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, size_packed));
  src_raw_ohwi.UploadDataRaw(
      absl::MakeConstSpan(src_int2_weights.data.data(), size_packed));
  TensorDescriptor dst_descriptor;
  if (conv_weight_desc.IsLinearLayout()) {
    dst_descriptor = TensorDescriptor(
        DataType::UINT8, TensorStorageType::BUFFER, Layout::LINEAR);
    dst_descriptor.SetBHWDCShape(BHWDC(
        1, 1, 1, 1, DivideRoundUp(elems_count, kUint2ValuesPerUint8Value)));
  } else if (conv_weight_desc.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    dst_descriptor = TensorDescriptor(
        DataType::UINT8, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(conv_weight_desc, weights_shape);
    const int kUint2ElementsPerUint8Value = DivideRoundUp(
        SizeInBitsOf(DataType::UINT8), SizeInBitsOf(DataType::UINT2));
    tex_size.x /= kUint2ElementsPerUint8Value;
    dst_descriptor.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  } else {
    return absl::InvalidArgumentError(
        "Unsupported weights layout for int2 to uint2 conversion.");
  }
  OperationDef op_def;
  op_def.src_tensors.push_back(src_raw_ohwi);
  op_def.dst_tensors.push_back(dst_descriptor);
  TensorDescriptor& dst_cpu_desc = op_def.dst_tensors.at(0);
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs{&op_def.src_tensors.at(0)};
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs{&dst_cpu_desc};
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation(src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
                              std::make_unique<ConverterToConvWeights>(
                                  gpu_info, op_def, src_int2_weights.shape,
                                  conv_weight_desc, Layout::OHWI)));
  auto dst_data = dst_cpu_desc.GetData();
  EXPECT_EQ(dst_data, reference_dst_data);
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsInt2OHWIToFloatTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  const int kUint2ValuesPerUint8Value =
      SizeInBitsOf(DataType::UINT8) / SizeInBitsOf(DataType::UINT2);
  const size_t size_packed = DivideRoundUp(weights_shape.DimensionsProduct(),
                                           kUint2ValuesPerUint8Value);

  // Prepare int2 SRC tensor for Cpu conversion (reference) and Gpu conversion.
  Tensor<OHWI, DataType::INT8> src_int2_weights;
  src_int2_weights.shape = weights_shape;
  src_int2_weights.data.resize(size_packed + XNN_EXTRA_BYTES / sizeof(int8_t));

  Tensor<OHWI, DataType::INT8> src_int8_weights;
  src_int8_weights.shape = weights_shape;
  src_int8_weights.data.resize(weights_shape.DimensionsProduct() +
                               XNN_EXTRA_BYTES / sizeof(int8_t));
  // For example, int2 values of [-2, -1, 0, 1] will be stored in one byte as
  // 01 00 11 10 (little endian) in an int8_t value.
  for (int i = 0; i < size_packed; ++i) {
    int8_t int2_val_0 = (i) % 4 - 2;
    int8_t int2_val_1 = (i + 1) % 4 - 2;
    int8_t int2_val_2 = (i + 2) % 4 - 2;
    int8_t int2_val_3 = (i + 3) % 4 - 2;
    src_int2_weights.data[i] = (int2_val_3 << 6 & 0xC0) |
                               (int2_val_2 << 4 & 0x30) |
                               (int2_val_1 << 2 & 0x0C) | (int2_val_0 & 0x03);
    src_int8_weights.data[i * 4] = int2_val_0;
    src_int8_weights.data[i * 4 + 1] = int2_val_1;
    src_int8_weights.data[i * 4 + 2] = int2_val_2;
    src_int8_weights.data[i * 4 + 3] = int2_val_3;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 8.0f;
  }
  Tensor<OHWI, DataType::INT32> weights_zero_point;
  Tensor<OHWI, DataType::FLOAT32> weights_zero_point_f32;
  weights_zero_point.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_zero_point_f32.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  weights_zero_point_f32.data.resize(
      weights_zero_point_f32.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = i % 5 - 10;
    weights_zero_point_f32.data[i] = weights_zero_point.data[i];
  }

  // Get reference results by Cpu conversion.
  Tensor<OHWI, DataType::FLOAT32> src_f32_weights =
      DequantizeTensor(src_int8_weights, weights_scales, weights_zero_point);
  src_f32_weights.data.resize(src_f32_weights.shape.DimensionsProduct() +
                              XNN_EXTRA_BYTES / sizeof(float));
  const int elems_count =
      GetTotalElementsCountForLayout(conv_weight_desc, weights_shape);
  std::vector<float> reference_dst_data(elems_count);
  RearrangeWeights(
      src_f32_weights, conv_weight_desc,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(reference_dst_data.data()),
                     SizeOf(conv_weight_desc.type)));

  OperationDef op_def;
  // Get results to test Gpu conversion.
  TensorDescriptor src_raw_ohwi = TensorDescriptor(
      DataType::INT2, TensorStorageType::BUFFER, Layout::LINEAR);
  op_def.src_tensors.push_back(src_raw_ohwi);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, size_packed));
  src_raw_ohwi.UploadDataRaw(
      absl::MakeConstSpan(src_int2_weights.data.data(), size_packed));
  TensorDescriptor dst_desc =
      GetTensorDescriptorsForWeightsLayout(weights_shape, conv_weight_desc)[0];
  op_def.dst_tensors.push_back(dst_desc);

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point_f32, DataType::FLOAT32);

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_raw_ohwi, &scale_desc, &zp_desc}, {&dst_desc},
      std::make_unique<ConverterToConvWeights>(
          gpu_info, op_def, src_int2_weights.shape, conv_weight_desc,
          Layout::OHWI, &scale_desc, &zp_desc)));
  TensorFloat32 gpu_output;
  dst_desc.DownloadData(&gpu_output);
  EXPECT_THAT(gpu_output.data, Pointwise(FloatNear(1e-5f), reference_dst_data));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsInt4OHWIToFloatTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  constexpr int uint4_bits = 4;
  constexpr int uint8_bits = 8;
  constexpr int kUint4ValuesPerUint8Value = uint8_bits / uint4_bits;
  const size_t size_packed =
      (weights_shape.DimensionsProduct() + kUint4ValuesPerUint8Value - 1) /
      kUint4ValuesPerUint8Value;

  // Prepare SRC int4 weights (quantized weights, scale and zero point).
  Tensor<OHWI, DataType::INT4> src_int4_weights;
  src_int4_weights.shape = weights_shape;
  src_int4_weights.data.resize(size_packed + XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < size_packed; ++i) {
    int8_t lower_int4 = (i * 2) % 16 - 8;
    int8_t upper_int4 = (i * 2 + 1) % 16 - 8;
    src_int4_weights.data[i] = (upper_int4 << 4) | (lower_int4 & 0x0F);
  }
  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 8.0f;
  }
  Tensor<OHWI, DataType::INT32> weights_zero_point;
  Tensor<OHWI, DataType::FLOAT32> weights_zero_point_f32;
  weights_zero_point.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_zero_point_f32.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  weights_zero_point_f32.data.resize(
      weights_zero_point_f32.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = i % 5 - 10;
    weights_zero_point_f32.data[i] = weights_zero_point.data[i];
  }

  // Get reference results by Cpu conversion.
  Tensor<OHWI, DataType::FLOAT32> src_f32_weights =
      DequantizeTensor(src_int4_weights, weights_scales, weights_zero_point);
  src_f32_weights.data.resize(src_f32_weights.shape.DimensionsProduct() +
                              XNN_EXTRA_BYTES / sizeof(float));
  const int elems_count =
      GetTotalElementsCountForLayout(conv_weight_desc, weights_shape);
  std::vector<float> reference_dst_data(elems_count);
  RearrangeWeights(
      src_f32_weights, conv_weight_desc,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(reference_dst_data.data()),
                     SizeOf(conv_weight_desc.type)));

  OperationDef op_def;
  // Get results to test Gpu conversion.
  TensorDescriptor src_raw_ohwi = TensorDescriptor(
      DataType::INT4, TensorStorageType::BUFFER, Layout::LINEAR);
  op_def.src_tensors.push_back(src_raw_ohwi);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, size_packed));
  src_raw_ohwi.UploadDataRaw(
      absl::MakeConstSpan(src_int4_weights.data.data(), size_packed));
  TensorDescriptor dst_desc =
      GetTensorDescriptorsForWeightsLayout(weights_shape, conv_weight_desc)[0];
  op_def.dst_tensors.push_back(dst_desc);

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point_f32, DataType::FLOAT32);

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_raw_ohwi, &scale_desc, &zp_desc}, {&dst_desc},
      std::make_unique<ConverterToConvWeights>(
          gpu_info, op_def, src_int4_weights.shape, conv_weight_desc,
          Layout::OHWI, &scale_desc, &zp_desc)));
  TensorFloat32 gpu_output;
  dst_desc.DownloadData(&gpu_output);
  EXPECT_THAT(gpu_output.data, Pointwise(FloatNear(1e-5f), reference_dst_data));
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsInt8OHWIToFloatTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  // Prepare SRC int8 weights (quantized weights, scale and zero point).
  Tensor<OHWI, DataType::INT8> src_int8_weights;
  src_int8_weights.shape = weights_shape;
  src_int8_weights.data.resize(src_int8_weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < src_int8_weights.shape.DimensionsProduct(); ++i) {
    src_int8_weights.data[i] = static_cast<int8_t>(i);
  }

  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 8.0f;
  }
  Tensor<OHWI, DataType::INT32> weights_zero_point;
  Tensor<OHWI, DataType::FLOAT32> weights_zero_point_f32;
  weights_zero_point.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_zero_point_f32.shape = OHWI(weights_shape.o, 1, 1, 1);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  weights_zero_point_f32.data.resize(
      weights_zero_point_f32.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = i % 5 - 10;
    weights_zero_point_f32.data[i] = weights_zero_point.data[i];
  }

  // Get reference results by Cpu conversion.
  Tensor<OHWI, DataType::FLOAT32> src_f32_weights =
      DequantizeTensor(src_int8_weights, weights_scales, weights_zero_point);
  src_f32_weights.data.resize(src_f32_weights.shape.DimensionsProduct() +
                              XNN_EXTRA_BYTES / sizeof(float));
  const int elems_count =
      GetTotalElementsCountForLayout(conv_weight_desc, weights_shape);
  std::vector<float> reference_dst_data(elems_count);
  RearrangeWeights(
      src_f32_weights, conv_weight_desc,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(reference_dst_data.data()),
                     SizeOf(conv_weight_desc.type)));

  // Get results to test Gpu conversion.
  TensorDescriptor src_raw_ohwi = TensorDescriptor(
      DataType::INT8, TensorStorageType::BUFFER, Layout::LINEAR);
  OperationDef op_def;
  op_def.src_tensors.push_back(src_raw_ohwi);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, weights_shape.DimensionsProduct()));
  src_raw_ohwi.UploadDataRaw(absl::MakeConstSpan(
      src_int8_weights.data.data(), weights_shape.DimensionsProduct()));
  TensorDescriptor dst_desc =
      GetTensorDescriptorsForWeightsLayout(weights_shape, conv_weight_desc)[0];
  op_def.dst_tensors.push_back(dst_desc);

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point_f32, DataType::FLOAT32);

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_raw_ohwi, &scale_desc, &zp_desc}, {&dst_desc},
      std::make_unique<ConverterToConvWeights>(
          gpu_info, op_def, src_int8_weights.shape, conv_weight_desc,
          Layout::OHWI, &scale_desc, &zp_desc)));
  TensorFloat32 gpu_output;
  dst_desc.DownloadData(&gpu_output);
  EXPECT_THAT(gpu_output.data, Pointwise(FloatNear(1e-5f), reference_dst_data));
  return absl::OkStatus();
}

absl::Status ConverterToOSpatialIOGroupITileOTileIXTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    WeightsDescription& weight_desc) {
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights_shape.DimensionsProduct(); ++i) {
    weights.data[i] = i;
  }

  weight_desc.type = DataType::FLOAT32;
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::FLOAT32, TensorStorageType::BUFFER, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {DataType::FLOAT32, TensorStorageType::BUFFER, Layout::UNKNOWN});
  ABSL_RETURN_IF_ERROR(
      ConvolutionWeightsConverterTest(weights, weight_desc, env, op_def));
  return absl::OkStatus();
}

absl::Status ConverterToISpatialOI4O4UnalignedIOTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    TensorStorageType src_storage, DataType src_type, DataType dst_type) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({src_type, src_storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {dst_type, TensorStorageType::BUFFER, Layout::UNKNOWN});

  // reinterpreting weights as HWIO-BHWC tensor
  TensorFloat32 src_tensor_as_hwio;
  src_tensor_as_hwio.shape =
      BHWC(weights.shape.h, weights.shape.w, weights.shape.i, weights.shape.o);
  src_tensor_as_hwio.data.resize(src_tensor_as_hwio.shape.DimensionsProduct(),
                                 2.0);

  // reinterpreting weights as OHWI-BHWC tensor
  TensorFloat32 src_tensor_as_ohwi;
  src_tensor_as_ohwi.shape =
      BHWC(weights.shape.o, weights.shape.h, weights.shape.w, weights.shape.i);
  src_tensor_as_ohwi.data.resize(src_tensor_as_ohwi.shape.DimensionsProduct(),
                                 2.0);
  for (int o = 0; o < weights.shape.o; ++o) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int i = 0; i < weights.shape.i; ++i) {
          const int f_index = weights.shape.LinearIndex({o, y, x, i});
          const int s_index_hwio =
              src_tensor_as_hwio.shape.LinearIndex({y, x, i, o});
          src_tensor_as_hwio.data[s_index_hwio] = weights.data[f_index];
          const int s_index_ohwi =
              src_tensor_as_ohwi.shape.LinearIndex({o, y, x, i});
          src_tensor_as_ohwi.data[s_index_ohwi] = weights.data[f_index];
        }
      }
    }
  }

  WeightsDescription weight_desc;
  weight_desc.type = dst_type;
  weight_desc.layout = WeightsLayout::kISpatialOI4O4UnalignedIO;

  const int elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights.shape);

  std::vector<uint8_t> weights_data(elements_count * SizeOf(weight_desc.type));
  RearrangeWeights(weights, weight_desc, absl::MakeSpan(weights_data));

  TensorFloat32 dst_tensor;
  dst_tensor.shape = BHWC(1, 1, 1, elements_count);
  dst_tensor.data.resize(elements_count);
  if (dst_type == DataType::FLOAT16) {
    half* weights_data_f16 = reinterpret_cast<half*>(weights_data.data());
    for (int i = 0; i < elements_count; ++i) {
      dst_tensor.data[i] = weights_data_f16[i];
    }
  } else {
    float* weights_data_f32 = reinterpret_cast<float*>(weights_data.data());
    for (int i = 0; i < elements_count; ++i) {
      dst_tensor.data[i] = weights_data_f32[i];
    }
  }

  TensorFloat32 dst_tensor_gpu;
  auto converter_from_ohwi =
      ConverterToConvWeights(gpu_info, op_def, weights.shape, weight_desc,
                             /*input layout*/ Layout::OHWI);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_as_ohwi},
      std::make_unique<ConverterToConvWeights>(std::move(converter_from_ohwi)),
      {dst_tensor.shape}, std::vector<TensorFloat32*>{&dst_tensor_gpu}));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), dst_tensor_gpu.data));

  auto converter_from_hwio =
      ConverterToConvWeights(gpu_info, op_def, weights.shape, weight_desc,
                             /*input layout*/ Layout::HWIO);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_as_hwio},
      std::make_unique<ConverterToConvWeights>(std::move(converter_from_hwio)),
      {dst_tensor.shape}, std::vector<TensorFloat32*>{&dst_tensor_gpu}));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), dst_tensor_gpu.data));
  return absl::OkStatus();
}

absl::Status ConverterToCustomGroupsTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    TensorStorageType src_storage, DataType src_type,
    const WeightsDescription& weights_desc) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = static_cast<float>(i);
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({src_type, src_storage, Layout::BHWC});
  op_def.dst_tensors.push_back(
      {weights_desc.type, TensorStorageType::BUFFER, Layout::UNKNOWN});

  // reinterpreting weights as HWIO-BHWC tensor
  TensorFloat32 src_tensor_as_hwio;
  src_tensor_as_hwio.shape =
      BHWC(weights.shape.h, weights.shape.w, weights.shape.i, weights.shape.o);
  src_tensor_as_hwio.data.resize(src_tensor_as_hwio.shape.DimensionsProduct(),
                                 2.0);

  // reinterpreting weights as OHWI-BHWC tensor
  TensorFloat32 src_tensor_as_ohwi;
  src_tensor_as_ohwi.shape =
      BHWC(weights.shape.o, weights.shape.h, weights.shape.w, weights.shape.i);
  src_tensor_as_ohwi.data.resize(src_tensor_as_ohwi.shape.DimensionsProduct(),
                                 2.0);
  for (int o = 0; o < weights.shape.o; ++o) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int i = 0; i < weights.shape.i; ++i) {
          const int f_index = weights.shape.LinearIndex({o, y, x, i});
          const int s_index_hwio =
              src_tensor_as_hwio.shape.LinearIndex({y, x, i, o});
          src_tensor_as_hwio.data[s_index_hwio] = weights.data[f_index];
          const int s_index_ohwi =
              src_tensor_as_ohwi.shape.LinearIndex({o, y, x, i});
          src_tensor_as_ohwi.data[s_index_ohwi] = weights.data[f_index];
        }
      }
    }
  }

  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(elements_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  TensorFloat32 dst_tensor;
  dst_tensor.shape = BHWC(1, 1, 1, elements_count);
  dst_tensor.data.resize(elements_count);
  if (weights_desc.type == DataType::FLOAT16) {
    half* weights_data_f16 = reinterpret_cast<half*>(weights_data.data());
    for (int i = 0; i < elements_count; ++i) {
      dst_tensor.data[i] = weights_data_f16[i];
    }
  } else {
    float* weights_data_f32 = reinterpret_cast<float*>(weights_data.data());
    for (int i = 0; i < elements_count; ++i) {
      dst_tensor.data[i] = weights_data_f32[i];
    }
  }

  TensorFloat32 dst_tensor_gpu;
  auto converter_from_ohwi =
      ConverterToConvWeights(gpu_info, op_def, weights.shape, weights_desc,
                             /*input layout*/ Layout::OHWI);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_as_ohwi},
      std::make_unique<ConverterToConvWeights>(std::move(converter_from_ohwi)),
      {dst_tensor.shape}, std::vector<TensorFloat32*>{&dst_tensor_gpu}));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), dst_tensor_gpu.data));

  auto converter_from_hwio =
      ConverterToConvWeights(gpu_info, op_def, weights.shape, weights_desc,
                             /*input layout*/ Layout::HWIO);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_as_hwio},
      std::make_unique<ConverterToConvWeights>(std::move(converter_from_hwio)),
      {dst_tensor.shape}, std::vector<TensorFloat32*>{&dst_tensor_gpu}));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), dst_tensor_gpu.data));
  return absl::OkStatus();
}

absl::Status Int8ToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, OHWI weights_shape, int src_ch_quant_groups,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc) {
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = weights_shape;
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = i % 256 - 128;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = weights_shape;
  weights_scales.shape.i = src_ch_quant_groups;
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 128.0f;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_shape;
  weights_zero_point.shape.i = src_ch_quant_groups;
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = std::sin((i + 1) * 0.123f);
  }

  Tensor<OHWI, DataType::FLOAT32> weights_f32 =
      MakeWeightsFromInt8(weights_i8, weights_scales, weights_zero_point);

  dst_weights_desc.type = DataType::FLOAT32;
  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT8;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i8.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_i8_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i8.shape));
  RearrangeWeightsInt8AsUint8(weights_i8, src_weights_desc,
                              absl::MakeSpan(src_data), 128, 128u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_i8_td = TensorDescriptor(DataType::UINT32,
                                     TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i8.shape);
    tex_size.x /= 4;  // because we store 4 uint8 as one uint32
    weights_i8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_i8_td = TensorDescriptor(src_weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_i8_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<TensorDescriptor> weights_f32_refs =
      GetTensorDescriptorsForWeightsLayout(weights_f32, dst_weights_desc);
  std::vector<TensorDescriptor> weights_f32_td;
  if (dst_weights_desc.IsLinearLayout()) {
    weights_f32_td.push_back(TensorDescriptor(
        DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR));
    weights_f32_td[0].SetBHWCShape(weights_f32_refs[0].GetBHWCShape());
  } else {
    for (int i = 0; i < 4; ++i) {
      weights_f32_td.push_back(TensorDescriptor(
          DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::HW));
      weights_f32_td.back().SetBHWCShape(weights_f32_refs[i].GetBHWCShape());
    }
  }

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point, DataType::FLOAT32);

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  for (int i = 0; i < weights_f32_td.size(); ++i) {
    op_def.dst_tensors.push_back(weights_f32_td[i]);
  }
  WeightsConverter int8_to_float_converter(
      env.GetGpuInfo(), op_def, weights_i8.shape, src_weights_desc,
      dst_weights_desc, &scale_desc, &zp_desc);

  std::vector<TensorDescriptor*> dst_ptrs(weights_f32_td.size());
  for (int i = 0; i < dst_ptrs.size(); ++i) {
    dst_ptrs[i] = &weights_f32_td[i];
  }
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_i8_td, &scale_desc, &zp_desc}, dst_ptrs,
      std::make_unique<WeightsConverter>(std::move(int8_to_float_converter))));
  for (int i = 0; i < weights_f32_refs.size(); ++i) {
    auto data = weights_f32_td[i].GetData();
    std::vector<float> data_f32(data.size() / 4);
    std::memcpy(data_f32.data(), data.data(), data.size());

    auto data_ref = weights_f32_refs[i].GetData();
    std::vector<float> data_ref_f32(data_ref.size() / 4);
    std::memcpy(data_ref_f32.data(), data_ref.data(), data_ref.size());
    EXPECT_THAT(data_ref_f32, Pointwise(FloatNear(5e-7f), data_f32));
  }
  return absl::OkStatus();
}

absl::Status Int4ToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, OHWI weights_shape, int src_ch_quant_groups,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc) {
  Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = weights_shape;
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    weights_i4.data[i] = i % 8 - 7;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = weights_shape;
  weights_scales.shape.i = src_ch_quant_groups;
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 8.0f;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_shape;
  weights_zero_point.shape.i = src_ch_quant_groups;
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = std::sin(i * 0.123f);
  }

  Tensor<OHWI, DataType::FLOAT32> weights_f32 =
      MakeWeightsFromInt8(weights_i4, weights_scales, weights_zero_point);

  dst_weights_desc.type = DataType::FLOAT32;
  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT4;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i4.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_i4_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i4.shape) / 2);
  RearrangeWeightsInt8AsUint4(weights_i4, src_weights_desc,
                              absl::MakeSpan(src_data), 8, 8u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_i4_td = TensorDescriptor(DataType::UINT16,
                                     TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i4.shape);
    tex_size.x /= 4;  // because we store 4 uint4 as one uint16
    weights_i4_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_i4_td = TensorDescriptor(DataType::UINT8, TensorStorageType::BUFFER,
                                     Layout::LINEAR);
    weights_i4_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_i4_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<TensorDescriptor> weights_f32_refs =
      GetTensorDescriptorsForWeightsLayout(weights_f32, dst_weights_desc);
  std::vector<TensorDescriptor> weights_f32_td;
  if (dst_weights_desc.IsLinearLayout()) {
    weights_f32_td.push_back(TensorDescriptor(
        DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR));
    weights_f32_td[0].SetBHWCShape(weights_f32_refs[0].GetBHWCShape());
  } else {
    for (int i = 0; i < 4; ++i) {
      weights_f32_td.push_back(TensorDescriptor(
          DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::HW));
      weights_f32_td.back().SetBHWCShape(weights_f32_refs[i].GetBHWCShape());
    }
  }

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point, DataType::FLOAT32);

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::UINT8, TensorStorageType::BUFFER, Layout::LINEAR});
  for (int i = 0; i < weights_f32_td.size(); ++i) {
    op_def.dst_tensors.push_back(weights_f32_td[i]);
  }
  WeightsConverter int4_to_float_converter(
      env.GetGpuInfo(), op_def, weights_i4.shape, src_weights_desc,
      dst_weights_desc, &scale_desc, &zp_desc);

  std::vector<TensorDescriptor*> dst_ptrs(weights_f32_td.size());
  for (int i = 0; i < dst_ptrs.size(); ++i) {
    dst_ptrs[i] = &weights_f32_td[i];
  }
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_i4_td, &scale_desc, &zp_desc}, dst_ptrs,
      std::make_unique<WeightsConverter>(std::move(int4_to_float_converter))));
  for (int i = 0; i < weights_f32_refs.size(); ++i) {
    auto data = weights_f32_td[i].GetData();
    std::vector<float> data_f32(data.size() / 4);
    std::memcpy(data_f32.data(), data.data(), data.size());

    auto data_ref = weights_f32_refs[i].GetData();
    std::vector<float> data_ref_f32(data_ref.size() / 4);
    std::memcpy(data_ref_f32.data(), data_ref.data(), data_ref.size());
    EXPECT_THAT(data_ref_f32, Pointwise(FloatNear(5e-7f), data_f32));
  }
  return absl::OkStatus();
}

absl::Status Int2ToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, OHWI weights_shape, int src_ch_quant_groups,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc) {
  Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = weights_shape;
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct());
  for (int i = 0; i < weights_i2.data.size(); ++i) {
    weights_i2.data[i] = i % 4 - 2;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = weights_shape;
  weights_scales.shape.i = src_ch_quant_groups;
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 2.0f;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_shape;
  weights_zero_point.shape.i = src_ch_quant_groups;
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = std::sin(i * 0.123f);
  }

  Tensor<OHWI, DataType::FLOAT32> weights_f32 =
      MakeWeightsFromInt8(weights_i2, weights_scales, weights_zero_point);

  dst_weights_desc.type = DataType::FLOAT32;
  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT2;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i2.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_i2_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i2.shape) / 4);
  RearrangeWeightsInt8AsUint2(weights_i2, src_weights_desc,
                              absl::MakeSpan(src_data), 2, 2u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_i2_td = TensorDescriptor(DataType::UINT8,
                                     TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i2.shape);
    tex_size.x /= 4;  // because we store 4 uint2 as one uint8
    weights_i2_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_i2_td = TensorDescriptor(DataType::UINT8, TensorStorageType::BUFFER,
                                     Layout::LINEAR);
    weights_i2_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_i2_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<TensorDescriptor> weights_f32_refs =
      GetTensorDescriptorsForWeightsLayout(weights_f32, dst_weights_desc);
  std::vector<TensorDescriptor> weights_f32_td;
  if (dst_weights_desc.IsLinearLayout()) {
    weights_f32_td.push_back(TensorDescriptor(
        DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR));
    weights_f32_td[0].SetBHWCShape(weights_f32_refs[0].GetBHWCShape());
  } else {
    for (int i = 0; i < 4; ++i) {
      weights_f32_td.push_back(TensorDescriptor(
          DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::HW));
      weights_f32_td.back().SetBHWCShape(weights_f32_refs[i].GetBHWCShape());
    }
  }

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point, DataType::FLOAT32);

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::UINT8, TensorStorageType::BUFFER, Layout::LINEAR});
  for (int i = 0; i < weights_f32_td.size(); ++i) {
    op_def.dst_tensors.push_back(weights_f32_td[i]);
  }
  WeightsConverter int2_to_float_converter(
      env.GetGpuInfo(), op_def, weights_i2.shape, src_weights_desc,
      dst_weights_desc, &scale_desc, &zp_desc);

  std::vector<TensorDescriptor*> dst_ptrs(weights_f32_td.size());
  for (int i = 0; i < dst_ptrs.size(); ++i) {
    dst_ptrs[i] = &weights_f32_td[i];
  }
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_i2_td, &scale_desc, &zp_desc}, dst_ptrs,
      std::make_unique<WeightsConverter>(std::move(int2_to_float_converter))));
  for (int i = 0; i < weights_f32_refs.size(); ++i) {
    auto data = weights_f32_td[i].GetData();
    std::vector<float> data_f32(data.size() / 4);
    std::memcpy(data_f32.data(), data.data(), data.size());

    auto data_ref = weights_f32_refs[i].GetData();
    std::vector<float> data_ref_f32(data_ref.size() / 4);
    std::memcpy(data_ref_f32.data(), data_ref.data(), data_ref.size());
    EXPECT_THAT(data_ref_f32, Pointwise(FloatNear(5e-7f), data_f32));
  }
  return absl::OkStatus();
}

absl::Status Int8ToFloatWeightsWithRuntimeInputTest(
    TestExecutionEnvironment& env) {
  const int i_channels = 304;
  const int i_channels_runtime = 104;
  const int o_channels = 404;
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = i % 256 - 128;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(o_channels, 1, 1, 1);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 128.0f;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = OHWI(o_channels, 1, 1, 1);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = std::sin((i + 1) * 0.123f);
  }

  Tensor<OHWI, DataType::FLOAT32> weights_f32 =
      MakeWeightsFromInt8(weights_i8, weights_scales, weights_zero_point);

  WeightsDescription dst_weights_desc;
  dst_weights_desc.type = DataType::FLOAT32;
  dst_weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  dst_weights_desc.output_group_size = 1;

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT8;
  src_weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  src_weights_desc.output_group_size = 1;

  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i8.shape));
  RearrangeWeightsInt8AsUint8(weights_i8, src_weights_desc,
                              absl::MakeSpan(src_data), 128, 128u);
  TensorDescriptor weights_i8_td;
  weights_i8_td = TensorDescriptor(src_weights_desc.type,
                                   TensorStorageType::BUFFER, Layout::LINEAR);
  weights_i8_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  weights_i8_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<TensorDescriptor> weights_f32_refs =
      GetTensorDescriptorsForWeightsLayout(weights_f32, dst_weights_desc);
  TensorDescriptor weights_f32_td = TensorDescriptor(
      DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_f32_td.SetBHWCShape(weights_f32_refs[0].GetBHWCShape());

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point, DataType::FLOAT32);

  ConvRuntimeCheckDesc runtime_check;
  runtime_check.src_end_ch_index = 0;

  TensorInt32 params;
  params.shape = BHWC(1, 1, 1, 1);
  params.data.resize(params.shape.DimensionsProduct());
  params.data[0] = i_channels_runtime;
  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  params_td.UploadData(params);

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(weights_f32_td);
  WeightsConverter int8_to_float_converter(
      env.GetGpuInfo(), op_def, weights_i8.shape, src_weights_desc,
      dst_weights_desc, &scale_desc, &zp_desc, runtime_check);

  TensorFloat32 zero_tensor =
      MakeZeroTensor(weights_f32_refs[0].GetBHWCShape());
  weights_f32_td.UploadData(zero_tensor);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_i8_td, &scale_desc, &zp_desc, &params_td}, {&weights_f32_td},
      std::make_unique<WeightsConverter>(std::move(int8_to_float_converter))));
  int i_slices = DivideRoundUp(i_channels, 4);
  int o_slices = DivideRoundUp(o_channels, 4);
  int i_slices_aligned = DivideRoundUp(
      AlignByN(i_channels_runtime, ConvRuntimeCheckDesc::kChannelsAlignment),
      4);
  auto data = weights_f32_td.GetData();
  std::vector<float> data_f32(data.size() / 4);
  std::memcpy(data_f32.data(), data.data(), data.size());

  auto data_ref = weights_f32_refs[0].GetData();
  std::vector<float> data_ref_f32(data_ref.size() / 4);
  std::memcpy(data_ref_f32.data(), data_ref.data(), data_ref.size());

  const float eps = 5e-7f;
  for (int i_s = 0; i_s < i_slices; ++i_s) {
    for (int o_s = 0; o_s < o_slices; ++o_s) {
      // kOSpatialIOGroupI4O4, o_group = 1, no spatial -> OII4O4
      int i4o4_offset = (o_s * i_slices + i_s) * 16;
      for (int k = 0; k < 16; ++k) {
        if (i_s >= i_slices_aligned) {
          EXPECT_THAT(data_f32[i4o4_offset + k], testing::FloatEq(0.0f));
        } else {
          EXPECT_THAT(data_f32[i4o4_offset + k],
                      testing::FloatNear(data_ref_f32[i4o4_offset + k], eps));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status Int8ToFloatWeightsWithRuntimeOutputTest(
    TestExecutionEnvironment& env) {
  const int i_channels = 304;
  const int o_channels = 404;
  const int o_channels_runtime = 104;
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = i % 256 - 128;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(o_channels, 1, 1, 1);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = std::sin((i + 1) * 0.123f) / 128.0f;
  }

  Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = OHWI(o_channels, 1, 1, 1);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());
  for (int i = 0; i < weights_zero_point.data.size(); ++i) {
    weights_zero_point.data[i] = std::sin((i + 1) * 0.123f);
  }

  Tensor<OHWI, DataType::FLOAT32> weights_f32 =
      MakeWeightsFromInt8(weights_i8, weights_scales, weights_zero_point);

  WeightsDescription dst_weights_desc;
  dst_weights_desc.type = DataType::FLOAT32;
  dst_weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  dst_weights_desc.output_group_size = 1;

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT8;
  src_weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  src_weights_desc.output_group_size = 1;

  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i8.shape));
  RearrangeWeightsInt8AsUint8(weights_i8, src_weights_desc,
                              absl::MakeSpan(src_data), 128, 128u);
  TensorDescriptor weights_i8_td;
  weights_i8_td = TensorDescriptor(src_weights_desc.type,
                                   TensorStorageType::BUFFER, Layout::LINEAR);
  weights_i8_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  weights_i8_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<TensorDescriptor> weights_f32_refs =
      GetTensorDescriptorsForWeightsLayout(weights_f32, dst_weights_desc);
  TensorDescriptor weights_f32_td = TensorDescriptor(
      DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_f32_td.SetBHWCShape(weights_f32_refs[0].GetBHWCShape());

  auto scale_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_scales, DataType::FLOAT32);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(
      env.GetGpuInfo(), weights_zero_point, DataType::FLOAT32);

  ConvRuntimeCheckDesc runtime_check;
  runtime_check.dst_end_ch_index = 0;

  TensorInt32 params;
  params.shape = BHWC(1, 1, 1, 1);
  params.data.resize(params.shape.DimensionsProduct());
  params.data[0] = o_channels_runtime;
  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  params_td.UploadData(params);

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(weights_f32_td);
  WeightsConverter int8_to_float_converter(
      env.GetGpuInfo(), op_def, weights_i8.shape, src_weights_desc,
      dst_weights_desc, &scale_desc, &zp_desc, runtime_check);

  TensorFloat32 zero_tensor =
      MakeZeroTensor(weights_f32_refs[0].GetBHWCShape());
  weights_f32_td.UploadData(zero_tensor);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_i8_td, &scale_desc, &zp_desc, &params_td}, {&weights_f32_td},
      std::make_unique<WeightsConverter>(std::move(int8_to_float_converter))));
  int i_slices = DivideRoundUp(i_channels, 4);
  int o_slices = DivideRoundUp(o_channels, 4);
  int o_slices_aligned = DivideRoundUp(
      AlignByN(o_channels_runtime, ConvRuntimeCheckDesc::kChannelsAlignment),
      4);
  auto data = weights_f32_td.GetData();
  std::vector<float> data_f32(data.size() / 4);
  std::memcpy(data_f32.data(), data.data(), data.size());

  auto data_ref = weights_f32_refs[0].GetData();
  std::vector<float> data_ref_f32(data_ref.size() / 4);
  std::memcpy(data_ref_f32.data(), data_ref.data(), data_ref.size());

  const float eps = 5e-7f;
  for (int i_s = 0; i_s < i_slices; ++i_s) {
    for (int o_s = 0; o_s < o_slices; ++o_s) {
      // kOSpatialIOGroupI4O4, o_group = 1, no spatial -> OII4O4
      int i4o4_offset = (o_s * i_slices + i_s) * 16;
      for (int k = 0; k < 16; ++k) {
        if (o_s >= o_slices_aligned) {
          EXPECT_THAT(data_f32[i4o4_offset + k], testing::FloatEq(0.0f));
        } else {
          EXPECT_THAT(data_f32[i4o4_offset + k],
                      testing::FloatNear(data_ref_f32[i4o4_offset + k], eps));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status Int8ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc, int i_channels, int o_channels) {
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = i % 256 - 128;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::INT8;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i8.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }
  TensorDescriptor src_weights_i8_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i8.shape));
  RearrangeWeights(weights_i8, src_weights_desc, absl::MakeSpan(src_data));
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_i8_td = TensorDescriptor(
        DataType::INT32, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i8.shape);
    tex_size.x /= sizeof(uint32_t) / sizeof(uint8_t);
    src_weights_i8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    src_weights_i8_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    src_weights_i8_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  src_weights_i8_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i8.shape));
  RearrangeWeightsInt8AsUint8(weights_i8, dst_weights_desc,
                              absl::MakeSpan(dst_ref_data), 128, 128u);

  TensorDescriptor dst_weights_ui8_td;
  if (dst_weights_desc.layout ==
      WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    dst_weights_ui8_td = TensorDescriptor(
        DataType::UINT32, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(dst_weights_desc, weights_i8.shape);
    tex_size.x /= sizeof(uint32_t) / sizeof(uint8_t);
    dst_weights_ui8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 1));
  } else {
    dst_weights_ui8_td = TensorDescriptor(
        dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    dst_weights_ui8_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));
  }

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter int8_to_uint8_converter(env.GetGpuInfo(), op_def,
                                           weights_i8.shape, src_weights_desc,
                                           dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_weights_i8_td}, {&dst_weights_ui8_td},
      std::make_unique<WeightsConverter>(std::move(int8_to_uint8_converter))));
  auto data = dst_weights_ui8_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status FloatToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc) {
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.shape.DimensionsProduct(); ++i) {
    weights.data[i] = std::sin(i * 0.123f);
  }

  dst_weights_desc.type = DataType::FLOAT32;
  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::FLOAT32;
  src_weights_desc.layout = src_layout;
  src_weights_desc.output_group_size = 32;

  TensorDescriptor weights_src_td = TensorDescriptor(
      DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR);
  {
    std::vector<float> data(
        GetTotalElementsCountForLayout(src_weights_desc, weights.shape));
    RearrangeWeights(weights, src_weights_desc,
                     absl::MakeSpan(reinterpret_cast<uint8_t*>(data.data()),
                                    data.size() * 4));
    weights_src_td.SetBHWCShape(BHWC(1, 1, 1, data.size()));
    weights_src_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  std::vector<TensorDescriptor> weights_f32_refs =
      GetTensorDescriptorsForWeightsLayout(weights, dst_weights_desc);
  std::vector<TensorDescriptor> weights_f32_td;
  if (dst_weights_desc.IsLinearLayout()) {
    weights_f32_td.push_back(TensorDescriptor(
        DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR));
    weights_f32_td[0].SetBHWCShape(weights_f32_refs[0].GetBHWCShape());
  } else {
    for (int i = 0; i < 4; ++i) {
      weights_f32_td.push_back(TensorDescriptor(
          DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::HW));
      weights_f32_td.back().SetBHWCShape(weights_f32_refs[i].GetBHWCShape());
    }
  }

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR});
  for (int i = 0; i < weights_f32_td.size(); ++i) {
    op_def.dst_tensors.push_back(weights_f32_td[i]);
  }
  WeightsConverter converter(env.GetGpuInfo(), op_def, weights.shape,
                             src_weights_desc, dst_weights_desc);

  std::vector<TensorDescriptor*> dst_ptrs(weights_f32_td.size());
  for (int i = 0; i < dst_ptrs.size(); ++i) {
    dst_ptrs[i] = &weights_f32_td[i];
  }
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_src_td}, dst_ptrs,
      std::make_unique<WeightsConverter>(std::move(converter))));
  for (int i = 0; i < weights_f32_refs.size(); ++i) {
    auto data = weights_f32_td[i].GetData();
    std::vector<float> data_f32(data.size() / 4);
    std::memcpy(data_f32.data(), data.data(), data.size());

    auto data_ref = weights_f32_refs[i].GetData();
    std::vector<float> data_ref_f32(data_ref.size() / 4);
    std::memcpy(data_ref_f32.data(), data_ref.data(), data_ref.size());
    EXPECT_THAT(data_ref_f32, Pointwise(FloatNear(5e-7f), data_f32));
  }
  return absl::OkStatus();
}

absl::Status Uint8ToInt8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = i % 256 - 128;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT8;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i8.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui8_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i8.shape));
  RearrangeWeightsInt8AsUint8(weights_i8, src_weights_desc,
                              absl::MakeSpan(src_data), 128, 128u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui8_td = TensorDescriptor(
        DataType::UINT32, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i8.shape);
    tex_size.x /= sizeof(uint32_t) / sizeof(uint8_t);
    weights_ui8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui8_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui8_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui8_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i8.shape));
  RearrangeWeights(weights_i8, dst_weights_desc, absl::MakeSpan(dst_ref_data));
  TensorDescriptor weights_dst_i8_td = TensorDescriptor(
      dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_i8_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint8_to_int8_converter(env.GetGpuInfo(), op_def,
                                           weights_i8.shape, src_weights_desc,
                                           dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui8_td}, {&weights_dst_i8_td},
      std::make_unique<WeightsConverter>(std::move(uint8_to_int8_converter))));
  auto data = weights_dst_i8_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status Uint4ToInt8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i4.shape.DimensionsProduct(); ++i) {
    weights_i4.data[i] = i % 8 - 7;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT4;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i4.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui4_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i4.shape) / 2);
  RearrangeWeightsInt8AsUint4(weights_i4, src_weights_desc,
                              absl::MakeSpan(src_data), 8, 8u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui4_td = TensorDescriptor(
        DataType::UINT16, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i4.shape);
    tex_size.x /= 4;  // because we store 4 uint4 as one uint16
    weights_ui4_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui4_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui4_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui4_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i4.shape));
  RearrangeWeights(weights_i4, dst_weights_desc, absl::MakeSpan(dst_ref_data));
  TensorDescriptor weights_dst_i8_td = TensorDescriptor(
      dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_i8_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint4_to_int8_converter(env.GetGpuInfo(), op_def,
                                           weights_i4.shape, src_weights_desc,
                                           dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui4_td}, {&weights_dst_i8_td},
      std::make_unique<WeightsConverter>(std::move(uint4_to_int8_converter))));
  auto data = weights_dst_i8_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status Uint2ToInt8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i2.shape.DimensionsProduct(); ++i) {
    weights_i2.data[i] = i % 2 - 1;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT2;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i2.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui2_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i2.shape) / 4);
  RearrangeWeightsInt8AsUint2(weights_i2, src_weights_desc,
                              absl::MakeSpan(src_data), 2, 2u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui2_td = TensorDescriptor(
        DataType::UINT8, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i2.shape);
    tex_size.x /= 4;  // because we store 4 uint2 as one uint8
    weights_ui2_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui2_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui2_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui2_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i2.shape));
  RearrangeWeights(weights_i2, dst_weights_desc, absl::MakeSpan(dst_ref_data));
  TensorDescriptor weights_dst_i8_td = TensorDescriptor(
      dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_i8_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint2_to_int8_converter(env.GetGpuInfo(), op_def,
                                           weights_i2.shape, src_weights_desc,
                                           dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui2_td}, {&weights_dst_i8_td},
      std::make_unique<WeightsConverter>(std::move(uint2_to_int8_converter))));
  auto data = weights_dst_i8_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status Uint8ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = i % 256 - 128;
  }
  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT8;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i8.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui8_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i8.shape));
  RearrangeWeightsInt8AsUint8(weights_i8, src_weights_desc,
                              absl::MakeSpan(src_data), 128, 128u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui8_td = TensorDescriptor(
        DataType::UINT32, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i8.shape);
    tex_size.x /= 4;  // because we store 4 uint8 as one uint32
    weights_ui8_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui8_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui8_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui8_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i8.shape));
  RearrangeWeightsInt8AsUint8(weights_i8, dst_weights_desc,
                              absl::MakeSpan(dst_ref_data), 128, 128u);
  TensorDescriptor weights_dst_ui8_td = TensorDescriptor(
      dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_ui8_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint8_to_uint8_converter(env.GetGpuInfo(), op_def,
                                            weights_i8.shape, src_weights_desc,
                                            dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui8_td}, {&weights_dst_ui8_td},
      std::make_unique<WeightsConverter>(std::move(uint8_to_uint8_converter))));
  auto data = weights_dst_ui8_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status Uint4ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i4.shape.DimensionsProduct(); ++i) {
    weights_i4.data[i] = i % 8 - 7;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT4;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i4.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui4_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i4.shape) / 2);
  RearrangeWeightsInt8AsUint4(weights_i4, src_weights_desc,
                              absl::MakeSpan(src_data), 8, 8u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui4_td = TensorDescriptor(
        DataType::UINT16, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i4.shape);
    tex_size.x /= 4;  // because we store 4 uint4 as one uint16
    weights_ui4_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui4_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui4_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui4_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i4.shape));
  RearrangeWeightsInt8AsUint8(weights_i4, dst_weights_desc,
                              absl::MakeSpan(dst_ref_data), 128, 128u);
  TensorDescriptor weights_dst_ui8_td = TensorDescriptor(
      dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_ui8_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint4_to_uint8_converter(env.GetGpuInfo(), op_def,
                                            weights_i4.shape, src_weights_desc,
                                            dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui4_td}, {&weights_dst_ui8_td},
      std::make_unique<WeightsConverter>(std::move(uint4_to_uint8_converter))));
  auto data = weights_dst_ui8_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status Uint2ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i2.shape.DimensionsProduct(); ++i) {
    weights_i2.data[i] = i % 2 - 1;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT2;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i2.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui2_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i2.shape) / 4);
  RearrangeWeightsInt8AsUint2(weights_i2, src_weights_desc,
                              absl::MakeSpan(src_data), 2, 2u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui2_td = TensorDescriptor(
        DataType::UINT8, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i2.shape);
    tex_size.x /= 4;  // because we store 4 uint2 as one uint8
    weights_ui2_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui2_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui2_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui2_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i2.shape));
  RearrangeWeightsInt8AsUint8(weights_i2, dst_weights_desc,
                              absl::MakeSpan(dst_ref_data), 128, 128u);
  TensorDescriptor weights_dst_ui8_td = TensorDescriptor(
      dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_ui8_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint2_to_int8_converter(env.GetGpuInfo(), op_def,
                                           weights_i2.shape, src_weights_desc,
                                           dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui2_td}, {&weights_dst_ui8_td},
      std::make_unique<WeightsConverter>(std::move(uint2_to_int8_converter))));
  auto data = weights_dst_ui8_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status Uint4ToInt4WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i4.shape.DimensionsProduct(); ++i) {
    weights_i4.data[i] = i % 8 - 7;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT4;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i4.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui4_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i4.shape) / 2);
  RearrangeWeightsInt8AsUint4(weights_i4, src_weights_desc,
                              absl::MakeSpan(src_data), 8, 8u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui4_td = TensorDescriptor(
        DataType::UINT16, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i4.shape);
    tex_size.x /= 4;  // because we store 4 uint4 as one uint16
    weights_ui4_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui4_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui4_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui4_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i4.shape) / 2);
  RearrangeWeightsInt4(weights_i4, dst_weights_desc,
                       absl::MakeSpan(dst_ref_data));
  TensorDescriptor weights_dst_i4_td = TensorDescriptor(
      DataType::INT8, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_i4_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint4_to_int4_converter(env.GetGpuInfo(), op_def,
                                           weights_i4.shape, src_weights_desc,
                                           dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui4_td}, {&weights_dst_i4_td},
      std::make_unique<WeightsConverter>(std::move(uint4_to_int4_converter))));
  auto data = weights_dst_i4_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

absl::Status Uint2ToInt4WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc) {
  const int i_channels = 128 - 4;
  const int o_channels = 128 + 4;
  Tensor<OHWI, DataType::INT8> weights_i2;
  weights_i2.shape = OHWI(o_channels, 1, 1, i_channels);
  weights_i2.data.resize(weights_i2.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  for (int i = 0; i < weights_i2.shape.DimensionsProduct(); ++i) {
    weights_i2.data[i] = i % 2 - 1;
  }

  WeightsDescription src_weights_desc;
  src_weights_desc.type = DataType::UINT2;
  src_weights_desc.layout = src_layout;
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    src_weights_desc.output_group_size = DivideRoundUp(weights_i2.shape.o, 4);
  } else {
    src_weights_desc.output_group_size = 32;
  }

  TensorDescriptor weights_ui2_td;
  std::vector<uint8_t> src_data(
      GetTotalElementsCountForLayout(src_weights_desc, weights_i2.shape) / 4);
  RearrangeWeightsInt8AsUint2(weights_i2, src_weights_desc,
                              absl::MakeSpan(src_data), 2, 2u);
  if (src_layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    weights_ui2_td = TensorDescriptor(
        DataType::UINT8, TensorStorageType::TEXTURE_2D, Layout::HW);
    uint2 tex_size = Get2dResourceSize(src_weights_desc, weights_i2.shape);
    tex_size.x /= 4;  // because we store 4 uint2 as one uint8
    weights_ui2_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
  } else {
    weights_ui2_td = TensorDescriptor(
        src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_ui2_td.SetBHWCShape(BHWC(1, 1, 1, src_data.size()));
  }
  weights_ui2_td.UploadDataRaw(absl::MakeConstSpan(src_data));

  std::vector<uint8_t> dst_ref_data(
      GetTotalElementsCountForLayout(dst_weights_desc, weights_i2.shape) / 2);
  RearrangeWeightsInt4(weights_i2, dst_weights_desc,
                       absl::MakeSpan(dst_ref_data));
  TensorDescriptor weights_dst_i4_td = TensorDescriptor(
      DataType::INT8, TensorStorageType::BUFFER, Layout::LINEAR);
  weights_dst_i4_td.SetBHWCShape(BHWC(1, 1, 1, dst_ref_data.size()));

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  op_def.dst_tensors.push_back(
      {dst_weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR});
  WeightsConverter uint2_to_int4_converter(env.GetGpuInfo(), op_def,
                                           weights_i2.shape, src_weights_desc,
                                           dst_weights_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&weights_ui2_td}, {&weights_dst_i4_td},
      std::make_unique<WeightsConverter>(std::move(uint2_to_int4_converter))));
  auto data = weights_dst_i4_td.GetData();
  EXPECT_EQ(data, dst_ref_data);
  return absl::OkStatus();
}

}  // namespace ml_drift
