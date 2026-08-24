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

#include "ml_drift/common/kernels/tests/conv_wave_matrix_mali_test_util.h"

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
#include "ml_drift/common/kernels/conv_wave_matrix_mali.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace {
absl::Status ConvWaveMatrixMaliInt8Test(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i8);

  std::unique_ptr<GPUOperation> conv = std::make_unique<ConvWaveMatrixMali>(
      ConvWaveMatrixMali(exec_env.GetGpuInfo(), op_def, weights_i8));

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.SetBHWCShape(BHWC(src_tensor_i8.shape.b, src_tensor_i8.shape.h,
                           DivideRoundUp(src_tensor_i8.shape.w, 4),
                           src_tensor_i8.shape.c));
  {
    TensorDescriptor src_desc_temp = {
        DataType::INT8, TensorStorageType::TEXTURE_2D, Layout::HWC};
    src_desc_temp.UploadData(src_tensor_i8);
    src_td.UploadDataRaw(absl::MakeConstSpan(src_desc_temp.GetData()));
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  ABSL_RETURN_IF_ERROR(
      exec_env.ExecuteGPUOperation({&src_td}, {&dst_td}, std::move(conv)));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvWaveMatrixMaliInt8BigTest(TestExecutionEnvironment& env,
                                           TensorStorageType dst_storage,
                                           const BHWC& src_shape) {
  const int src_channels = src_shape.c;
  const int dst_channels = 128;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         +XNN_EXTRA_BYTES / sizeof(uint8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  ml_drift::Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  {  // float32
    const DataType src_data_type = DataType::INT32;
    const DataType dst_data_type = DataType::INT32;
    OperationDef op_def;
    op_def.src_tensors.push_back(
        {src_data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
    op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
    ABSL_RETURN_IF_ERROR(
        ConvWaveMatrixMaliInt8Test(env, src_i8_tensor, weights_i8, op_def));
  }
  return absl::OkStatus();
}

namespace {
absl::Status ConvWaveMatrixMaliInt8ExternalWeightsTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const OperationDef& op_def) {
  const TensorInt32 dst_ref_tensor =
      FullyConnectedReference(src_tensor_i8, weights_i8);

  auto conv =
      ConvWaveMatrixMali(exec_env.GetGpuInfo(), op_def, weights_i8.shape);

  TensorDescriptor weights_i8_td;
  {
    WeightsDescription weights_desc = conv.GetWeightsDescription();
    std::vector<uint8_t> data(
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape));
    RearrangeWeights(weights_i8, weights_desc, absl::MakeSpan(data));

    weights_i8_td = TensorDescriptor(weights_desc.type,
                                     TensorStorageType::BUFFER, Layout::LINEAR);
    weights_i8_td.SetBHWDCShape(BHWDC(1, 1, 1, 1, data.size()));
    weights_i8_td.UploadDataRaw(absl::MakeConstSpan(data));
  }

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.SetBHWCShape(BHWC(src_tensor_i8.shape.b, src_tensor_i8.shape.h,
                           DivideRoundUp(src_tensor_i8.shape.w, 4),
                           src_tensor_i8.shape.c));
  {
    TensorDescriptor src_desc_temp = {
        DataType::INT8, TensorStorageType::TEXTURE_2D, Layout::HWC};
    src_desc_temp.UploadData(src_tensor_i8);
    src_td.UploadDataRaw(absl::MakeConstSpan(src_desc_temp.GetData()));
  }

  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td, &weights_i8_td}, {&dst_td},
      std::make_unique<ConvWaveMatrixMali>(std::move(conv))));
  TensorInt32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, dst_ref_tensor.data);
  return absl::OkStatus();
}
}  // namespace

absl::Status ConvWaveMatrixMaliInt8ExternalWeightsBigTest(
    TestExecutionEnvironment& env, TensorStorageType dst_storage,
    const BHWC& src_shape) {
  const int src_channels = src_shape.c;
  const int dst_channels = 128;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         +XNN_EXTRA_BYTES / sizeof(uint8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  ml_drift::Tensor<BHWC, DataType::INT8> src_i8_tensor;
  src_i8_tensor.shape = src_shape;
  src_i8_tensor.data.resize(src_shape.DimensionsProduct());
  for (int i = 0; i < src_shape.DimensionsProduct(); ++i) {
    src_i8_tensor.data[i] = src_tensor.data[i] * 127.0f;
  }

  const DataType src_data_type = DataType::INT32;
  const DataType dst_data_type = DataType::INT32;
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {src_data_type, TensorStorageType::TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back({dst_data_type, dst_storage, Layout::HWC});
  ABSL_RETURN_IF_ERROR(ConvWaveMatrixMaliInt8ExternalWeightsTest(
      env, src_i8_tensor, weights_i8, op_def));
  return absl::OkStatus();
}

namespace {

absl::Status ConvWaveMatrixMaliInt8WithSrcQuantizationTest(
    TestExecutionEnvironment& exec_env,
    const ml_drift::TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>& weights_scale,
    const ml_drift::Tensor<Linear, DataType::FLOAT32>* weights_zero_point,
    const DataType float_type, const TensorStorageType float_storage_type,
    const TensorStorageType int_storage_type) {
  FullyConnectedAttributes attr;
  attr.weights =
      MakeWeightsFromInt8(weights_i8, weights_scale, weights_zero_point);
  attr.bias = MakeZeroTensor(Linear(weights_i8.shape.o));
  TensorFloat32 dst_ref_tensor = FullyConnectedReference(attr, src_tensor);

  TensorDescriptor src_params_td;
  TensorDescriptor src_i8_td;
  const Layout layout = src_tensor.shape.b == 1 ? Layout::HWC : Layout::BHWC;
  {
    OperationDef op_def;
    op_def.src_tensors.push_back({float_type, float_storage_type, layout});
    op_def.dst_tensors.push_back({DataType::INT32, int_storage_type, layout});
    op_def.dst_tensors.push_back({float_type, float_storage_type, layout});
    const bool calculate_sum = weights_zero_point != nullptr;
    auto quantization_op =
        CreateQuantization(op_def, PackedType::kInt8W4C4, exec_env.GetGpuInfo(),
                           src_tensor.shape, calculate_sum);
    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(src_tensor);
    src_i8_td = op_def.dst_tensors[0];
    src_i8_td.SetBHWCShape(BHWC(src_tensor.shape.b, src_tensor.shape.h,
                                DivideRoundUp(src_tensor.shape.w, 4),
                                src_tensor.shape.c));
    src_params_td = op_def.dst_tensors[1];
    BHWC params_shape = src_tensor.shape;
    params_shape.c = calculate_sum ? 3 : 2;
    src_params_td.SetBHWCShape(params_shape);

    ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
        {&src_td}, {&src_i8_td, &src_params_td}, std::move(quantization_op)));
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
  conv_def.src_tensors.push_back({DataType::INT32, int_storage_type, layout});
  conv_def.dst_tensors.push_back({DataType::INT32, int_storage_type, layout});

  auto conv =
      ConvWaveMatrixMali(exec_env.GetGpuInfo(), conv_def, weights_i8.shape);

  TensorDescriptor dequant_dst = {float_type, float_storage_type, layout};
  auto dequant_op = CreateDequantization(
      weights_i8.shape, exec_env.GetGpuInfo(), conv_def.dst_tensors[0],
      dequant_dst, src_params_td, weights_sum_i_td, weights_scale_td,
      weights_zero_point_td_ptr);
  ABSL_RETURN_IF_ERROR(conv.AddOperation(exec_env.GetGpuInfo(), &dequant_op));

  TensorDescriptor weights_i8_td;
  {
    WeightsDescription weights_desc = conv.GetWeightsDescription();
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
  std::vector<TensorDescriptor*> src_descs = {&src_i8_td, &weights_i8_td,
                                              &src_params_td, &weights_sum_i_td,
                                              &weights_scale_td};
  if (weights_zero_point != nullptr) {
    src_descs.push_back(&weights_zero_point_td);
  }
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_descs, {&dst_td},
      std::make_unique<ConvWaveMatrixMali>(std::move(conv))));
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
}  // namespace

absl::Status ConvWaveMatrixMaliInt8WithSrcQuantizationBigTest(
    TestExecutionEnvironment& env, TensorStorageType int_storage,
    TensorStorageType float_storage, DataType float_type,
    const BHWC& src_shape) {
  if (float_type != DataType::FLOAT16 && float_type != DataType::FLOAT32) {
    return absl::InvalidArgumentError("Unsupported float type: " +
                                      ToString(float_type));
  }
  const int src_channels = src_shape.c;
  const int dst_channels = 128;

  auto weights_f32 =
      MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_channels);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         +XNN_EXTRA_BYTES / sizeof(uint8_t));
  for (int i = 0; i < weights_i8.shape.DimensionsProduct(); ++i) {
    weights_i8.data[i] = weights_f32.data[i] * 127.0f;
  }
  auto weights_scale = MakeSyntheticTensor(Linear(dst_channels));
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] /= 127.0f;
  }
  auto weights_zp = MakeSyntheticTensor(Linear(dst_channels));

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 2.0f + 2.5f;
  }

  ABSL_RETURN_IF_ERROR(ConvWaveMatrixMaliInt8WithSrcQuantizationTest(
      env, src_tensor, weights_i8, weights_scale,
      /*weights_zero_point=*/nullptr, float_type, float_storage, int_storage));
  ABSL_RETURN_IF_ERROR(ConvWaveMatrixMaliInt8WithSrcQuantizationTest(
      env, src_tensor, weights_i8, weights_scale, &weights_zp, float_type,
      float_storage, int_storage));
  return absl::OkStatus();
}
}  // namespace ml_drift
