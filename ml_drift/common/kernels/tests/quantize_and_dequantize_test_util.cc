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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace {
absl::Status QuantizationUint8Test(TestExecutionEnvironment& exec_env,
                                   const TensorFloat32& src_tensor,
                                   const OperationDef& op_def,
                                   bool calculate_sum) {
  BHWC params_shape = src_tensor.shape;
  params_shape.c = calculate_sum ? 3 : 2;
  Tensor<BHWC, DataType::kUint8> dst_ref_u8;
  dst_ref_u8.shape = src_tensor.shape;
  dst_ref_u8.data.resize(dst_ref_u8.shape.DimensionsProduct());
  Tensor<BHWC, DataType::kFloat32> dst_ref_params;
  dst_ref_params.shape = params_shape;
  dst_ref_params.data.resize(dst_ref_params.shape.DimensionsProduct());
  for (int b = 0; b < src_tensor.shape.b; ++b) {
    for (int h = 0; h < src_tensor.shape.h; ++h) {
      for (int w = 0; w < src_tensor.shape.w; ++w) {
        float min_value = src_tensor.data[0];
        float max_value = min_value;
        for (int c = 0; c < src_tensor.shape.c; ++c) {
          float src_value =
              src_tensor.data[src_tensor.shape.LinearIndex({b, h, w, c})];
          min_value = std::min(min_value, src_value);
          max_value = std::max(max_value, src_value);
        }
        float offset = min_value;
        float range = max_value - min_value;
        float scale = 255.0f / range;
        int sum = 0;
        for (int c = 0; c < src_tensor.shape.c; ++c) {
          float src_value =
              src_tensor.data[src_tensor.shape.LinearIndex({b, h, w, c})];
          float dst_value = (src_value - offset) * scale;
          dst_value = std::max(0.0f, dst_value);
          dst_value = std::min(255.0f, dst_value);
          dst_ref_u8.data[dst_ref_u8.shape.LinearIndex({b, h, w, c})] =
              dst_value;
          if (calculate_sum) {
            sum += static_cast<int>(dst_value);
          }
        }
        dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 0})] =
            1.0f / scale;
        dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 1})] =
            offset;
        if (calculate_sum) {
          dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 2})] =
              sum;
        }
      }
    }
  }

  auto operation =
      CreateQuantization(op_def, PackedType::kUint8C4, exec_env.GetGpuInfo(),
                         src_tensor.shape, calculate_sum);
  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_u8_td = op_def.dst_tensors[0];
  dst_u8_td.SetBHWCShape(src_tensor.shape);
  TensorDescriptor dst_params_td = op_def.dst_tensors[1];
  dst_params_td.SetBHWCShape(params_shape);

  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td}, {&dst_u8_td, &dst_params_td}, std::move(operation)));
  Tensor<BHWC, DataType::kUint8> dst_u8;
  dst_u8_td.DownloadData(&dst_u8);
  Tensor<BHWC, DataType::kFloat32> dst_params;
  dst_params_td.DownloadData(&dst_params);

  const float eps = op_def.src_tensors[0].GetDataType() == DataType::kFloat32
                        ? 0.00005f
                        : 0.005f;
  for (int b = 0; b < src_tensor.shape.b; ++b) {
    for (int h = 0; h < src_tensor.shape.h; ++h) {
      for (int w = 0; w < src_tensor.shape.w; ++w) {
        float ref_param0 =
            dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 0})];
        float ref_param1 =
            dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 1})];
        float dst_param0 =
            dst_params.data[dst_params.shape.LinearIndex({b, h, w, 0})];
        float dst_param1 =
            dst_params.data[dst_params.shape.LinearIndex({b, h, w, 1})];
        EXPECT_LE(std::abs(ref_param0 - dst_param0), eps);
        EXPECT_LE(std::abs(ref_param1 - dst_param1), eps);
        if (calculate_sum) {
          float ref_param2 =
              dst_ref_params
                  .data[dst_ref_params.shape.LinearIndex({b, h, w, 2})];
          float dst_param2 =
              dst_params.data[dst_params.shape.LinearIndex({b, h, w, 2})];
          EXPECT_LE(std::abs(ref_param2 - dst_param2), src_tensor.shape.c);
        }
      }
    }
  }
  int max_diff = 0;
  for (int i = 0; i < dst_u8.data.size(); ++i) {
    max_diff = std::max(
        max_diff, static_cast<int>(abs(dst_u8.data[i] - dst_ref_u8.data[i])));
  }
  EXPECT_LE(max_diff, 1);
  return absl::OkStatus();
}

absl::Status QuantizationInt8Test(TestExecutionEnvironment& exec_env,
                                  const TensorFloat32& src_tensor,
                                  const OperationDef& op_def,
                                  bool calculate_sum) {
  BHWC params_shape = src_tensor.shape;
  params_shape.c = calculate_sum ? 3 : 2;
  Tensor<BHWC, DataType::kInt8> dst_ref_i8;
  dst_ref_i8.shape = src_tensor.shape;
  dst_ref_i8.data.resize(dst_ref_i8.shape.DimensionsProduct());
  Tensor<BHWC, DataType::kFloat32> dst_ref_params;
  dst_ref_params.shape = params_shape;
  dst_ref_params.data.resize(dst_ref_params.shape.DimensionsProduct());
  for (int b = 0; b < src_tensor.shape.b; ++b) {
    for (int h = 0; h < src_tensor.shape.h; ++h) {
      for (int w = 0; w < src_tensor.shape.w; ++w) {
        float min_value = src_tensor.data[0];
        float max_value = min_value;
        for (int c = 0; c < src_tensor.shape.c; ++c) {
          float src_value =
              src_tensor.data[src_tensor.shape.LinearIndex({b, h, w, c})];
          min_value = std::min(min_value, src_value);
          max_value = std::max(max_value, src_value);
        }
        float range = max_value - min_value;
        float offset = min_value + range / 2.0f;
        float scale = 255.0f / range;
        int sum = 0;
        for (int c = 0; c < src_tensor.shape.c; ++c) {
          float src_value =
              src_tensor.data[src_tensor.shape.LinearIndex({b, h, w, c})];
          float dst_value = (src_value - offset) * scale;
          dst_value = std::max(-128.0f, dst_value);
          dst_value = std::min(127.0f, dst_value);
          dst_ref_i8.data[dst_ref_i8.shape.LinearIndex({b, h, w, c})] =
              dst_value;
          if (calculate_sum) {
            sum += static_cast<int>(dst_value);
          }
        }
        dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 0})] =
            1.0f / scale;
        dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 1})] =
            offset;
        if (calculate_sum) {
          dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 2})] =
              sum;
        }
      }
    }
  }

  auto operation =
      CreateQuantization(op_def, PackedType::kInt8C4, exec_env.GetGpuInfo(),
                         src_tensor.shape, calculate_sum);
  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_u8_td = op_def.dst_tensors[0];
  dst_u8_td.SetBHWCShape(src_tensor.shape);
  TensorDescriptor dst_params_td = op_def.dst_tensors[1];
  dst_params_td.SetBHWCShape(params_shape);

  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td}, {&dst_u8_td, &dst_params_td}, std::move(operation)));
  Tensor<BHWC, DataType::kInt8> dst_i8;
  dst_u8_td.DownloadData(&dst_i8);
  Tensor<BHWC, DataType::kFloat32> dst_params;
  dst_params_td.DownloadData(&dst_params);

  const float eps = op_def.src_tensors[0].GetDataType() == DataType::kFloat32
                        ? 0.00005f
                        : 0.005f;
  for (int b = 0; b < src_tensor.shape.b; ++b) {
    for (int h = 0; h < src_tensor.shape.h; ++h) {
      for (int w = 0; w < src_tensor.shape.w; ++w) {
        float ref_param0 =
            dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 0})];
        float ref_param1 =
            dst_ref_params.data[dst_ref_params.shape.LinearIndex({b, h, w, 1})];
        float dst_param0 =
            dst_params.data[dst_params.shape.LinearIndex({b, h, w, 0})];
        float dst_param1 =
            dst_params.data[dst_params.shape.LinearIndex({b, h, w, 1})];
        EXPECT_LE(std::abs(ref_param0 - dst_param0), eps);
        EXPECT_LE(std::abs(ref_param1 - dst_param1), eps);
        if (calculate_sum) {
          float ref_param2 =
              dst_ref_params
                  .data[dst_ref_params.shape.LinearIndex({b, h, w, 2})];
          float dst_param2 =
              dst_params.data[dst_params.shape.LinearIndex({b, h, w, 2})];
          EXPECT_LE(std::abs(ref_param2 - dst_param2), src_tensor.shape.c);
        }
      }
    }
  }
  int max_diff = 0;
  for (int i = 0; i < dst_i8.data.size(); ++i) {
    max_diff = std::max(
        max_diff, static_cast<int>(abs(dst_i8.data[i] - dst_ref_i8.data[i])));
  }
  EXPECT_LE(max_diff, 1);
  return absl::OkStatus();
}

void NudgeQuantizationRange(const float min, const float max,
                            const int quant_min, const int quant_max,
                            float* nudged_min, float* nudged_max,
                            float* nudged_scale) {
  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  *nudged_scale = (max - min) / (quant_max_float - quant_min_float);
  const float zero_point_from_min = quant_min_float - min / *nudged_scale;
  uint16_t nudged_zero_point;
  if (zero_point_from_min < quant_min_float) {
    nudged_zero_point = static_cast<uint16_t>(quant_min);
  } else if (zero_point_from_min > quant_max_float) {
    nudged_zero_point = static_cast<uint16_t>(quant_max);
  } else {
    nudged_zero_point = static_cast<uint16_t>(std::round(zero_point_from_min));
  }
  *nudged_min = (quant_min_float - nudged_zero_point) * (*nudged_scale);
  *nudged_max = (quant_max_float - nudged_zero_point) * (*nudged_scale);
}

}  // namespace

absl::Status QuantAndDequant_Dim2Bits8Test(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 0.25f, 0.50f, 0.4444444f, 0.00001f};

  // Assume pre-nudged values as this should be done during model conversion.
  const int num_bits = 8;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ 0.0, /**original_max**/ 1.0,
                         quant_min, quant_max, &attr.min, &attr.max,
                         &attr.scale);

  const float eps = precision == CalculationsPrecision::kF32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateQuantizeAndDequantize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 2, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 1.0f, 0.25098f, 0.498039f, 0.443137f, 0.0f}));
  return absl::OkStatus();
}

absl::Status QuantAndDequant_Dim3Bits8_NegativeRangeTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 1, 2);
  src_tensor.data = {0.0f, -0.9f, 0.25f, 0.50f, 0.4444444f, -0.00001f};

  // Assume pre-nudged values as this should be done during model conversion.
  const int num_bits = 8;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ -0.9,
                         /**original_max**/ 0.9, quant_min, quant_max,
                         &attr.min, &attr.max, &attr.scale);

  const float eps = precision == CalculationsPrecision::kF32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateQuantizeAndDequantize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, -0.896471f, 0.247059f, 0.501176f,
                                         0.444706f, 0.0f}));
  return absl::OkStatus();
}

absl::Status QuantAndDequant_Dim3Bits16Test(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 0.25f, 0.50f, 0.4444444f, 0.00001f};

  // Assume pre-nudged values as this should be done during model conversion.
  const int num_bits = 16;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ 0.0, /**original_max**/ 1.0,
                         quant_min, quant_max, &attr.min, &attr.max,
                         &attr.scale);

  const float eps = precision == CalculationsPrecision::kF32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateQuantizeAndDequantize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 1.0f, 0.250004f, 0.500008f,
                                         0.44445f, 1.5259e-05f}));
  return absl::OkStatus();
}

absl::Status QuantAndDequant_Dim2Bits16_NegativeRangeTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 2, 1);
  src_tensor.data = {0.0f, -0.9f, 0.25f, 0.50f, 0.4444444f, -0.00001f};

  // Assume pre-nudged values as this should be done during model conversion.
  const int num_bits = 16;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ -0.9,
                         /**original_max**/ 0.9, quant_min, quant_max,
                         &attr.min, &attr.max, &attr.scale);

  const float eps = precision == CalculationsPrecision::kF32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateQuantizeAndDequantize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 2, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, -0.900014f, 0.249998f, 0.499995f,
                                         0.444431f, 0.0f}));
  return absl::OkStatus();
}

absl::Status QuantizationUint8Test(TestExecutionEnvironment& env,
                                   TensorStorageType float_storage,
                                   DataType float_type,
                                   TensorStorageType dst_storage) {
  const auto src_shape = BHWC(1, 13, 15, 237);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 32.0f + 16.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({float_type, float_storage, Layout::kHWC});
  op_def.dst_tensors.push_back({DataType::kUint8, dst_storage, Layout::kHWC});
  op_def.dst_tensors.push_back({float_type, float_storage, Layout::kHWC});
  ABSL_EXPECT_OK(QuantizationUint8Test(env, src_tensor, op_def,
                                  /*calculate_sum=*/false));
  ABSL_EXPECT_OK(QuantizationUint8Test(env, src_tensor, op_def,
                                  /*calculate_sum=*/true));
  return absl::OkStatus();
}

absl::Status QuantizationInt8Test(TestExecutionEnvironment& env,
                                  TensorStorageType float_storage,
                                  DataType float_type,
                                  TensorStorageType dst_storage) {
  const auto src_shape = BHWC(1, 13, 15, 237);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = src_tensor.data[i] * 32.0f + 16.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({float_type, float_storage, Layout::kHWC});
  op_def.dst_tensors.push_back({DataType::kInt8, dst_storage, Layout::kHWC});
  op_def.dst_tensors.push_back({float_type, float_storage, Layout::kHWC});
  ABSL_EXPECT_OK(QuantizationInt8Test(env, src_tensor, op_def,
                                 /*calculate_sum=*/false));
  ABSL_EXPECT_OK(QuantizationInt8Test(env, src_tensor, op_def,
                                 /*calculate_sum=*/true));
  return absl::OkStatus();
}

}  // namespace ml_drift
