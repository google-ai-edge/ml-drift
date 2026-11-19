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
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

using ::testing::Each;
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

absl::Status QuantAndDequant_Bits8_SmallValuesPrecisionTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage) {
  // Values k / 1024 are exactly representable in F16, so any code mismatch
  // comes from the kernel arithmetic rather than from the input upload.
  constexpr int kMaxK = 300;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2 * kMaxK + 1, 1);
  src_tensor.data.reserve(2 * kMaxK + 1);
  for (int k = -kMaxK; k <= kMaxK; ++k) {
    src_tensor.data.push_back(k / 1024.0f);
  }

  // Symmetric 8-bit range with zero point 128, as used by static range
  // quantization (SRQ). Inputs span roughly +-29 quantization steps.
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ -1.28f, /**original_max**/ 1.27f,
                         /*quant_min=*/0, /*quant_max=*/255, &attr.min,
                         &attr.max, &attr.scale);

  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateQuantizeAndDequantize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      src_tensor.shape, &dst_tensor));

  // A half-precision kernel computing (x - min) / scale lands in [0, 255],
  // where F16 spacing is 0.125-0.25, and rounds ~5% of these codes
  // differently than F32 arithmetic would.
  int checked = 0;
  int mismatches = 0;
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    const double t = static_cast<double>(src_tensor.data[i]) / attr.scale;
    if (std::abs(t - std::floor(t) - 0.5) < 0.02) {
      continue;  // Skip near-ties, which may legitimately round either way.
    }
    ++checked;
    const double expected = std::floor(t + 0.5) * attr.scale;
    // Output rounding error is far below half a quantization step, so a
    // larger difference means a different code.
    if (std::abs(dst_tensor.data[i] - expected) > 0.5 * attr.scale) {
      ++mismatches;
    }
  }
  if (data_type == DataType::kFloat32) {
    EXPECT_EQ(mismatches, 0);
  } else {
    EXPECT_LE(mismatches, checked / 100)
        << mismatches << " of " << checked << " codes differ from F32";
  }
  return absl::OkStatus();
}

absl::Status QuantAndDequant_ZeroScaleTest(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 4, 1);
  src_tensor.data = {-1.0f, 0.0f, 0.5f, 2.0f};

  // A degenerate range with min == max, and therefore scale == 0, as produced
  // for a constant tensor.
  QuantizeAndDequantizeAttributes attr;
  attr.min = 0.5f;
  attr.max = 0.5f;
  attr.scale = 0.0f;

  const float eps = precision == CalculationsPrecision::kF32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});

  // Every input maps to code 0, which dequantizes to min.
  TensorFloat32 dequantized;
  GPUOperation dequantize_op = CreateQuantizeAndDequantize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(dequantize_op)),
      src_tensor.shape, &dequantized));
  EXPECT_THAT(dequantized.data, Each(FloatNear(attr.min, eps)));

  TensorFloat32 quantized;
  GPUOperation quantize_op = CreateStaticRangeQuantization(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(quantize_op)),
      src_tensor.shape, &quantized));
  EXPECT_THAT(quantized.data, Each(FloatNear(0.0f, eps)));
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

namespace {

// Returns an input for the symmetric quantization tests:
// - (b, h, w) = (0, 0, 0) is all zeros;
// - (b, h, w) = (0, 0, 1) spans [-kMaxHalf, kMaxHalf], the largest finite FP16
//   magnitudes, for which the unsigned offset overflows FP16 unless inv_scale
//   is capped;
// - the other rows span about [-12, 4] or [-4, 12], alternating with h.
// Every value is exactly representable in FP16, so that an FP16 tensor holds
// the same numbers that the checks on the CPU use.
// Requires shape.w >= 2 and shape.c >= 2.
TensorFloat32 MakeSymmetricTestTensor(const BHWC& shape) {
  TensorFloat32 src_tensor = MakeSyntheticTensor(shape);
  for (int b = 0; b < shape.b; ++b) {
    for (int h = 0; h < shape.h; ++h) {
      for (int w = 0; w < shape.w; ++w) {
        const bool all_zero_row = (b == 0 && h == 0 && w == 0);
        const bool large_row = (b == 0 && h == 0 && w == 1);
        const float bias = (h % 2 == 0) ? -4.0f : 4.0f;
        for (int c = 0; c < shape.c; ++c) {
          const int idx = shape.LinearIndex({b, h, w, c});
          float value = src_tensor.data[idx];  // In [-1, 1).
          if (all_zero_row) {
            value = 0.0f;
          } else if (large_row) {
            // The first two channels pin both ends of the range.
            if (c == 0) {
              value = kMaxHalf;
            } else if (c == 1) {
              value = -kMaxHalf;
            } else {
              value *= kMaxHalf;
            }
          } else {
            value = value * 8.0f + bias;
          }
          src_tensor.data[idx] = static_cast<float>(half(value));
        }
      }
    }
  }
  return src_tensor;
}

// Checks the `codes` and `params` that symmetric 8-bit quantization (kUint8C4
// or kInt8C4) of `src` produced on the GPU, both against a CPU reference that
// mirrors the kernel and against the contract x ~= q * inv_scale + offset,
// which does not depend on how the kernel rounds. `codes` is in BHWC order,
// like `src`. `is_src_fp16` and `is_params_fp16` describe the GPU tensors.
void ExpectSymmetricQuantization(const TensorFloat32& src,
                                 const std::vector<int>& codes,
                                 const TensorFloat32& params, bool is_signed,
                                 bool is_src_fp16, bool is_params_fp16,
                                 bool calculate_sum) {
  constexpr float kZeroPoint = 128.0f;
  constexpr float kHalfRange = 127.0f;
  constexpr float kMaxCode = 255.0f;
  // Signed codes are the unsigned ones shifted down by the zero point.
  const int code_shift = is_signed ? 128 : 0;
  // Same conditions as in the kernel: the scale floor and, for unsigned types,
  // the scale cap depend on whether the params may be used as FP16.
  const bool scale_may_be_fp16 = is_src_fp16 || is_params_fp16;
  const float min_scale = scale_may_be_fp16 ? kMinNormalHalf : 1.0e-30f;
  const bool cap_scale = scale_may_be_fp16 && !is_signed;
  const float max_scale = kMaxHalf / kZeroPoint;
  // Tolerance for the params against the reference. The inputs are exact in
  // FP16, so only the storage type of the params matters.
  const float eps = is_params_fp16 ? 0.005f : 0.00005f;
  // Largest allowed |x - (q * inv_scale + offset)|, in percent of a
  // quantization step: half a step from rounding to the nearest code, plus
  // float rounding in the kernel. With FP16 params, inv_scale is also off by
  // up to 1 ulp (2^-10), because a GPU may truncate when it stores FP16. That
  // adds up to 127 codes * 2^-10 = 12.4%.
  const int max_round_trip_percent = is_params_fp16 ? 65 : 51;
  ASSERT_EQ(codes.size(), src.data.size());
  ASSERT_EQ(params.data.size(),
            src.data.size() / src.shape.c * (calculate_sum ? 3 : 2));

  int max_code_diff = 0;
  std::string max_code_diff_at;
  int round_trip_failures = 0;
  std::string first_round_trip_failure;
  for (int b = 0; b < src.shape.b; ++b) {
    for (int h = 0; h < src.shape.h; ++h) {
      for (int w = 0; w < src.shape.w; ++w) {
        const auto row = [&] {
          return (::testing::Message()
                  << "(b, h, w) = (" << b << ", " << h << ", " << w << ")")
              .GetString();
        };
        float abs_max = 0.0f;
        for (int c = 0; c < src.shape.c; ++c) {
          abs_max = std::max(
              abs_max, std::abs(src.data[src.shape.LinearIndex({b, h, w, c})]));
        }
        float ref_inv_scale = 0.0f;
        float ref_scale = 0.0f;
        float ref_offset = 0.0f;
        if (abs_max > 0.0f) {
          ref_inv_scale = std::max(min_scale, abs_max / kHalfRange);
          if (cap_scale) {
            ref_inv_scale = std::min(ref_inv_scale, max_scale);
          }
          ref_scale = 1.0f / ref_inv_scale;
          if (!is_signed) {
            ref_offset = -ref_inv_scale * kZeroPoint;
          }
        }
        const float inv_scale =
            params.data[params.shape.LinearIndex({b, h, w, 0})];
        const float offset =
            params.data[params.shape.LinearIndex({b, h, w, 1})];
        EXPECT_LE(std::abs(ref_inv_scale - inv_scale),
                  std::max(eps, eps * std::abs(ref_inv_scale)))
            << "inv_scale at " << row();
        EXPECT_LE(std::abs(ref_offset - offset),
                  std::max(eps, eps * std::abs(ref_offset)))
            << "offset at " << row();
        // The largest value that the codes can represent with these params.
        const float max_dequantized =
            (kMaxCode - code_shift) * inv_scale + offset;
        int ref_sum = 0;
        for (int c = 0; c < src.shape.c; ++c) {
          const int idx = src.shape.LinearIndex({b, h, w, c});
          const float x = src.data[idx];
          const int code = codes[idx];
          const int ref_code =
              static_cast<int>(
                  std::clamp(std::floor(x * ref_scale + (kZeroPoint + 0.5f)),
                             0.0f, kMaxCode)) -
              code_shift;
          ref_sum += ref_code;
          const auto element = [&] {
            return (::testing::Message()
                    << "(b, h, w, c) = (" << b << ", " << h << ", " << w << ", "
                    << c << "): x = " << x << ", code = " << code
                    << ", reference code = " << ref_code
                    << ", inv_scale = " << inv_scale << ", offset = " << offset)
                .GetString();
          };
          if (std::abs(code - ref_code) > max_code_diff) {
            max_code_diff = std::abs(code - ref_code);
            max_code_diff_at = element();
          }
          const float dequantized =
              static_cast<float>(code) * inv_scale + offset;
          // Inputs above the largest code saturate, which only happens where
          // inv_scale is capped.
          const float expected = cap_scale ? std::min(x, max_dequantized) : x;
          // Written so that non-finite params also count as failures.
          if (!(std::abs(dequantized - expected) <=
                0.01f * max_round_trip_percent * inv_scale)) {
            if (++round_trip_failures == 1) {
              first_round_trip_failure = element();
            }
          }
        }
        if (calculate_sum) {
          const float sum = params.data[params.shape.LinearIndex({b, h, w, 2})];
          EXPECT_LE(std::abs(static_cast<float>(ref_sum) - sum), src.shape.c)
              << "sum at " << row();
        }
      }
    }
  }
  // The all-zero row must produce exactly zero params and midpoint codes.
  EXPECT_EQ(params.data[params.shape.LinearIndex({0, 0, 0, 0})], 0.0f);
  EXPECT_EQ(params.data[params.shape.LinearIndex({0, 0, 0, 1})], 0.0f);
  for (int c = 0; c < src.shape.c; ++c) {
    EXPECT_EQ(codes[src.shape.LinearIndex({0, 0, 0, c})], 128 - code_shift)
        << "c = " << c;
  }
  EXPECT_LE(max_code_diff, 1)
      << "largest difference to the reference at " << max_code_diff_at;
  EXPECT_EQ(round_trip_failures, 0)
      << "elements are not within " << max_round_trip_percent
      << "% of a quantization step of q * inv_scale + offset, the first at "
      << first_round_trip_failure;
}

// Runs symmetric quantization of `src_tensor` to `packed_type`, whose codes
// have the type `CodeType`, as a single GPU operation and checks the result.
template <DataType CodeType>
absl::Status RunQuantizationSymmetricTest(TestExecutionEnvironment& exec_env,
                                          const TensorFloat32& src_tensor,
                                          const OperationDef& op_def,
                                          PackedType packed_type,
                                          bool calculate_sum) {
  BHWC params_shape = src_tensor.shape;
  params_shape.c = calculate_sum ? 3 : 2;

  auto operation =
      CreateQuantization(op_def, packed_type, exec_env.GetGpuInfo(),
                         src_tensor.shape, calculate_sum, /*symmetric=*/true);
  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorDescriptor dst_codes_td = op_def.dst_tensors[0];
  dst_codes_td.SetBHWCShape(src_tensor.shape);
  TensorDescriptor dst_params_td = op_def.dst_tensors[1];
  dst_params_td.SetBHWCShape(params_shape);

  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {&src_td}, {&dst_codes_td, &dst_params_td}, std::move(operation)));
  Tensor<BHWC, CodeType> dst_codes;
  dst_codes_td.DownloadData(&dst_codes);
  TensorFloat32 dst_params;
  dst_params_td.DownloadData(&dst_params);

  ExpectSymmetricQuantization(
      src_tensor,
      std::vector<int>(dst_codes.data.begin(), dst_codes.data.end()),
      dst_params, IsSigned(packed_type),
      op_def.src_tensors[0].GetDataType() == DataType::kFloat16,
      op_def.dst_tensors[1].GetDataType() == DataType::kFloat16, calculate_sum);
  return absl::OkStatus();
}

template <DataType CodeType>
absl::Status QuantizationSymmetricTest(TestExecutionEnvironment& env,
                                       TensorStorageType float_storage,
                                       DataType float_type,
                                       TensorStorageType dst_storage,
                                       PackedType packed_type) {
  OperationDef op_def;
  op_def.src_tensors.push_back({float_type, float_storage, Layout::kHWC});
  op_def.dst_tensors.push_back({CodeType, dst_storage, Layout::kHWC});
  op_def.dst_tensors.push_back({float_type, float_storage, Layout::kHWC});
  // C = 237 leaves the last slice partly filled and C = 64 does not. The kernel
  // generates different reduction code for the two cases.
  for (const BHWC& src_shape : {BHWC(1, 13, 15, 237), BHWC(1, 5, 7, 64)}) {
    const TensorFloat32 src_tensor = MakeSymmetricTestTensor(src_shape);
    for (const bool calculate_sum : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "C = " << src_shape.c
                   << ", calculate_sum = " << calculate_sum);
      ABSL_EXPECT_OK(RunQuantizationSymmetricTest<CodeType>(
          env, src_tensor, op_def, packed_type, calculate_sum));
    }
  }
  return absl::OkStatus();
}

// Quantizes symmetrically through GpuModelBuilder::Quantize() and checks the
// result. Unlike in the tests above, the params are always FP32 here, also for
// an FP16 source.
template <DataType CodeType>
absl::Status GpuModelBuilderQuantizeSymmetricTest(TestExecutionEnvironment& env,
                                                  TensorStorageType storage,
                                                  DataType float_type,
                                                  PackedType packed_type) {
  const BHWC src_shape(1, 5, 7, 64);
  const TensorFloat32 src_tensor = MakeSymmetricTestTensor(src_shape);
  for (const bool calculate_sum : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "calculate_sum = " << calculate_sum);
    GpuModelBuilderOptions options;
    options.storage = storage;
    GpuModelBuilder builder(env.GetGpuInfo(), options);
    const auto src = builder.AddTensor(src_shape, float_type);
    const auto outputs = builder.Quantize(src, packed_type, calculate_sum,
                                          /*symmetric=*/true);
    if (outputs.size() != 2) {
      return absl::InternalError("Quantize() must return codes and params.");
    }
    const auto& codes = outputs[0];
    const auto& params = outputs[1];

    // Only the params are a model output: ExecuteGpuModel() downloads every
    // model output into the same tensor type, and the codes are not floats.
    // The codes are read from their descriptor instead, where ExecuteGpuModel()
    // leaves the data of every node output.
    GpuModel gpu_model;
    ABSL_RETURN_IF_ERROR(
        builder.GetGpuModel({src.id}, {params.id}, &gpu_model));
    TensorFloat32 dst_params;
    ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(
        {src_tensor}, std::vector<TensorFloat32*>{&dst_params}, &gpu_model));
    Tensor<BHWC, CodeType> dst_codes;
    gpu_model.tensors.at(codes.id).DownloadData(&dst_codes);

    ExpectSymmetricQuantization(
        src_tensor,
        std::vector<int>(dst_codes.data.begin(), dst_codes.data.end()),
        dst_params, IsSigned(packed_type), float_type == DataType::kFloat16,
        params.tensor_desc.GetDataType() == DataType::kFloat16, calculate_sum);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status QuantizationSymmetricUint8Test(TestExecutionEnvironment& env,
                                            TensorStorageType float_storage,
                                            DataType float_type,
                                            TensorStorageType dst_storage) {
  return QuantizationSymmetricTest<DataType::kUint8>(
      env, float_storage, float_type, dst_storage, PackedType::kUint8C4);
}

absl::Status QuantizationSymmetricInt8Test(TestExecutionEnvironment& env,
                                           TensorStorageType float_storage,
                                           DataType float_type,
                                           TensorStorageType dst_storage) {
  return QuantizationSymmetricTest<DataType::kInt8>(
      env, float_storage, float_type, dst_storage, PackedType::kInt8C4);
}

absl::Status GpuModelBuilderQuantizeSymmetricUint8Test(
    TestExecutionEnvironment& env, TensorStorageType storage,
    DataType float_type) {
  return GpuModelBuilderQuantizeSymmetricTest<DataType::kUint8>(
      env, storage, float_type, PackedType::kUint8C4);
}

absl::Status GpuModelBuilderQuantizeSymmetricInt8Test(
    TestExecutionEnvironment& env, TensorStorageType storage,
    DataType float_type) {
  return GpuModelBuilderQuantizeSymmetricTest<DataType::kInt8>(
      env, storage, float_type, PackedType::kInt8C4);
}

}  // namespace ml_drift
