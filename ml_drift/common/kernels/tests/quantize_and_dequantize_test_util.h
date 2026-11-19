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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_QUANTIZE_AND_DEQUANTIZE_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_QUANTIZE_AND_DEQUANTIZE_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status QuantAndDequant_Dim2Bits8Test(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage);

absl::Status QuantAndDequant_Dim3Bits8_NegativeRangeTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

absl::Status QuantAndDequant_Dim3Bits16Test(TestExecutionEnvironment& env,
                                            CalculationsPrecision precision,
                                            TensorStorageType storage);

absl::Status QuantAndDequant_Dim2Bits16_NegativeRangeTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

// Checks that 8-bit fake quantization of small values rounds to the same codes
// as F32 arithmetic, even when tensors are F16.
absl::Status QuantAndDequant_Bits8_SmallValuesPrecisionTest(
    TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage);

// Checks that a degenerate range (min == max, scale == 0) yields min when
// dequantizing and code 0 when only quantizing, rather than NaN.
absl::Status QuantAndDequant_ZeroScaleTest(TestExecutionEnvironment& env,
                                           CalculationsPrecision precision,
                                           TensorStorageType storage);

absl::Status QuantizationUint8Test(TestExecutionEnvironment& env,
                                   TensorStorageType float_storage,
                                   DataType float_type,
                                   TensorStorageType dst_storage);
absl::Status QuantizationInt8Test(TestExecutionEnvironment& env,
                                  TensorStorageType float_storage,
                                  DataType float_type,
                                  TensorStorageType dst_storage);
absl::Status QuantizationSymmetricUint8Test(TestExecutionEnvironment& env,
                                            TensorStorageType float_storage,
                                            DataType float_type,
                                            TensorStorageType dst_storage);
absl::Status QuantizationSymmetricInt8Test(TestExecutionEnvironment& env,
                                           TensorStorageType float_storage,
                                           DataType float_type,
                                           TensorStorageType dst_storage);

// Check symmetric quantization through GpuModelBuilder::Quantize(), which
// always produces FP32 params, also for an FP16 source.
absl::Status GpuModelBuilderQuantizeSymmetricUint8Test(
    TestExecutionEnvironment& env, TensorStorageType storage,
    DataType float_type);
absl::Status GpuModelBuilderQuantizeSymmetricInt8Test(
    TestExecutionEnvironment& env, TensorStorageType storage,
    DataType float_type);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_QUANTIZE_AND_DEQUANTIZE_TEST_UTIL_H_
