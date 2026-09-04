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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WEIGHTS_CONVERTER_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WEIGHTS_CONVERTER_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_layout.h"

namespace ml_drift {

absl::Status ConverterToConvWeights1x1OutX4Test(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage,
                                                WeightsLayout weights_layout);
absl::Status ConverterToConvWeights1x1OutX4UnalignedTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, WeightsLayout weights_layout);
absl::Status ConverterToConvWeights1x1OutX2Test(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage,
                                                WeightsLayout weights_layout);
absl::Status ConverterToConvWeightsOutX2Test(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage,
                                             WeightsLayout weights_layout);
absl::Status ConverterToConvTransposedWeights4x4Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, WeightsLayout weights_layout);
absl::Status ConverterToConvWeights4xTexturesTest(TestExecutionEnvironment& env,
                                                  DataType data_type,
                                                  TensorStorageType storage,
                                                  WeightsLayout weights_layout);
absl::Status ConverterToConvWeightsFloat32OHWItoFloat32Test(
    TestExecutionEnvironment& env, DataType data_type,
    const OHWI& weights_shape);
absl::Status ConverterToConvWeightsInt8OHWIToUint8Test(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc);
absl::Status ConverterToConvWeightsInt4OHWIToUint4Test(
    TestExecutionEnvironment& env,
    const OHWI& weights_shape, const WeightsDescription& conv_weight_desc);
absl::Status ConverterToConvWeightsInt2OHWIToUint2Test(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc);
absl::Status ConverterToConvWeightsInt2OHWIToFloatTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    const WeightsDescription& conv_weight_desc);
absl::Status ConverterToConvWeightsInt4OHWIToFloatTest(
    TestExecutionEnvironment& env,
    const OHWI& weights_shape, const WeightsDescription& conv_weight_desc);
absl::Status ConverterToConvWeightsInt8OHWIToFloatTest(
    TestExecutionEnvironment& env,
    const OHWI& weights_shape, const WeightsDescription& conv_weight_desc);
absl::Status ConverterToOSpatialIOGroupITileOTileIXTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    WeightsDescription& weight_desc);
absl::Status ConverterToISpatialOI4O4UnalignedIOTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    TensorStorageType src_storage, DataType src_type, DataType dst_type);
absl::Status ConverterToCustomGroupsTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    TensorStorageType src_storage, DataType src_type,
    const WeightsDescription& weights_desc);

absl::Status Int8ToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, OHWI weights_shape, int src_ch_quant_groups,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc);
absl::Status Int4ToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, OHWI weights_shape, int src_ch_quant_groups,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc);
absl::Status Int2ToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, OHWI weights_shape, int src_ch_quant_groups,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc);

absl::Status Int8ToFloatWeightsWithRuntimeInputTest(
    TestExecutionEnvironment& env);
absl::Status Int8ToFloatWeightsWithRuntimeOutputTest(
    TestExecutionEnvironment& env);

absl::Status Int8ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc, int i_channels, int o_channels);
absl::Status FloatToFloatWeightsConverterTest(
    TestExecutionEnvironment& env, const OHWI& weights_shape,
    WeightsLayout src_layout, WeightsDescription& dst_weights_desc);
absl::Status FloatWeightsWithRingedOTest(TestExecutionEnvironment& env,
                                         DataType data_type);
absl::Status FloatWeightsWithRingedITest(TestExecutionEnvironment& env,
                                         DataType data_type);

absl::Status Uint8ToInt8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);
absl::Status Uint4ToInt8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);
absl::Status Uint2ToInt8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);

absl::Status Uint8ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);
absl::Status Uint4ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);
absl::Status Uint2ToUint8WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);

absl::Status Uint4ToInt4WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);
absl::Status Uint2ToInt4WeightsConverterTest(
    TestExecutionEnvironment& env, WeightsLayout src_layout,
    WeightsDescription& dst_weights_desc);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_CONV_WEIGHTS_CONVERTER_TEST_UTIL_H_
