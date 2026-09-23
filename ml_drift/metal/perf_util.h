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

#ifndef ML_DRIFT_METAL_PERF_UTIL_H_
#define ML_DRIFT_METAL_PERF_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {
namespace metal {
absl::Status ConvolutionPerfTest(CalculationsPrecision precision,
                                 const BHWC& src_shape, int dst_channels,
                                 const HW& kernel_size = HW(1, 1));

absl::Status ConvolutionInt8PerfTest(const BHWC& src_shape = BHWC(1, 32, 32,
                                                                  1024),
                                     int dst_channels = 1024);

absl::Status ConvolutionSf16Wi4BatchedPerfTest(const BHWC& src_shape,
                                               int dst_channels,
                                               OHWI scale_zp_shape);

absl::Status ConvolutionSi8Wi4PerfTest(const BHWC& src_shape, int dst_channels);

absl::Status ConvMoEPerfTest(int seq_size, int src_channels, int dst_channels,
                             int num_experts, int num_active_experts,
                             DataType weights_type);

absl::Status ConvSoftmaxConvPerfTest();

// scale_zp_shape defines dequantization area
// if scale_zp_shape = 1x1 -> scalar quantization
// if scale_zp_shape = Ox1 -> linear per output
// if scale_zp_shape = OxK -> blocked/grouped
absl::Status FullyConnectedOptimalWGSize(CalculationsPrecision precision,
                                         DataType weights_type,
                                         const BHWC& src_shape,
                                         int dst_channels,
                                         OHWI scale_zp_shape = OHWI(1, 1, 1, 1),
                                         bool sparse_2x4 = false);

absl::Status FullyConnectedPerfTest(CalculationsPrecision precision,
                                    DataType weights_type,
                                    const BHWC& src_shape, int dst_channels,
                                    OHWI scale_zp_shape = OHWI(1, 1, 1, 1),
                                    bool sparse_2x4 = false);

absl::Status FullyConnectedWeightsBatchIdsPerfTest(
    CalculationsPrecision precision, DataType weights_type,
    const BHWC& src_shape, int dst_channels, int batch_size,
    int active_ids_size, OHWI scale_zp_shape = OHWI(1, 1, 1, 1));

absl::Status FullyConnectedOIPerfTest(const BHWC& src_shape, int dst_channels,
                                      int src_group_size,
                                      DataType weights_type);

absl::Status FullyConnectedOIWeightsBatchIdsPerfTest(
    CalculationsPrecision precision, DataType weights_type,
    const BHWC& src_shape, int dst_channels, int batch_size,
    int active_ids_size, OHWI scale_zp_shape = OHWI(1, 1, 1, 1));

// bandwidth test
absl::Status AddScalarTest(const BHWC& shape, const DataType& data_type);

absl::Status WinogradForwardTest(const BHWC& src_shape,
                                 const DataType& data_type);
absl::Status WinogradBackwardTest(const BHWC& dst_shape,
                                  const DataType& data_type);

absl::Status DepthwiseConvPerfTest(CalculationsPrecision precision,
                                   const BHWC& src_shape,
                                   const HW& kernel_size = HW(3, 3),
                                   const HW& strides = HW(1, 1),
                                   const HW& dilation = HW(1, 1));

absl::Status QuantizationPerfTest(const BHWC& src_shape, DataType src_type,
                                  PackedType dst_type, bool calculate_sum);

absl::Status SoftmaxPerfTest(const BHWC& shape = BHWC(1, 1, 4096, 4096),
                             bool reduce_only = false);
}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_PERF_UTIL_H_
