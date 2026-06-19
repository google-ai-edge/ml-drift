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

#ifndef ML_DRIFT_CL_TESTING_PERF_UTIL_H_
#define ML_DRIFT_CL_TESTING_PERF_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace cl {

absl::Status ConvolutionPerfTest(CalculationsPrecision precision,
                                 const BHWC& src_shape = BHWC(1, 32, 32, 1024),
                                 int dst_channels = 1024,
                                 const HW& kernel_size = HW(1, 1),
                                 const HW& strides = HW(1, 1),
                                 const HW& dilations = HW(1, 1));

absl::Status ConvolutionWinogradPerfTest(CalculationsPrecision precision,
                                         const BHWC& src_shape = BHWC(1, 36,
                                                                      128, 512),
                                         int dst_channels = 512);

absl::Status ConvolutionInt8PerfTest(const BHWC& src_shape = BHWC(1, 32, 32,
                                                                  1024),
                                     int dst_channels = 1024);

absl::Status ConvolutionInt8GroupedPerfTest(
    const BHWC& src_shape = BHWC(1, 32, 32, 1024), int dst_channels = 1024,
    int group_size = 64);

absl::Status ConvolutionSf16Wi4BatchedPerfTest(const BHWC& src_shape,
                                               int dst_channels,
                                               OHWI scale_zp_shape);

absl::Status ConvolutionInt4PerfTest(const BHWC& src_shape = BHWC(1, 32, 32,
                                                                  1024),
                                     int dst_channels = 1024);

absl::Status ConvMoEPerfTest(int seq_size, int src_channels, int dst_channels,
                             int num_experts, int num_active_experts);

absl::Status SoftmaxPerfTest(const BHWC& shape = BHWC(1, 1, 4096, 4096),
                             bool reduce_only = false);

absl::Status ConvSoftmaxConvPerfTest();

// scale_zp_shape defines dequantization area
// if scale_zp_shape = 1x1 -> scalar quantization
// if scale_zp_shape = Ox1 -> linear per output
// if scale_zp_shape = OxK -> blocked/grouped
absl::Status FullyConnectedOptimalWGSize(CalculationsPrecision precision,
                                         DataType weights_type,
                                         const BHWC& src_shape,
                                         int dst_channels,
                                         OHWI scale_zp_shape = OHWI(1, 1));

absl::Status FullyConnectedPerfTest(CalculationsPrecision precision,
                                    DataType weights_type,
                                    const BHWC& src_shape, int dst_channels,
                                    OHWI scale_zp_shape = OHWI(1, 1));

absl::Status FullyConnectedWeightsBatchIdsPerfTest(
    CalculationsPrecision precision, DataType weights_type,
    const BHWC& src_shape, int dst_channels, int batch_size,
    int active_ids_size, OHWI scale_zp_shape = OHWI(1, 1));

absl::Status FullyConnectedInt4Sparse2x4PerfTest(
    CalculationsPrecision precision, const BHWC& src_shape, int dst_channels);

absl::Status DepthwiseConvPerfTest(const BHWC& src_shape,
                                   const HW& kernel_size = HW(3, 3),
                                   const HW& strides = HW(1, 1),
                                   const HW& dilation = HW(1, 1));

absl::Status QuantizationPerfTest(const BHWC& src_shape, DataType src_type,
                                  PackedType dst_type, bool calculate_sum);
}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_TESTING_PERF_UTIL_H_
