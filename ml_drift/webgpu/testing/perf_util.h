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

#ifndef ML_DRIFT_WEBGPU_TESTING_PERF_UTIL_H_
#define ML_DRIFT_WEBGPU_TESTING_PERF_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {
namespace webgpu {
absl::Status ConvolutionPerfTest(CalculationsPrecision precision,
                                 const BHWC& src_shape = BHWC(1, 32, 32, 1024),
                                 int dst_channels = 1024,
                                 const HW& kernel_size = HW(1, 1));

absl::Status ConvolutionInt8PerfTest(const BHWC& src_shape = BHWC(1, 32, 32,
                                                                  1024),
                                     int dst_channels = 1024);

absl::Status FullyConnectedPerfTest(CalculationsPrecision precision,
                                    DataType weights_type,
                                    const BHWC& src_shape, int dst_channels,
                                    OHWI scale_zp_shape = OHWI(1, 1));

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_TESTING_PERF_UTIL_H_
