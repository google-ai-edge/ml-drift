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

#ifndef ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_UTIL_H_

#include <string>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
namespace fc {

void AddRuntimeParam(const ConvRuntimeCheckDesc& runtime_check,
                     GPUOperation* op);

std::string ReadScaleZeroPointBlock(const OHWI& scale_zp_shape,
                                    bool has_zero_point, DataType weights_type);
std::string ReadScaleZeroPointLinear(const OHWI& scale_zp_shape,
                                     bool has_zero_point,
                                     DataType weights_type);
std::string ReadScaleZeroPointScalar(bool has_zero_point,
                                     DataType weights_type);

std::string WeightsScaleAddBias(const std::string& w_scale,
                                const std::string& w_bias, bool isI4O4,
                                bool use_fma);

std::string AccumulateFloat(const std::string& r_name,
                            const std::string& src_name,
                            CalculationsPrecision precision, bool isI4O4,
                            bool use_fma);

std::string AccumulateUint(const std::string& r_name,
                           const std::string& src_name,
                           const GpuInfo& gpu_info);
std::string AdjustUintSum(const std::string& r_name, DataType weights_type);

}  // namespace fc
}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_UTIL_H_
