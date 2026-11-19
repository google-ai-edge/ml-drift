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

#ifndef ML_DRIFT_COMMON_PRECISION_H_
#define ML_DRIFT_COMMON_PRECISION_H_

#include <array>
#include <string>

#include "ml_drift/common/data_type.h"

namespace ml_drift {

enum class CalculationsPrecision {
  kF32,
  kF32F16,
  kF16,
};
// F32 - all data and all math ops in F32
// F16 - all data and all math ops in F16
// F32_F16 - as F16, but some operations (Convolution,
// DepthwiseConvolution, FullyConnected, ConvolutionTransposed)
// have accumulator in F32 and usually it calculates 4 mads in F16, sum them,
// than converts this partial sum to F32 and add to accumulator.

DataType DeduceDataTypeFromPrecision(CalculationsPrecision precision);

constexpr std::array<CalculationsPrecision, 3> GetCalculationsPrecisions() {
  return {CalculationsPrecision::kF32, CalculationsPrecision::kF16,
          CalculationsPrecision::kF32F16};
}
std::string ToString(CalculationsPrecision precision);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_PRECISION_H_
