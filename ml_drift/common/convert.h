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

#ifndef ML_DRIFT_COMMON_CONVERT_H_
#define ML_DRIFT_COMMON_CONVERT_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

// Converts a tensor from BHWC layout to PHWC4 layout.
// PHWC4 layout is where channels are grouped by 4 in a row and P stands for
// a plane that was derived by dividing channels by 4.
absl::Status ConvertToPHWC4(absl::Span<const float> in, const BHWC& shape,
                            absl::Span<float> out);

// Returns the number of elements when shape is converted into PHWC4.
uint32_t GetElementsSizeForPHWC4(const BHWC& shape);

// Converts a tensor from PHWC4 layout to BHWC layout.
// Operation is opposite to ConvertToPHWC4.
absl::Status ConvertFromPHWC4(absl::Span<const float> in, const BHWC& shape,
                              absl::Span<float> out);

// Convenience wrapper around ConvertToPHWC4.
std::vector<float> ConvertToPHWC4(
    const Tensor<BHWC, DataType::FLOAT32>& tensor);
// Convenience wrapper around ConvertToPHWC4.
std::vector<float> ConvertToPHWC4(const Tensor<HWC, DataType::FLOAT32>& tensor);

// Returns the number of elements when shape is converted into PIOHW4.
uint32_t GetElementsSizeForPIOHW4(const OHWI& shape);

// Converts a tensor from OHWI layout to PIOHW4 layout.
// PIOHW4 layout re-arranges weights in groups by 4, where outer dimension is
// P which is OxI/4.
absl::Status ConvertToPIOHW4(absl::Span<const float> in, const OHWI& shape,
                             absl::Span<float> out);

// Convenience wrapper around ConvertToPIOHW4.
std::vector<float> ConvertToPIOHW4(
    const Tensor<OHWI, DataType::FLOAT32>& tensor);

// Returns the number of elements when shape is converted into PHWO4I4.
uint32_t GetElementsSizeForPHWO4I4(const OHWI& shape);

// Convenience wrapper around ConvertToPHWO4I4.
std::vector<float> ConvertToPHWO4I4(
    const Tensor<OHWI, DataType::FLOAT32>& tensor);

// Convenience wrapper around ConvertToPHWO4I4 for Transposed Convolution.
std::vector<float> ConvertToPHWO4I4Transposed(
    const Tensor<OHWI, DataType::FLOAT32>& tensor);

// Returns the (x,y,z) size for PHWO4I4 to access elements where each element
// consists of 4 values.
uint3 Get3DSizeForPHWO4I4(const OHWI& shape);

// Returns the number of elements when shape is converted into PHWO4I4.
uint32_t GetElementsSizeForPHWO4I4(const IHWO& shape);

// Converts a tensor from IHWO layout to PHWO4I4 layout.
// Layout is Po,H,W,OI4x4.
absl::Status ConvertToPHWO4I4(absl::Span<const float> in, const IHWO& shape,
                              absl::Span<float> out);

// Convenience wrapper around ConvertToPHWO4I4.
std::vector<float> ConvertToPHWO4I4(
    const Tensor<IHWO, DataType::FLOAT32>& tensor);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_CONVERT_H_
