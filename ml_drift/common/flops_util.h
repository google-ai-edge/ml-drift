// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_COMMON_FLOPS_UTIL_H_
#define ML_DRIFT_COMMON_FLOPS_UTIL_H_

#include <cstdint>

#include "ml_drift/common/shape.h"

namespace ml_drift {

uint64_t GetConvolutionFlops(const BHWC& dst_shape, const OHWI& weights_shape);
uint64_t GetConvolutionWinograd3x3TileNxNFlops(const BHWC& dst_shape,
                                               const OHWI& weights_shape,
                                               int tile_size);

uint64_t GetConvolutionTransposedFlops(const BHWC& src_shape,
                                       const OHWI& weights_shape);

uint64_t GetDepthwiseConvolutionFlops(const BHWC& dst_shape,
                                      const OHWI& weights_shape);

uint64_t GetFullyConnectedFlops(const BHWC& dst_shape,
                                const OHWI& weights_shape);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_FLOPS_UTIL_H_
