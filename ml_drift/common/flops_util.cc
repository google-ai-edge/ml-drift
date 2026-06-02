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

#include "ml_drift/common/flops_util.h"

#include <cstdint>

namespace ml_drift {

uint64_t GetConvolutionFlops(const BHWC& dst_shape, const OHWI& weights_shape) {
  uint64_t dst_elements = dst_shape.b * dst_shape.h * dst_shape.w * dst_shape.c;
  // 2 flops per operation( s = a * b + s);
  return dst_elements * weights_shape.i * weights_shape.w * weights_shape.h * 2;
}

uint64_t GetConvolutionWinograd3x3TileNxNFlops(const BHWC& dst_shape,
                                               const OHWI& weights_shape,
                                               int tile_size) {
  int tile_size_outer = tile_size;
  int tile_size_inner = tile_size_outer - 2;
  double flops_reduction =
      3.0 * 3.0 * static_cast<double>(tile_size_inner * tile_size_inner) /
      static_cast<double>(tile_size_outer * tile_size_outer);
  return GetConvolutionFlops(dst_shape, weights_shape) / flops_reduction;
}

uint64_t GetConvolutionTransposedFlops(const BHWC& src_shape,
                                       const OHWI& weights_shape) {
  uint64_t elements = src_shape.b * src_shape.h * src_shape.w * weights_shape.o;
  // 2 flops per operation( s = a * b + s);
  return elements * weights_shape.i * weights_shape.w * weights_shape.h * 2;
}

uint64_t GetDepthwiseConvolutionFlops(const BHWC& dst_shape,
                                      const OHWI& weights_shape) {
  uint64_t dst_elements = dst_shape.b * dst_shape.h * dst_shape.w * dst_shape.c;
  // 2 flops per operation( s = a * b + s);
  return dst_elements * weights_shape.w * weights_shape.h * 2;
}

uint64_t GetFullyConnectedFlops(const BHWC& dst_shape,
                                const OHWI& weights_shape) {
  uint64_t dst_elements = dst_shape.b * dst_shape.h * dst_shape.w * dst_shape.c;
  // 2 flops per operation( s = a * b + s);
  return dst_elements * weights_shape.i * 2;
}

}  // namespace ml_drift
