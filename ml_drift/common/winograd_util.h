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

#ifndef ML_DRIFT_COMMON_WINOGRAD_UTIL_H_
#define ML_DRIFT_COMMON_WINOGRAD_UTIL_H_

#include <memory>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

// Matrices for Winograd trasformations received with method described here
// https://openreview.net/pdf?id=H1ZaRZVKg

// returns A transposed matrix(N, (N - 2)) as array for Winograd3x3TileNxN
// tile_size is outer tile size
// outer tile size = tile_size = N
// inner tile size = tile_size - 2 = N - 2
std::vector<float> AtMatrixForWinograd3x3TileNxN(int tile_size);

// returns B transposed matrix(N, N) as array for Winograd3x3TileNxN
// tile_size is outer tile size
// outer tile size = tile_size = N
// inner tile size = tile_size - 2 = N - 2
std::vector<float> BtMatrixForWinograd3x3TileNxN(int tile_size);

std::unique_ptr<float[]> GetTransposedMatrixForWinograd3(int width);

// tile_size is outer tile size
// outer tile size = tile_size = N
// inner tile size = tile_size - 2 = N - 2
void RearrangeWeightsToWinograd3x3TileNxN(
    const Tensor<OHWI, DataType::FLOAT32>& src_weights,
    Tensor<OHWI, DataType::FLOAT32>* dst_weights, int tile_size);

bool IsSuitableForWinograd3x3(const Convolution2DAttributes& attr);

// tile_size is outer tile size
// outer tile size = tile_size = N
// inner tile size = tile_size - 2 = N - 2
bool IsRecommendedForWinograd3x3TileNxN(const Convolution2DAttributes& attr,
                                        const GpuInfo& gpu_info,
                                        const BHWC& dst_shape, int tile_size);

OHWI GetWinograd3x3TileNxNWeightsShape(const OHWI& src_shape, int tile_size);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_WINOGRAD_UTIL_H_
