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

#include "ml_drift/common/winograd_util.h"

#include <cmath>
#include <memory>
#include <variant>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {

// Winograd matrices as described in https://openreview.net/pdf?id=H1ZaRZVKg

struct Point2d {
  float x;
  float y;
};

void FillPoints(Point2d* points, int n) {
  constexpr float kDelta = 0.7071067811865476f;  // sqrt(2) / 2
  *points++ = {/*x=*/0.0f, /*y=*/1.0f};
  if (n == 6) {
    *points++ = {/*x=*/kDelta, /*y=*/1.0f};
    *points++ = {/*x=*/-kDelta, /*y=*/1.0f};
    *points++ = {/*x=*/1.0f / kDelta, /*y=*/1.0f};
    *points++ = {/*x=*/-1.0f / kDelta, /*y=*/1.0f};
  } else if (n == 8) {
    *points++ = {/*x=*/0.6f, /*y=*/1.0f};
    *points++ = {/*x=*/1.1f, /*y=*/1.0f};
    *points++ = {/*x=*/1.6f, /*y=*/1.0f};
    *points++ = {/*x=*/-0.6f, /*y=*/1.0f};
    *points++ = {/*x=*/-1.1f, /*y=*/1.0f};
    *points++ = {/*x=*/-1.6f, /*y=*/1.0f};
  } else if (n == 10) {
    *points++ = {/*x=*/0.5f, /*y=*/1.0f};
    *points++ = {/*x=*/0.9f, /*y=*/1.0f};
    *points++ = {/*x=*/1.0f / 0.9f, /*y=*/1.0f};
    *points++ = {/*x=*/2.0f, /*y=*/1.0f};
    *points++ = {/*x=*/-0.5f, /*y=*/1.0f};
    *points++ = {/*x=*/-0.9f, /*y=*/1.0f};
    *points++ = {/*x=*/-1.0f / 0.9f, /*y=*/1.0f};
    *points++ = {/*x=*/-2.0f, /*y=*/1.0f};
  } else {
    for (int i = 0; i < (n - 1) / 2; ++i) {
      *points++ = {/*x=*/kDelta * (i + 1.0f), /*y=*/1.0f};
      *points++ = {/*x=*/-kDelta * (i + 1.0f), /*y=*/1.0f};
    }
  }
  *points = {/*x=*/1.0f, /*y=*/0.0f};
}

std::vector<float> GenerateVandermondeMatrix(
    const std::vector<Point2d>& points, int width, int height) {
  std::vector<float> result(height * width);
  for (int w = 0; w < width; ++w) {
    for (int h = 0; h < height; ++h) {
      result[h * width + w] = std::pow(points[w].x, 1.0 * h) *
                              std::pow(points[w].y, (height - 1.0) - h);
    }
  }
  return result;
}

std::vector<float> GetTransposedMatrixForWinograd(int width, int height) {
  std::vector<Point2d> points(width);
  FillPoints(&points[0], width);
  return GenerateVandermondeMatrix(points, width, height);
}

std::unique_ptr<float[]> GenerateVandermondeMatrix3(const Point2d* points,
                                                    int width) {
  auto result = std::make_unique<float[]>(3 * width);
  for (int i = 0; i < width; ++i) {
    result[i] = points[i].y * points[i].y;
    result[i + width] = points[i].x * points[i].y;
    result[i + 2 * width] = points[i].x * points[i].x;
  }
  return result;
}

std::vector<float> GetInversedMatrixForWinograd(int rank) {
  auto matrix = GetTransposedMatrixForWinograd(rank, rank);
  std::vector<float> inverted(rank * rank);
  for (int i = 0; i < rank; ++i) {
    inverted[i * rank + i] = 1.0f;
  }

  for (int i = 1; i < rank - 1; ++i) {
    float inv_t = 1.0f / matrix[i * rank + i];
    for (int x = i; x < rank; ++x) {
      matrix[i * rank + x] *= inv_t;
    }
    for (int x = 0; x < rank; ++x) {
      inverted[i * rank + x] *= inv_t;
    }

    for (int y = 0; y < rank; ++y) {
      if (y == i) continue;
      float t = matrix[y * rank + i];
      for (int x = i; x < rank; ++x) {
        matrix[y * rank + x] -= t * matrix[i * rank + x];
      }
      for (int x = 0; x < rank; ++x) {
        inverted[y * rank + x] -= t * inverted[i * rank + x];
      }
    }
  }

  return inverted;
}

}  // namespace

std::unique_ptr<float[]> GetTransposedMatrixForWinograd3(int width) {
  auto points = std::make_unique<Point2d[]>(width);
  FillPoints(points.get(), width);
  return GenerateVandermondeMatrix3(points.get(), width);
}

std::vector<float> AtMatrixForWinograd3x3TileNxN(int tile_size) {
  auto mat_f64 = GetTransposedMatrixForWinograd(tile_size, tile_size - 2);
  return std::vector<float>(mat_f64.begin(), mat_f64.end());
}

std::vector<float> BtMatrixForWinograd3x3TileNxN(int tile_size) {
  auto mat_f64 = GetInversedMatrixForWinograd(tile_size);
  return std::vector<float>(mat_f64.begin(), mat_f64.end());
}

void RearrangeWeightsToWinograd3x3TileNxN(
    const Tensor<OHWI, DataType::FLOAT32>& src_weights,
    Tensor<OHWI, DataType::FLOAT32>* dst_weights, int tile_size) {
  OHWI dst_shape;
  dst_shape.o = src_weights.shape.o;
  dst_shape.h = tile_size;
  dst_shape.w = tile_size;
  dst_shape.i = src_weights.shape.i;
  dst_weights->shape = dst_shape;
  dst_weights->data.resize(dst_shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  const int stride = src_weights.shape.i;

  auto g_t = GetTransposedMatrixForWinograd3(tile_size);
  float* d2 = &dst_weights->data[0];
  for (int d = 0; d < src_weights.shape.o; ++d) {
    for (int i = 0; i < tile_size; ++i) {
      const float a0 = g_t[i];
      const float a1 = g_t[i + tile_size];
      const float a2 = g_t[i + 2 * tile_size];
      for (int k = 0; k < tile_size; ++k) {
        const float c0 = g_t[k];
        const float c1 = g_t[k + tile_size];
        const float c2 = g_t[k + 2 * tile_size];
        const float* in = src_weights.data.data() + 9 * d * stride;
        for (int s = 0; s < stride; ++s) {
          *d2++ = a0 * c0 * in[s] +               //
                  a0 * c1 * in[s + stride] +      //
                  a0 * c2 * in[s + 2 * stride] +  //
                  a1 * c0 * in[s + 3 * stride] +  //
                  a1 * c1 * in[s + 4 * stride] +  //
                  a1 * c2 * in[s + 5 * stride] +  //
                  a2 * c0 * in[s + 6 * stride] +  //
                  a2 * c1 * in[s + 7 * stride] +  //
                  a2 * c2 * in[s + 8 * stride];
        }
      }
    }
  }
}

bool IsSuitableForWinograd3x3(const Convolution2DAttributes& attr) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return weights_shape.w == 3 && weights_shape.h == 3 &&
         attr.dilations == HW(1, 1) && attr.strides == HW(1, 1) &&
         attr.groups == 1;
}

bool IsRecommendedForWinograd3x3TileNxN(const Convolution2DAttributes& attr,
                                        const GpuInfo& gpu_info,
                                        const BHWC& dst_shape, int tile_size) {
  const int tile_size_inner = tile_size - 2;
  const int tiles_x = DivideRoundUp(dst_shape.w, tile_size_inner);
  const int tiles_y = DivideRoundUp(dst_shape.h, tile_size_inner);
  const int total_tiles = tiles_x * tiles_y * dst_shape.b;
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  const int src_depth = DivideRoundUp(weights_shape.i, 4);
  const int dst_depth = DivideRoundUp(weights_shape.o, 4);
  int min_src_depth = 16;
  int min_dst_depth = 16;
  if (gpu_info.IsAdreno()) {
    min_src_depth = 32;
    min_dst_depth = 32;
  } else if (gpu_info.IsAMD()) {
    min_dst_depth = 8;
  }
  int min_combined_depth = min_src_depth * min_dst_depth;
  if (gpu_info.IsPowerVR() &&
      (gpu_info.powervr_info.IsImgCxx() || gpu_info.powervr_info.IsImgDxx())) {
    min_combined_depth = 16 * 32;
  }
  int min_tiles = 32;
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx()) {
      min_tiles = 32;
    } else if (gpu_info.adreno_info.IsAdreno4xx() ||
               gpu_info.adreno_info.IsAdreno5xx()) {
      min_tiles = 64;
    } else {
      min_tiles = 128;
    }
  }
  const bool recommended_channels = src_depth >= min_src_depth &&
                                    dst_depth >= min_dst_depth &&
                                    src_depth * dst_depth >= min_combined_depth;
  const bool recommended_hw = total_tiles >= min_tiles;
  return recommended_channels && recommended_hw;
}

OHWI GetWinograd3x3TileNxNWeightsShape(const OHWI& src_shape, int tile_size) {
  return OHWI(src_shape.o, tile_size, tile_size, src_shape.i);
}

}  // namespace ml_drift
