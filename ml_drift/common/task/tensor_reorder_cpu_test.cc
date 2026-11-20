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

#include "ml_drift/common/task/tensor_reorder_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace {

using ::testing::ElementsAreArray;

size_t Product(absl::Span<const size_t> v) {
  size_t p = 1;
  for (size_t x : v) p *= x;
  return p;
}

// Row-major strides for `shape`.
std::vector<size_t> Strides(absl::Span<const size_t> shape) {
  std::vector<size_t> strides(shape.size());
  size_t s = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = s;
    s *= shape[i - 1];
  }
  return strides;
}

// Advances a row-major multi-index over `shape`.
void Next(absl::Span<const size_t> shape, std::vector<size_t>& idx) {
  for (size_t i = shape.size(); i > 0; --i) {
    if (++idx[i - 1] < shape[i - 1]) break;
    idx[i - 1] = 0;
  }
}

// Naive reference transpose, element-wise, no fast paths.
void ReferenceTranspose(const uint8_t* src, uint8_t* dst, size_t element_size,
                        absl::Span<const size_t> shape,
                        absl::Span<const size_t> perm) {
  const size_t n = shape.size();
  const std::vector<size_t> src_stride = Strides(shape);
  std::vector<size_t> dst_shape(n);
  for (size_t i = 0; i < n; ++i) dst_shape[i] = shape[perm[i]];
  const size_t total = Product(shape);
  std::vector<size_t> idx(n, 0);
  for (size_t dst_i = 0; dst_i < total; ++dst_i) {
    size_t src_i = 0;
    for (size_t i = 0; i < n; ++i) src_i += idx[i] * src_stride[perm[i]];
    std::memcpy(dst + dst_i * element_size, src + src_i * element_size,
                element_size);
    Next(dst_shape, idx);
  }
}

// Naive reference constant pad, element-wise.
void ReferencePad(const uint8_t* src, uint8_t* dst, size_t element_size,
                  absl::Span<const size_t> shape, absl::Span<const size_t> pre,
                  absl::Span<const size_t> post, const void* pad_value) {
  const size_t n = shape.size();
  const std::vector<size_t> src_stride = Strides(shape);
  std::vector<size_t> dst_shape(n);
  for (size_t i = 0; i < n; ++i) dst_shape[i] = pre[i] + shape[i] + post[i];
  const size_t total = Product(dst_shape);
  std::vector<size_t> idx(n, 0);
  for (size_t dst_i = 0; dst_i < total; ++dst_i) {
    size_t src_i = 0;
    bool inside = true;
    for (size_t i = 0; i < n; ++i) {
      if (idx[i] < pre[i] || idx[i] >= pre[i] + shape[i]) {
        inside = false;
        break;
      }
      src_i += (idx[i] - pre[i]) * src_stride[i];
    }
    std::memcpy(dst + dst_i * element_size,
                inside ? src + src_i * element_size
                       : static_cast<const uint8_t*>(pad_value),
                element_size);
    Next(dst_shape, idx);
  }
}

// IEEE 754 binary32 -> binary16, round-to-nearest-even, written from the
// format definition so it is independent of both the FP16 library and the
// hardware conversion paths under test. NaN handling is left to the caller.
uint16_t ReferenceF32ToF16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint16_t sign = (u >> 16) & 0x8000;
  const uint32_t abs = u & 0x7fffffff;
  if (abs >= 0x7f800000) return sign | 0x7c00;  // inf
  const int exp = static_cast<int>(abs >> 23) - 127;
  const uint32_t mant = abs & 0x7fffff;
  if (exp < -14) {
    // Half subnormal: the result is (mant | 2^23) * 2^(exp - 23) in units of
    // 2^-24, i.e. shifted right by -(exp + 1) bits.
    const int shift = -(exp + 1);
    if (shift > 24) return sign;
    const uint32_t m = mant | 0x800000;
    uint32_t q = m >> shift;
    const uint32_t r = m & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (r > halfway || (r == halfway && (q & 1))) ++q;
    return sign | static_cast<uint16_t>(q);
  }
  uint32_t q = mant >> 13;
  const uint32_t r = mant & 0x1fff;
  int e = exp + 15;
  if (r > 0x1000 || (r == 0x1000 && (q & 1))) {
    if (++q == 0x400) {
      q = 0;
      ++e;
    }
  }
  if (e >= 31) return sign | 0x7c00;
  return sign | static_cast<uint16_t>(e << 10) | static_cast<uint16_t>(q);
}

std::vector<uint8_t> RandomBytes(size_t count, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> v(count);
  for (auto& b : v) b = static_cast<uint8_t>(rng());
  return v;
}

class TensorReorderCpuTest : public ::testing::Test {
 protected:
  void CheckTranspose(absl::Span<const size_t> shape,
                      absl::Span<const size_t> perm, size_t element_size) {
    SCOPED_TRACE(::testing::Message()
                 << "element_size=" << element_size
                 << " shape=" << ::testing::PrintToString(shape)
                 << " perm=" << ::testing::PrintToString(perm));
    const size_t bytes = Product(shape) * element_size;
    const std::vector<uint8_t> src = RandomBytes(bytes, 42);
    std::vector<uint8_t> expected(bytes), actual(bytes);
    ReferenceTranspose(src.data(), expected.data(), element_size, shape, perm);
    TransposeCpu(src.data(), actual.data(), element_size, shape, perm);
    EXPECT_THAT(actual, ElementsAreArray(expected));
  }

  void CheckPad(absl::Span<const size_t> shape, absl::Span<const size_t> pre,
                absl::Span<const size_t> post, size_t element_size) {
    SCOPED_TRACE(::testing::Message()
                 << "element_size=" << element_size
                 << " shape=" << ::testing::PrintToString(shape)
                 << " pre=" << ::testing::PrintToString(pre)
                 << " post=" << ::testing::PrintToString(post));
    std::vector<size_t> dst_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      dst_shape[i] = pre[i] + shape[i] + post[i];
    }
    const size_t src_bytes = Product(shape) * element_size;
    const size_t dst_bytes = Product(dst_shape) * element_size;
    const std::vector<uint8_t> src = RandomBytes(src_bytes, 7);
    const uint64_t pad_value = 0xA5A5A5A5A5A5A5A5ull;
    std::vector<uint8_t> expected(dst_bytes), actual(dst_bytes);
    ReferencePad(src.data(), expected.data(), element_size, shape, pre, post,
                 &pad_value);
    ConstantPadCpu(src.data(), actual.data(), element_size, shape, pre, post,
                   &pad_value);
    EXPECT_THAT(actual, ElementsAreArray(expected));
  }
};

// The 6-D reorders used by weights_conversion.cc. Dims are
// {dst_groups, group_size, 4 (o), spatial, src_groups, 4 (i)}.
const std::vector<std::vector<size_t>>& WeightsPerms() {
  static const auto* perms = new std::vector<std::vector<size_t>>{
      {0, 3, 4, 1, 5, 2},  // kOSpatialIOGroupI4O4
      {3, 4, 0, 1, 5, 2},  // k2DYIsSpatialIOAndXIsOGroupI4O4
      {0, 3, 4, 1, 2, 5},  // kOSpatialIOGroupO4I4
      {5, 3, 4, 0, 1, 2},  // k2DX4I4YIsSpatialIAndXIsOOGroupO4
      {2, 3, 4, 0, 1, 5},  // k2DX4O4YIsSpatialIAndXIsOOGroupI4
      {0, 4, 3, 1, 2, 5},  // kCustomGroups O4I4
      {0, 4, 3, 1, 5, 2},  // kCustomGroups I4O4
  };
  return *perms;
}

TEST_F(TensorReorderCpuTest, WeightsLayouts6D) {
  for (size_t element_size : {1, 2, 4}) {
    for (const auto& perm : WeightsPerms()) {
      CheckTranspose({3, 2, 4, 5, 3, 4}, perm, element_size);
      CheckTranspose({1, 1, 4, 1, 1, 4}, perm, element_size);
      CheckTranspose({2, 4, 4, 1, 7, 4}, perm, element_size);
    }
  }
}

TEST_F(TensorReorderCpuTest, Ohwdi5D) {
  for (size_t element_size : {1, 2, 4}) {
    CheckTranspose({5, 2, 3, 4, 6}, {0, 3, 1, 2, 4}, element_size);
    CheckTranspose({1, 2, 3, 4, 1}, {0, 3, 1, 2, 4}, element_size);
  }
}

TEST_F(TensorReorderCpuTest, Identity) {
  for (size_t element_size : {1, 2, 4, 8}) {
    CheckTranspose({3, 5, 7}, {0, 1, 2}, element_size);
    CheckTranspose({1}, {0}, element_size);
    CheckTranspose({1, 1, 1}, {2, 0, 1}, element_size);
  }
}

TEST_F(TensorReorderCpuTest, Generic2D) {
  for (size_t element_size : {1, 2, 4, 8}) {
    CheckTranspose({3, 5}, {1, 0}, element_size);
    CheckTranspose({4, 4}, {1, 0}, element_size);
    CheckTranspose({17, 9}, {1, 0}, element_size);
    // Larger than one tile in both dimensions, with ragged edges.
    CheckTranspose({67, 45}, {1, 0}, element_size);
  }
}

TEST_F(TensorReorderCpuTest, RandomShapesAndPerms) {
  std::mt19937 rng(1234);
  for (int iter = 0; iter < 200; ++iter) {
    const size_t n = 1 + rng() % 6;
    std::vector<size_t> shape(n), perm(n);
    for (size_t i = 0; i < n; ++i) {
      shape[i] = 1 + rng() % 6;
      perm[i] = i;
    }
    std::shuffle(perm.begin(), perm.end(), rng);
    const size_t element_size = size_t{1} << (rng() % 4);
    CheckTranspose(shape, perm, element_size);
  }
}

TEST_F(TensorReorderCpuTest, EmptyTensorIsNoOp) {
  for (size_t element_size : {1, 2, 4, 8}) {
    CheckTranspose({0}, {0}, element_size);
    CheckTranspose({0, 4}, {1, 0}, element_size);
    CheckTranspose({4, 0}, {1, 0}, element_size);
    CheckTranspose({2, 0, 4}, {0, 2, 1}, element_size);
    CheckTranspose({2, 4, 0}, {0, 2, 1}, element_size);
    CheckTranspose({0, 2, 4}, {2, 1, 0}, element_size);
    CheckTranspose({3, 2, 4, 0, 3, 4}, {0, 3, 4, 1, 5, 2}, element_size);
    CheckTranspose({3, 2, 4, 5, 0, 4}, {5, 3, 4, 0, 1, 2}, element_size);
  }
}

TEST_F(TensorReorderCpuTest, RejectsInvalidPerm) {
  const uint32_t src[16] = {};
  uint32_t dst[16];
  EXPECT_DEATH(TransposeCpu(src, dst, 4, {8, 2}, {0, 0}), "Duplicate dim");
  EXPECT_DEATH(TransposeCpu(src, dst, 4, {8, 2}, {0, 2}), "");
  EXPECT_DEATH(TransposeCpu(src, dst, 4, {8, 2}, {0}), "");
}

TEST_F(TensorReorderCpuTest, Pad4DPostOnly) {
  // weights_conversion.cc: OHWI padded on O and I.
  for (size_t element_size : {1, 2, 4}) {
    CheckPad({5, 2, 3, 6}, {0, 0, 0, 0}, {3, 0, 0, 2}, element_size);
    CheckPad({8, 1, 1, 8}, {0, 0, 0, 0}, {0, 0, 0, 0}, element_size);
  }
}

TEST_F(TensorReorderCpuTest, Pad5DPostOnly) {
  for (size_t element_size : {1, 2, 4}) {
    CheckPad({5, 2, 3, 2, 6}, {0, 0, 0, 0, 0}, {3, 0, 0, 0, 2}, element_size);
  }
}

TEST_F(TensorReorderCpuTest, PadGeneric) {
  for (size_t element_size : {1, 2, 4}) {
    CheckPad({3}, {2}, {1}, element_size);
    CheckPad({2, 3}, {1, 2}, {3, 1}, element_size);
    CheckPad({2, 3, 4}, {1, 0, 2}, {0, 2, 1}, element_size);
    CheckPad({1, 1, 1}, {1, 1, 1}, {1, 1, 1}, element_size);
    // Empty source: the destination is all padding (or empty).
    CheckPad({0}, {2}, {1}, element_size);
    CheckPad({0, 3}, {1, 0}, {0, 0}, element_size);
    CheckPad({2, 0}, {0, 0}, {0, 2}, element_size);
    CheckPad({2, 0, 3}, {1, 0, 1}, {0, 0, 0}, element_size);
    CheckPad({2, 0, 3}, {0, 0, 0}, {0, 0, 0}, element_size);
  }
}

TEST_F(TensorReorderCpuTest, ConvertF32ToF16RoundsToNearestEven) {
  std::vector<float> src = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      0.1f,
      65504.0f,        // max half
      65520.0f,        // rounds to inf
      -70000.0f,       // overflow
      6.1035156e-05f,  // min normal half
      5.9604645e-08f,  // min subnormal half
      2.9802322e-08f,  // half of min subnormal: tie, rounds to even (0)
      1e-10f,          // underflow to zero
      3.0517578e-05f,  // subnormal
      6.0975552e-05f,  // max subnormal
      1.00048828125f,  // tie: rounds to even (1.0)
      1.00146484375f,  // tie: rounds to even (1.001953125)
      2047.0f / 1024,  // rounds up into the next binade
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
  };
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (int i = 0; i < 10000; ++i) src.push_back(dist(rng));
  std::uniform_real_distribution<float> small(-1e-4f, 1e-4f);
  for (int i = 0; i < 10000; ++i) src.push_back(small(rng));
  std::uniform_real_distribution<float> large(-1e5f, 1e5f);
  for (int i = 0; i < 1000; ++i) src.push_back(large(rng));

  std::vector<half> actual(src.size());
  ConvertF32ToF16Cpu(src.data(), actual.data(), src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    uint16_t a;
    std::memcpy(&a, &actual[i], sizeof(a));
    EXPECT_EQ(a, ReferenceF32ToF16(src[i])) << "i=" << i << " src=" << src[i];
  }
}

TEST_F(TensorReorderCpuTest, ConvertF32ToF16Nan) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  half out;
  ConvertF32ToF16Cpu(&nan, &out, 1);
  EXPECT_TRUE(std::isnan(static_cast<float>(out)));
}

}  // namespace
}  // namespace ml_drift
