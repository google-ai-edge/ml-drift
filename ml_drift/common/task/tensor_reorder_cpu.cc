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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ml_drift/common/types.h"

// Clang's vector extensions (`vector_size` + `__builtin_shufflevector`) give a
// single portable source for the block kernels that lowers to NEON
// zip/st4, SSE unpck/shufps, or WASM SIMD shuffles. Other compilers (MSVC)
// use the scalar fallbacks.
#if defined(__clang__)
#define MLD_TENSOR_REORDER_VECTOR_EXT 1
#else
#define MLD_TENSOR_REORDER_VECTOR_EXT 0
#endif

// On x86-64, f32 -> f16 conversion hardware (F16C) is optional: present on
// every mainstream core since 2012 but absent on Atom-derived parts shipped
// as late as 2021. One binary serves both, so this is the one place a runtime
// CPU check is used. arm64 has the conversion in the base ISA and needs none.
#if defined(__x86_64__) && defined(__clang__)
#include <immintrin.h>  // NOLINT
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define MLD_TENSOR_REORDER_X86_F16C 1
#else
#define MLD_TENSOR_REORDER_X86_F16C 0
#endif

namespace ml_drift {
namespace {

constexpr size_t kMaxDims = 8;

// Copies `bytes` bytes.
//
// With vector extensions this is an inline 16-byte vector loop rather than a
// memcpy call: libc memcpy switches to a streaming strategy for rows of a few
// KiB on some platforms (notably Apple's), which is markedly slower for the
// row sizes weight layouts produce, and the call itself matters for short
// rows. Measured on Apple silicon: for short rows, 64-byte chunks cut loop
// overhead (1.1-1.3x over XNNPACK's 16-byte loop), but for rows of 8 KiB and
// up the same 64-byte loop runs at half the bandwidth of the plain 16-byte
// loop, so long rows keep the 16-byte shape.
constexpr size_t kWideCopyLimit = 2048;

inline void CopyBytes(void* dst, const void* src, size_t bytes) {
#if MLD_TENSOR_REORDER_VECTOR_EXT
  typedef char V16 __attribute__((vector_size(16)));
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  if (bytes < kWideCopyLimit) {
    for (; bytes >= 64; bytes -= 64, d += 64, s += 64) {
      V16 c0, c1, c2, c3;
      std::memcpy(&c0, s, 16);
      std::memcpy(&c1, s + 16, 16);
      std::memcpy(&c2, s + 32, 16);
      std::memcpy(&c3, s + 48, 16);
      std::memcpy(d, &c0, 16);
      std::memcpy(d + 16, &c1, 16);
      std::memcpy(d + 32, &c2, 16);
      std::memcpy(d + 48, &c3, 16);
    }
  }
  for (; bytes >= 16; bytes -= 16, d += 16, s += 16) {
    V16 c;
    std::memcpy(&c, s, 16);
    std::memcpy(d, &c, 16);
  }
  if (bytes & 8) {
    uint64_t v;
    std::memcpy(&v, s, 8);
    std::memcpy(d, &v, 8);
    d += 8;
    s += 8;
  }
  if (bytes & 4) {
    uint32_t v;
    std::memcpy(&v, s, 4);
    std::memcpy(d, &v, 4);
    d += 4;
    s += 4;
  }
  if (bytes & 2) {
    uint16_t v;
    std::memcpy(&v, s, 2);
    std::memcpy(d, &v, 2);
    d += 2;
    s += 2;
  }
  if (bytes & 1) *d = *s;
#else
  std::memcpy(dst, src, bytes);
#endif
}

// A transpose rewritten in destination order: `shape[i]` is the size of
// destination dim `i`, `src_stride[i]` is the distance (in elements) between
// consecutive elements of that dim in the source. Size-1 dims are dropped and
// destination-adjacent dims that are also source-contiguous are merged, so the
// plan is the minimal loop nest.
struct TransposePlan {
  size_t num_dims = 0;
  size_t shape[kMaxDims];
  size_t src_stride[kMaxDims];
};

TransposePlan MakeTransposePlan(absl::Span<const size_t> src_shape,
                                absl::Span<const size_t> perm) {
  const size_t n = src_shape.size();
  ABSL_CHECK_EQ(perm.size(), n);
  ABSL_CHECK_LE(n, kMaxDims);

  size_t src_stride[kMaxDims];
  size_t stride = 1;
  for (size_t i = n; i > 0; --i) {
    src_stride[i - 1] = stride;
    stride *= src_shape[i - 1];
  }

  TransposePlan plan;
  uint32_t seen = 0;  // `perm` must be a permutation: each dim exactly once.
  for (size_t i = 0; i < n; ++i) {
    const size_t d = perm[i];
    ABSL_CHECK_LT(d, n);
    ABSL_CHECK_EQ(seen & (1u << d), 0u) << "Duplicate dim " << d << " in perm";
    seen |= 1u << d;
    const size_t size = src_shape[d];
    if (size == 1) continue;
    if (plan.num_dims > 0) {
      const size_t last = plan.num_dims - 1;
      // Merge if dst-adjacent dims are src-contiguous.
      if (plan.src_stride[last] == src_stride[d] * size) {
        plan.shape[last] *= size;
        plan.src_stride[last] = src_stride[d];
        continue;
      }
    }
    plan.shape[plan.num_dims] = size;
    plan.src_stride[plan.num_dims] = src_stride[d];
    ++plan.num_dims;
  }
  if (plan.num_dims == 0) {
    plan.shape[0] = 1;
    plan.src_stride[0] = 1;
    plan.num_dims = 1;
  }
  return plan;
}

// Walks every index of the plan dims listed in `dims` (in that order, last
// innermost), calling `kernel(src_ptr, dst_ptr)` once per index.
template <typename T, typename Kernel>
void ForEachIndex(const T* src, T* dst, const TransposePlan& plan,
                  const size_t* dst_stride, const size_t* dims, size_t num,
                  Kernel&& kernel) {
  size_t count[kMaxDims];
  size_t src_adv[kMaxDims];
  size_t dst_adv[kMaxDims];
  size_t total = 1;
  for (size_t k = 0; k < num; ++k) {
    const size_t d = dims[k];
    count[k] = plan.shape[d];
    src_adv[k] = plan.src_stride[d];
    dst_adv[k] = dst_stride[d];
    total *= count[k];
  }
  size_t idx[kMaxDims] = {0};
  const T* s = src;
  T* t = dst;
  for (size_t it = 0; it < total; ++it) {
    kernel(s, t);
    for (size_t k = num; k > 0; --k) {
      const size_t j = k - 1;
      s += src_adv[j];
      t += dst_adv[j];
      if (++idx[j] < count[j]) break;
      s -= src_adv[j] * count[j];
      t -= dst_adv[j] * count[j];
      idx[j] = 0;
    }
  }
}

// Runs `kernel(src_ptr, dst_ptr)` for every index of the outer dims `dims`
// (given in destination order) around a block kernel that moves
// `kernel_bytes` per call.
//
// The last outer dim is the one whose steps are adjacent in the destination;
// the outer dim with the smallest source stride is the one whose steps are
// adjacent in the source. When these differ, iterating either one innermost
// scatters the other side's accesses, so the two are tiled together: within a
// tile both sides touch a small set of cache lines that stay resident. The
// remaining dims are walked with a generic odometer.
template <typename T, typename Kernel>
void ForEachOuter(const T* src, T* dst, const TransposePlan& plan,
                  const size_t* dst_stride, const size_t* dims, size_t num,
                  size_t kernel_bytes, Kernel&& kernel) {
  if (num == 0) {
    kernel(src, dst);
    return;
  }
  const size_t it = num - 1;
  size_t is = 0;
  for (size_t k = 1; k < num; ++k) {
    if (plan.src_stride[dims[k]] < plan.src_stride[dims[is]]) is = k;
  }
  // Tile so that each side of a tile covers about 16 KiB.
  const size_t tile = std::clamp<size_t>((16u << 10) / kernel_bytes, 1, 16);

  if (is == it || tile == 1) {
    const size_t count = plan.shape[dims[it]];
    const size_t ss = plan.src_stride[dims[it]];
    const size_t ts = dst_stride[dims[it]];
    ForEachIndex(src, dst, plan, dst_stride, dims, num - 1,
                 [&](const T* s, T* t) {
                   // Local copies: captured references would be re-read from
                   // memory after every store (google3 builds without strict
                   // aliasing).
                   const size_t n = count, sa = ss, ta = ts;
                   for (size_t k = 0; k < n; ++k, s += sa, t += ta) {
                     kernel(s, t);
                   }
                 });
    return;
  }

  const size_t count_t = plan.shape[dims[it]];
  const size_t ss_t = plan.src_stride[dims[it]];
  const size_t ts_t = dst_stride[dims[it]];
  const size_t count_s = plan.shape[dims[is]];
  const size_t ss_s = plan.src_stride[dims[is]];
  const size_t ts_s = dst_stride[dims[is]];
  size_t rest[kMaxDims];
  size_t num_rest = 0;
  for (size_t k = 0; k < num; ++k) {
    if (k != is && k != it) rest[num_rest++] = dims[k];
  }
  ForEachIndex(
      src, dst, plan, dst_stride, rest, num_rest, [&](const T* s, T* t) {
        const size_t nt = count_t, sst = ss_t, tst = ts_t;
        const size_t ns = count_s, sss = ss_s, tss = ts_s;
        const size_t tl = tile;
        for (size_t t0 = 0; t0 < nt; t0 += tl) {
          const size_t t1 = std::min(t0 + tl, nt);
          for (size_t s0 = 0; s0 < ns; s0 += tl) {
            const size_t s1 = std::min(s0 + tl, ns);
            for (size_t ti = t0; ti < t1; ++ti) {
              const T* sp = s + ti * sst + s0 * sss;
              T* tp = t + ti * tst + s0 * tss;
              for (size_t si = s0; si < s1; ++si, sp += sss, tp += tss) {
                kernel(sp, tp);
              }
            }
          }
        }
      });
}

#if MLD_TENSOR_REORDER_VECTOR_EXT
// Interleaves one 16-byte vector (N lanes) from each of 4 source rows into 64
// contiguous destination bytes: dst[4 * k + r] = s<r>[k], 0 <= k < N. The
// store of the 4N-lane result is an interleaved store, which LLVM lowers to
// `st4` on NEON.
//
// Takes pointers rather than vectors: a vector wider than 16 bytes must not
// cross a function boundary, because how it is passed depends on which ISA
// extensions are enabled (-Wpsabi on x86).
template <typename T, size_t... I2, size_t... I4>
inline void Interleave4Vectors(const T* s0, const T* s1, const T* s2,
                               const T* s3, T* dst, std::index_sequence<I2...>,
                               std::index_sequence<I4...>) {
  constexpr size_t N = 16 / sizeof(T);
  typedef T VN __attribute__((vector_size(16)));
  VN r0, r1, r2, r3;
  std::memcpy(&r0, s0, sizeof(VN));
  std::memcpy(&r1, s1, sizeof(VN));
  std::memcpy(&r2, s2, sizeof(VN));
  std::memcpy(&r3, s3, sizeof(VN));
  // a = r0|r1 and b = r2|r3, 2N lanes each.
  const auto a = __builtin_shufflevector(r0, r1, I2...);  // NOLINT
  const auto b = __builtin_shufflevector(r2, r3, I2...);  // NOLINT
  const auto o =
      __builtin_shufflevector(a, b, ((I4 % 4) * N + I4 / 4)...);  // NOLINT
  static_assert(sizeof(o) == 64);
  std::memcpy(dst, &o, sizeof(o));
}
#endif

// Interleaves 4 source rows into the destination:
//   dst[4 * k + r] = src[r * src_row_stride + k], 0 <= k < count, 0 <= r < 4.
// `count` must be a multiple of 4. This is the 4x4 block transpose with a
// contiguous destination block, which covers every I4O4-style interleave.
template <typename T>
inline void Interleave4Rows(const T* src, size_t src_row_stride, T* dst,
                            size_t count) {
  const T* s0 = src;
  const T* s1 = src + src_row_stride;
  const T* s2 = src + 2 * src_row_stride;
  const T* s3 = src + 3 * src_row_stride;
  size_t k = 0;
#if MLD_TENSOR_REORDER_VECTOR_EXT
  constexpr size_t N = 16 / sizeof(T);
  const auto step = [&](size_t k) {
    Interleave4Vectors(s0 + k, s1 + k, s2 + k, s3 + k, dst + 4 * k,
                       std::make_index_sequence<2 * N>(),
                       std::make_index_sequence<4 * N>());
  };
  for (; k + 2 * N <= count; k += 2 * N) {
    step(k);
    step(k + N);
  }
  if (k + N <= count) {
    step(k);
    k += N;
  }
#endif
  for (; k < count; k += 4) {
    T* d = dst + 4 * k;
    for (size_t a = 0; a < 4; ++a) {
      d[4 * a + 0] = s0[k + a];
      d[4 * a + 1] = s1[k + a];
      d[4 * a + 2] = s2[k + a];
      d[4 * a + 3] = s3[k + a];
    }
  }
}

// Transposes a 4x4 block:
//   dst[a * dst_row_stride + b] = src[b * src_row_stride + a].
// Source is 4 rows (indexed by b) of 4 contiguous elements; destination is 4
// rows (indexed by a) of 4 contiguous elements.
template <typename T>
inline void Transpose4x4(const T* src, size_t src_row_stride, T* dst,
                         size_t dst_row_stride) {
#if MLD_TENSOR_REORDER_VECTOR_EXT
  if constexpr (sizeof(T) <= 2) {
    // 16 or 32 bytes total: gather the rows and do one shuffle.
    typedef T V4 __attribute__((vector_size(4 * sizeof(T))));
    typedef T V16 __attribute__((vector_size(16 * sizeof(T))));
    alignas(V16) T rows[16];
    std::memcpy(rows + 0, src + 0 * src_row_stride, sizeof(V4));
    std::memcpy(rows + 4, src + 1 * src_row_stride, sizeof(V4));
    std::memcpy(rows + 8, src + 2 * src_row_stride, sizeof(V4));
    std::memcpy(rows + 12, src + 3 * src_row_stride, sizeof(V4));
    V16 v;
    std::memcpy(&v, rows, sizeof(v));
    const V16 t = __builtin_shufflevector(  // NOLINT
        v, v, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    alignas(V16) T out[16];
    std::memcpy(out, &t, sizeof(t));
    std::memcpy(dst + 0 * dst_row_stride, out + 0, sizeof(V4));
    std::memcpy(dst + 1 * dst_row_stride, out + 4, sizeof(V4));
    std::memcpy(dst + 2 * dst_row_stride, out + 8, sizeof(V4));
    std::memcpy(dst + 3 * dst_row_stride, out + 12, sizeof(V4));
  } else {
    // Classic two-stage 4x4 transpose (same dataflow as _MM_TRANSPOSE4_PS /
    // NEON vtrn+vzip), one vector per row.
    typedef T V4 __attribute__((vector_size(4 * sizeof(T))));
    V4 r0, r1, r2, r3;
    std::memcpy(&r0, src + 0 * src_row_stride, sizeof(V4));
    std::memcpy(&r1, src + 1 * src_row_stride, sizeof(V4));
    std::memcpy(&r2, src + 2 * src_row_stride, sizeof(V4));
    std::memcpy(&r3, src + 3 * src_row_stride, sizeof(V4));
    const V4 t0 = __builtin_shufflevector(r0, r1, 0, 4, 1, 5);  // NOLINT
    const V4 t1 = __builtin_shufflevector(r0, r1, 2, 6, 3, 7);  // NOLINT
    const V4 t2 = __builtin_shufflevector(r2, r3, 0, 4, 1, 5);  // NOLINT
    const V4 t3 = __builtin_shufflevector(r2, r3, 2, 6, 3, 7);  // NOLINT
    const V4 o0 = __builtin_shufflevector(t0, t2, 0, 1, 4, 5);  // NOLINT
    const V4 o1 = __builtin_shufflevector(t0, t2, 2, 3, 6, 7);  // NOLINT
    const V4 o2 = __builtin_shufflevector(t1, t3, 0, 1, 4, 5);  // NOLINT
    const V4 o3 = __builtin_shufflevector(t1, t3, 2, 3, 6, 7);  // NOLINT
    std::memcpy(dst + 0 * dst_row_stride, &o0, sizeof(V4));
    std::memcpy(dst + 1 * dst_row_stride, &o1, sizeof(V4));
    std::memcpy(dst + 2 * dst_row_stride, &o2, sizeof(V4));
    std::memcpy(dst + 3 * dst_row_stride, &o3, sizeof(V4));
  }
#else
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      dst[a * dst_row_stride + b] = src[b * src_row_stride + a];
    }
  }
#endif
}

// Transposes a 2x2 block of 8-byte elements (one 16-byte vector per row):
//   dst[a * dst_row_stride + b] = src[b * src_row_stride + a].
template <typename T>
inline void Transpose2x2(const T* src, size_t src_row_stride, T* dst,
                         size_t dst_row_stride) {
  static_assert(sizeof(T) == 8);
#if MLD_TENSOR_REORDER_VECTOR_EXT
  typedef T V2 __attribute__((vector_size(16)));
  V2 r0, r1;
  std::memcpy(&r0, src, sizeof(V2));
  std::memcpy(&r1, src + src_row_stride, sizeof(V2));
  const V2 o0 = __builtin_shufflevector(r0, r1, 0, 2);  // NOLINT
  const V2 o1 = __builtin_shufflevector(r0, r1, 1, 3);  // NOLINT
  std::memcpy(dst, &o0, sizeof(V2));
  std::memcpy(dst + dst_row_stride, &o1, sizeof(V2));
#else
  const T a = src[0], b = src[1];
  const T c = src[src_row_stride], d = src[src_row_stride + 1];
  dst[0] = a;
  dst[1] = c;
  dst[dst_row_stride] = b;
  dst[dst_row_stride + 1] = d;
#endif
}

// Copies one row of `kBytes` contiguous bytes. A compile-time size lets the
// compiler emit a plain load/store pair instead of a memcpy call.
template <size_t kBytes>
inline void CopyRow(void* dst, const void* src) {
  std::memcpy(dst, src, kBytes);
}

// Runs `fn(copy_row)` with `copy_row(dst, src)` specialized for the common
// small row sizes used by weight layouts.
template <typename Fn>
void WithRowCopier(size_t row_bytes, Fn&& fn) {
  switch (row_bytes) {
    case 16:
      fn(CopyRow<16>);
      break;
    case 32:
      fn(CopyRow<32>);
      break;
    case 64:
      fn(CopyRow<64>);
      break;
    default:
      fn([row_bytes](void* d, const void* s) { CopyBytes(d, s, row_bytes); });
      break;
  }
}

template <typename T>
void TransposeImpl(const T* src, T* dst, const TransposePlan& plan);

// Re-runs the transpose with each contiguous `inner`-element row treated as
// one element of type `Wide`. Requires that `inner * sizeof(T) == sizeof(Wide)`
// and that all other strides are multiples of `inner` (always true: the inner
// dim is exactly the trailing source dims).
template <typename T, typename Wide>
void TransposeWidened(const T* src, T* dst, const TransposePlan& plan) {
  const size_t n = plan.num_dims;
  const size_t inner = plan.shape[n - 1];
  TransposePlan wide;
  wide.num_dims = n - 1;
  for (size_t d = 0; d + 1 < n; ++d) {
    wide.shape[d] = plan.shape[d];
    wide.src_stride[d] = plan.src_stride[d] / inner;
  }
  TransposeImpl(reinterpret_cast<const Wide*>(src),
                reinterpret_cast<Wide*>(dst), wide);
}

template <typename T>
void TransposeImpl(const T* src, T* dst, const TransposePlan& plan) {
  const size_t n = plan.num_dims;
  const size_t inner = plan.shape[n - 1];
  const size_t inner_stride = plan.src_stride[n - 1];

  size_t dst_stride[kMaxDims];
  dst_stride[n - 1] = 1;
  for (size_t d = n - 1; d > 0; --d) {
    dst_stride[d - 1] = dst_stride[d] * plan.shape[d];
  }

  size_t dims[kMaxDims];

  if (inner_stride == 1) {
    // Innermost destination dim is contiguous in source.
    if (n == 1) {
      std::memcpy(dst, src, inner * sizeof(T));
      return;
    }

    // Short rows: treat each row as one wider element and transpose those.
    // This turns e.g. an x8 O4I4 interleave into an x32 4x4 block transpose.
    const size_t row_bytes = inner * sizeof(T);
    const bool aligned = reinterpret_cast<uintptr_t>(src) % row_bytes == 0 &&
                         reinterpret_cast<uintptr_t>(dst) % row_bytes == 0;
    if (aligned && sizeof(T) < 8) {
      if (row_bytes == 2) {
        TransposeWidened<T, uint16_t>(src, dst, plan);
        return;
      }
      if (row_bytes == 4) {
        TransposeWidened<T, uint32_t>(src, dst, plan);
        return;
      }
      if (row_bytes == 8) {
        TransposeWidened<T, uint64_t>(src, dst, plan);
        return;
      }
    }

    // Rows of >= 16 bytes. If some other destination dim `c` continues the
    // source-contiguous run (src_stride == inner), tile over (c, n-2) so that
    // both the source reads and the destination writes stay within a few
    // cache lines; otherwise just copy row by row.
    const size_t rows = plan.shape[n - 2];
    const size_t row_src_stride = plan.src_stride[n - 2];
    size_t c = n;
    for (size_t d = 0; d + 2 < n; ++d) {
      if (plan.src_stride[d] == inner) {
        c = d;
        break;
      }
    }
    WithRowCopier(row_bytes, [&](auto copy_row) {
      if (c < n) {
        // Tiles of 16 rows of `c`. Rows shorter than a cache line share lines
        // across consecutive `c`s, so only a few source rows may be live at a
        // time (no cache-set aliasing for power-of-two row strides); longer
        // rows are consumed whole, so all rows are walked and the destination
        // is written sequentially.
        constexpr size_t kCTile = 16;
        const size_t row_tile = row_bytes < 64 ? 8 : rows;
        const size_t c_size = plan.shape[c];
        const size_t c_dst_stride = dst_stride[c];
        size_t num_outer = 0;
        for (size_t d = 0; d + 2 < n; ++d) {
          if (d != c) dims[num_outer++] = d;
        }
        ForEachIndex(
            src, dst, plan, dst_stride, dims, num_outer, [&](const T* s, T* t) {
              const size_t cs = c_size, cds = c_dst_stride;
              const size_t nr = rows, rss = row_src_stride;
              const size_t in = inner, rtl = row_tile;
              for (size_t ct = 0; ct < cs; ct += kCTile) {
                const size_t c_end = std::min(ct + kCTile, cs);
                for (size_t rt = 0; rt < nr; rt += rtl) {
                  const size_t r_end = std::min(rt + rtl, nr);
                  for (size_t ci = ct; ci < c_end; ++ci) {
                    for (size_t ri = rt; ri < r_end; ++ri) {
                      copy_row(t + ci * cds + ri * in, s + ci * in + ri * rss);
                    }
                  }
                }
              }
            });
      } else {
        for (size_t d = 0; d + 2 < n; ++d) dims[d] = d;
        ForEachIndex(src, dst, plan, dst_stride, dims, n - 2,
                     [&](const T* s, T* t) {
                       const size_t nr = rows, rss = row_src_stride;
                       const size_t in = inner;
                       for (size_t r = 0; r < nr; ++r) {
                         copy_row(t + r * in, s + r * rss);
                       }
                     });
      }
    });
    return;
  }

  // Find the destination dim that is contiguous in source, if any. Together
  // with the innermost destination dim it forms a 2-D transpose that is done
  // in 4-wide vector blocks. All the O4/I4 weight interleaves end up here.
  size_t c = n;
  for (size_t d = 0; d + 1 < n; ++d) {
    if (plan.src_stride[d] == 1) {
      c = d;
      break;
    }
  }
  if (c < n && plan.shape[c] % 4 == 0 && inner % 4 == 0) {
    const size_t c_size = plan.shape[c];
    const size_t c_dst_stride = dst_stride[c];
    const size_t kernel_bytes = c_size * inner * sizeof(T);
    size_t num_outer = 0;
    for (size_t d = 0; d + 1 < n; ++d) {
      if (d != c) dims[num_outer++] = d;
    }
    if (inner == 4 && c_dst_stride == 4) {
      // Destination block is contiguous: a plain 4-row interleave.
      ForEachOuter(src, dst, plan, dst_stride, dims, num_outer, kernel_bytes,
                   [&](const T* s, T* t) {
                     Interleave4Rows(s, inner_stride, t, c_size);
                   });
    } else if (inner == 4) {
      ForEachOuter(src, dst, plan, dst_stride, dims, num_outer, kernel_bytes,
                   [&](const T* s, T* t) {
                     const size_t cs = c_size, is = inner_stride;
                     const size_t cds = c_dst_stride;
                     for (size_t cb = 0; cb < cs; cb += 4) {
                       Transpose4x4(s + cb, is, t + cb * cds, cds);
                     }
                   });
    } else {
      // Both block dims are long: tile so the working set stays in L1 (32x32
      // blocks: 2-8 KiB per side). Smaller tiles leave too few iterations
      // per tile and the loop overhead dominates.
      constexpr size_t kTile = 32;
      static_assert(kTile % 4 == 0);
      ForEachOuter(src, dst, plan, dst_stride, dims, num_outer, kernel_bytes,
                   [&](const T* s, T* t) {
                     const size_t cs = c_size, is = inner_stride,
                                  cds = c_dst_stride;
                     const size_t in = inner;
                     for (size_t ct = 0; ct < cs; ct += kTile) {
                       const size_t c_end = std::min(ct + kTile, cs);
                       for (size_t jt = 0; jt < in; jt += kTile) {
                         const size_t j_end = std::min(jt + kTile, in);
            // 8-byte elements: on NEON (Cortex-A710/X2, Apple M), 2x2
            // blocks (one 16-byte vector per row, zip1/zip2) measure
            // 1.5x faster than the 4x4 block below; on x86 the 4x4 block
            // is ~10% faster.
#if MLD_TENSOR_REORDER_VECTOR_EXT && defined(__ARM_NEON)
                         constexpr bool kUse2x2 = sizeof(T) == 8;
#else
                constexpr bool kUse2x2 = false;
#endif
                         if constexpr (kUse2x2) {
                           // Walk 8 source rows at a time (a 64-byte line of
                           // every destination row), and within those, all
                           // columns of the tile; keeping few source rows live
                           // avoids cache-set aliasing when the row stride is a
                           // large power of two.
                           constexpr size_t kRows = 8;
                           for (size_t j0 = jt; j0 < j_end; j0 += kRows) {
                             const size_t j1 = std::min(j0 + kRows, j_end);
                             for (size_t cb = ct; cb < c_end; cb += 2) {
                               for (size_t jb = j0; jb < j1; jb += 2) {
                                 Transpose2x2(s + cb + jb * is, is,
                                              t + cb * cds + jb, cds);
                               }
                             }
                           }
                         } else {
                           // Inner loop along c: streams 4 source rows
                           // sequentially (prefetcher-friendly) and fills the
                           // L1-resident destination tile 32 bytes at a time.
                           for (size_t jb = jt; jb < j_end; jb += 4) {
                             for (size_t cb = ct; cb < c_end; cb += 4) {
                               Transpose4x4(s + cb + jb * is, is,
                                            t + cb * cds + jb, cds);
                             }
                           }
                         }
                       }
                     }
                   });
    }
    return;
  }

  // Generic strided gather.
  for (size_t d = 0; d + 1 < n; ++d) dims[d] = d;
  ForEachIndex(src, dst, plan, dst_stride, dims, n - 1, [&](const T* s, T* t) {
    const size_t in = inner, is = inner_stride;
    for (size_t j = 0; j < in; ++j) {
      t[j] = s[j * is];
    }
  });
}

// Writes `bytes` bytes of the `sizeof(T)`-periodic fill pattern. `bytes` is a
// multiple of sizeof(T), so every chunk below starts in phase with the pattern.
template <typename T>
inline char* FillBytes(char* d, size_t bytes, T value) {
  // `value` replicated across 64 bits. Built with shifts rather than a lane
  // loop: at -Oz (Android) a lane loop is not unrolled and goes through the
  // stack once per lane, per call.
  uint64_t pattern = value;
  for (size_t bits = sizeof(T) * 8; bits < 64; bits *= 2) {
    pattern |= pattern << bits;
  }
#if MLD_TENSOR_REORDER_VECTOR_EXT
  typedef uint64_t V2 __attribute__((vector_size(16)));
  const V2 v = {pattern, pattern};
  for (; bytes >= 16; bytes -= 16, d += 16) std::memcpy(d, &v, 16);
#endif
  for (; bytes >= 8; bytes -= 8, d += 8) std::memcpy(d, &pattern, 8);
  if (bytes & 4) {
    std::memcpy(d, &pattern, 4);
    d += 4;
  }
  if (bytes & 2) {
    std::memcpy(d, &pattern, 2);
    d += 2;
  }
  if (bytes & 1) {
    std::memcpy(d, &pattern, 1);
    d += 1;
  }
  return d;
}

inline char* CopyRowBytes(char* d, const char* s, size_t bytes) {
  CopyBytes(d, s, bytes);
  return d + bytes;
}

// Pads `rows` rows: each destination row is `pre_bytes` of fill, `row_bytes`
// copied from the source, then `post_bytes` of fill.
template <typename T>
char* PadRows(size_t rows, size_t pre_bytes, size_t row_bytes,
              size_t post_bytes, const char* src, size_t src_stride, char* dst,
              T value) {
  for (; rows > 0; --rows, src += src_stride) {
    dst = FillBytes(dst, pre_bytes, value);
    dst = CopyRowBytes(dst, src, row_bytes);
    dst = FillBytes(dst, post_bytes, value);
  }
  return dst;
}

struct PadPlan {
  size_t num_dims;
  size_t shape[kMaxDims];
  size_t pre[kMaxDims];
  size_t post[kMaxDims];
  // Number of elements in one "row" below dim d, for source and destination.
  size_t src_row[kMaxDims];
  size_t dst_row[kMaxDims];
};

template <typename T>
char* PadDim(size_t d, const PadPlan& p, const char* src, char* dst, T value) {
  const size_t n = p.num_dims;
  dst = FillBytes(dst, p.pre[d] * p.dst_row[d] * sizeof(T), value);
  if (d + 1 == n) {
    dst = CopyRowBytes(dst, src, p.shape[d] * sizeof(T));
  } else if (d + 2 == n) {
    dst = PadRows(p.shape[d], p.pre[d + 1] * sizeof(T),
                  p.shape[d + 1] * sizeof(T), p.post[d + 1] * sizeof(T), src,
                  p.src_row[d] * sizeof(T), dst, value);
  } else {
    for (size_t i = 0; i < p.shape[d]; ++i) {
      dst = PadDim(d + 1, p, src + i * p.src_row[d] * sizeof(T), dst, value);
    }
  }
  return FillBytes(dst, p.post[d] * p.dst_row[d] * sizeof(T), value);
}

template <typename T>
void ConstantPadImpl(const T* src, T* dst, absl::Span<const size_t> shape,
                     absl::Span<const size_t> pre_padding,
                     absl::Span<const size_t> post_padding, T pad_value) {
  const size_t num_dims = shape.size();
  ABSL_CHECK_GE(num_dims, 1u);
  ABSL_CHECK_LE(num_dims, kMaxDims);
  ABSL_CHECK_EQ(pre_padding.size(), num_dims);
  ABSL_CHECK_EQ(post_padding.size(), num_dims);

  // Drop unpadded size-1 dims and fold unpadded trailing dims into the last
  // dim, so each leaf copy is as long as possible.
  PadPlan p;
  size_t* sh = p.shape;
  size_t* pre = p.pre;
  size_t* post = p.post;
  size_t n = 0;
  for (size_t i = 0; i < num_dims; ++i) {
    if (shape[i] == 1 && pre_padding[i] == 0 && post_padding[i] == 0) continue;
    sh[n] = shape[i];
    pre[n] = pre_padding[i];
    post[n] = post_padding[i];
    ++n;
  }
  if (n == 0) {
    sh[0] = 1;
    pre[0] = 0;
    post[0] = 0;
    n = 1;
  }
  while (n > 1 && pre[n - 1] == 0 && post[n - 1] == 0) {
    const size_t inner = sh[n - 1];
    sh[n - 2] *= inner;
    pre[n - 2] *= inner;
    post[n - 2] *= inner;
    --n;
  }
  p.num_dims = n;
  p.src_row[n - 1] = 1;
  p.dst_row[n - 1] = 1;
  for (size_t i = n - 1; i > 0; --i) {
    p.src_row[i - 1] = p.src_row[i] * sh[i];
    p.dst_row[i - 1] = p.dst_row[i] * (pre[i] + sh[i] + post[i]);
  }
  PadDim(0, p, reinterpret_cast<const char*>(src), reinterpret_cast<char*>(dst),
         pad_value);
}

#if MLD_TENSOR_REORDER_X86_F16C
// True if the CPU has AVX and F16C and the OS preserves the YMM registers.
bool HasAvxAndF16c() {
#if defined(_MSC_VER)
  // __builtin_cpu_supports() needs compiler runtime symbols (__cpu_model) that
  // are not linked in MSVC-style builds, so query CPUID directly.
  int info[4];
  __cpuid(info, 1);
  constexpr int kOsxsave = 1 << 27;
  constexpr int kAvx = 1 << 28;
  constexpr int kF16c = 1 << 29;
  constexpr int kNeeded = kOsxsave | kAvx | kF16c;
  if ((info[2] & kNeeded) != kNeeded) return false;
  // XCR0 bits 1 and 2: the OS saves the XMM and YMM state.
  return (_xgetbv(0) & 0x6) == 0x6;
#else
  return __builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c");
#endif
}

__attribute__((target("avx,f16c"))) void ConvertF32ToF16F16c(const float* src,
                                                             half* dst,
                                                             size_t count) {
  static_assert(sizeof(half) == 2);
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 f = _mm256_loadu_ps(src + i);
    const __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);  // NOLINT
  }
  for (; i < count; ++i) {
    const uint16_t h = _cvtss_sh(src[i], _MM_FROUND_TO_NEAREST_INT);
    std::memcpy(dst + i, &h, sizeof(h));
  }
}
#endif  // MLD_TENSOR_REORDER_X86_F16C

}  // namespace

void TransposeCpu(const void* src, void* dst, size_t element_size,
                  absl::Span<const size_t> shape,
                  absl::Span<const size_t> perm) {
  const TransposePlan plan = MakeTransposePlan(shape, perm);
  // Transpose of an empty tensor (any dim is zero) is a no-op.
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return;
  switch (element_size) {
    case 1:
      TransposeImpl(static_cast<const uint8_t*>(src),
                    static_cast<uint8_t*>(dst), plan);
      break;
    case 2:
      TransposeImpl(static_cast<const uint16_t*>(src),
                    static_cast<uint16_t*>(dst), plan);
      break;
    case 4:
      TransposeImpl(static_cast<const uint32_t*>(src),
                    static_cast<uint32_t*>(dst), plan);
      break;
    case 8:
      TransposeImpl(static_cast<const uint64_t*>(src),
                    static_cast<uint64_t*>(dst), plan);
      break;
    default:
      ABSL_CHECK(false) << "Unsupported element size: " << element_size;
  }
}

void ConstantPadCpu(const void* src, void* dst, size_t element_size,
                    absl::Span<const size_t> shape,
                    absl::Span<const size_t> pre_padding,
                    absl::Span<const size_t> post_padding,
                    const void* pad_value) {
  switch (element_size) {
    case 1:
      ConstantPadImpl(static_cast<const uint8_t*>(src),
                      static_cast<uint8_t*>(dst), shape, pre_padding,
                      post_padding, *static_cast<const uint8_t*>(pad_value));
      break;
    case 2: {
      uint16_t v;
      std::memcpy(&v, pad_value, sizeof(v));
      ConstantPadImpl(static_cast<const uint16_t*>(src),
                      static_cast<uint16_t*>(dst), shape, pre_padding,
                      post_padding, v);
      break;
    }
    case 4: {
      uint32_t v;
      std::memcpy(&v, pad_value, sizeof(v));
      ConstantPadImpl(static_cast<const uint32_t*>(src),
                      static_cast<uint32_t*>(dst), shape, pre_padding,
                      post_padding, v);
      break;
    }
    case 8: {
      uint64_t v;
      std::memcpy(&v, pad_value, sizeof(v));
      ConstantPadImpl(static_cast<const uint64_t*>(src),
                      static_cast<uint64_t*>(dst), shape, pre_padding,
                      post_padding, v);
      break;
    }
    default:
      ABSL_CHECK(false) << "Unsupported element size: " << element_size;
  }
}

void ConvertF32ToF16Cpu(const float* src, half* dst, size_t count) {
#if MLD_TENSOR_REORDER_X86_F16C
  static const bool has_f16c = HasAvxAndF16c();
  if (has_f16c) {
    ConvertF32ToF16F16c(src, dst, count);
    return;
  }
#endif  // MLD_TENSOR_REORDER_X86_F16C
  // Hardware fcvt on arm64; software (third_party/FP16) elsewhere.
  for (size_t i = 0; i < count; ++i) {
    dst[i] = half(src[i]);
  }
}

}  // namespace ml_drift
