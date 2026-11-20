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

#ifndef ML_DRIFT_COMMON_TASK_TENSOR_REORDER_CPU_H_
#define ML_DRIFT_COMMON_TASK_TENSOR_REORDER_CPU_H_

#include <cstddef>

#include "absl/types/span.h"
#include "ml_drift/common/types.h"

// Single-threaded CPU routines for reordering tensor memory: N-D transpose,
// N-D constant padding, and f32 -> f16 conversion. These are the host-side
// building blocks used to repack weights before upload to the GPU.
//
// All routines operate on raw bytes; `element_size` selects the element width
// and must be 1, 2, 4 or 8. Nothing here reads or writes outside the exact
// extents of `src` and `dst`.
//
// The vectorized kernels use Clang's vector extensions, so they are only
// enabled when building with Clang. Other compilers (e.g. MSVC) get the scalar
// fallbacks, which produce identical output but are slower.
namespace ml_drift {

// Permutes an N-D row-major tensor.
//
// `shape` is the source shape. `perm[i]` is the index of the source dimension
// that becomes destination dimension `i`, i.e. dst.shape[i] = shape[perm[i]].
// This is the same convention as the `output_perm` parameter of XNNPACK's
// `xnn_run_transpose_nd_*`, which this replaces. `src` and `dst` must not
// overlap. `perm` must be a permutation of [0, shape.size()). A shape with a
// zero-sized dim is an empty tensor and is a no-op.
void TransposeCpu(const void* src, void* dst, size_t element_size,
                  absl::Span<const size_t> shape,
                  absl::Span<const size_t> perm);

// Pads an N-D row-major tensor with a constant.
//
// dst.shape[i] = pre_padding[i] + shape[i] + post_padding[i]. `pad_value`
// points to one element of `element_size` bytes. `src` and `dst` must not
// overlap. Same semantics as XNNPACK's `xnn_run_constant_pad_nd_*`, which
// this replaces.
void ConstantPadCpu(const void* src, void* dst, size_t element_size,
                    absl::Span<const size_t> shape,
                    absl::Span<const size_t> pre_padding,
                    absl::Span<const size_t> post_padding,
                    const void* pad_value);

// Converts `count` floats to half precision (round-to-nearest-even).
void ConvertF32ToF16Cpu(const float* src, half* dst, size_t count);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_TENSOR_REORDER_CPU_H_
