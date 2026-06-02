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

#include "ml_drift/common/task/weights_conversion.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

// Forward declaration to avoid full dependency on pthreadpool
struct pthreadpool;
typedef struct pthreadpool* pthreadpool_t;

extern "C" {
enum xnn_status xnn_run_convert_nc_f32_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  void* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_constant_pad_nd_x8(
    uint32_t flags, size_t num_dims, const size_t* input_shape,
    const size_t* pre_paddings, const size_t* post_paddings, const void* input,
    void* output, const void* padding_value, pthreadpool_t threadpool);

enum xnn_status xnn_run_constant_pad_nd_x16(
    uint32_t flags, size_t num_dims, const size_t* input_shape,
    const size_t* pre_paddings, const size_t* post_paddings, const void* input,
    void* output, const void* padding_value, pthreadpool_t threadpool);

enum xnn_status xnn_run_constant_pad_nd_x32(
    uint32_t flags, size_t num_dims, const size_t* input_shape,
    const size_t* pre_paddings, const size_t* post_paddings, const void* input,
    void* output, const void* padding_value, pthreadpool_t threadpool);

enum xnn_status xnn_run_transpose_nd_x8(
    const void* input, void* output, size_t num_dims, const size_t* input_shape,
    const size_t* output_perm, uint32_t flags, pthreadpool_t threadpool);

enum xnn_status xnn_run_transpose_nd_x16(
    const void* input, void* output, size_t num_dims, const size_t* input_shape,
    const size_t* output_perm, uint32_t flags, pthreadpool_t threadpool);

enum xnn_status xnn_run_transpose_nd_x32(
    const void* input, void* output, size_t num_dims, const size_t* input_shape,
    const size_t* output_perm, uint32_t flags, pthreadpool_t threadpool);

enum xnn_status xnn_run_transpose_nd_x64(
    const void* input, void* output, size_t num_dims, const size_t* input_shape,
    const size_t* output_perm, uint32_t flags, pthreadpool_t threadpool);

}  // extern "C"

namespace ml_drift {
namespace {
std::unique_ptr<half[]> ConvertF32F16(const float* src, size_t size) {
  std::unique_ptr<half[]> dst(new half[size + XNN_EXTRA_BYTES / sizeof(half)]);
  xnn_run_convert_nc_f32_f16(
      1, 1, 1, size, src, dst.get(),
      XNN_FLAG_DONT_SPIN_WORKERS, nullptr);
  return dst;
}

template <typename T>
void Reshape(const T* src, const OHWI& src_shape, int dst_group_size,
             absl::Span<const size_t> reshape_order, T pad_value, T* dst) {
  ABSL_CHECK_EQ(reshape_order.size(), 6u);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);

  const size_t dst_groups = DivideRoundUp(src_shape.o, dst_group_size * 4);
  const size_t src_groups = DivideRoundUp(src_shape.i, 4);

  const void* src_ptr = nullptr;
  const size_t padded_dst = dst_groups * dst_group_size * 4;
  const size_t padded_src = src_groups * 4;

  std::unique_ptr<T[]> padded_buffer;
  if (static_cast<size_t>(src_shape.o) == padded_dst &&
      static_cast<size_t>(src_shape.i) == padded_src) {
    // No padding needed.
    src_ptr = src;
  } else {
    // Padding the src buffer.
    padded_buffer.reset(
        new T[padded_dst * padded_src * src_shape.h * src_shape.w +
              XNN_EXTRA_BYTES / sizeof(T)]);
    const size_t prepadding[] = {0, 0, 0, 0};
    const size_t post_padding[] = {padded_dst - src_shape.o, 0, 0,
                                   padded_src - src_shape.i};
    const size_t original_shape[] = {
        static_cast<size_t>(src_shape.o), static_cast<size_t>(src_shape.h),
        static_cast<size_t>(src_shape.w), static_cast<size_t>(src_shape.i)};
    if constexpr (std::is_same_v<T, float>) {
      xnn_run_constant_pad_nd_x32(
          /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*num_dims=*/4, original_shape,
          prepadding, post_padding, src, padded_buffer.get(), &pad_value,
          /*threadpool=*/nullptr);
    } else if constexpr (std::is_same_v<T, half>) {
      xnn_run_constant_pad_nd_x16(
          /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*num_dims=*/4, original_shape,
          prepadding, post_padding, src, padded_buffer.get(), &pad_value,
          /*threadpool=*/nullptr);
    } else {
      xnn_run_constant_pad_nd_x8(
          /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*num_dims=*/4, original_shape,
          prepadding, post_padding, src, padded_buffer.get(), &pad_value,
          /*threadpool=*/nullptr);
    }
    src_ptr = padded_buffer.get();
  }
  // clang-format off
  const size_t dimensions[6] = {
      dst_groups, static_cast<size_t>(dst_group_size), 4,
      static_cast<size_t>(src_shape.h * src_shape.w),
      src_groups, 4
  };
  // clang-format on
  if constexpr (std::is_same_v<T, float>) {
    xnn_run_transpose_nd_x32(
        src_ptr, dst, /*num_dims=*/6, dimensions, reshape_order.data(),
        /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*threadpool=*/nullptr);
  } else if constexpr (std::is_same_v<T, half>) {
    xnn_run_transpose_nd_x16(
        src_ptr, dst, /*num_dims=*/6, dimensions, reshape_order.data(),
        /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*threadpool=*/nullptr);
  } else {
    xnn_run_transpose_nd_x8(
        src_ptr, dst, /*num_dims=*/6, dimensions, reshape_order.data(),
        /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*threadpool=*/nullptr);
  }
}

template <typename T>
void Reshape(const T* src, const OHWDI& src_shape, int dst_group_size,
             absl::Span<const size_t> reshape_order, T pad_value, T* dst) {
  ABSL_CHECK_EQ(reshape_order.size(), 6u);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);

  // We pass in OHWDI, but we need ODHWI...
  std::unique_ptr<T[]> dwh_buffer(
      new T[src_shape.o * src_shape.h * src_shape.w * src_shape.d *
                src_shape.i +
            XNN_EXTRA_BYTES / sizeof(T)]);
  const size_t original_shape[] = {
      static_cast<size_t>(src_shape.o), static_cast<size_t>(src_shape.h),
      static_cast<size_t>(src_shape.w), static_cast<size_t>(src_shape.d),
      static_cast<size_t>(src_shape.i)};
  const size_t dwh_perm[] = {0, 3, 1, 2, 4};
  if constexpr (std::is_same_v<T, float>) {
    xnn_run_transpose_nd_x32(src, dwh_buffer.get(), 5, original_shape, dwh_perm,
                             /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS,
                             /*threadpool=*/nullptr);
  } else if constexpr (std::is_same_v<T, half>) {
    xnn_run_transpose_nd_x16(src, dwh_buffer.get(), 5, original_shape, dwh_perm,
                             /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS,
                             /*threadpool=*/nullptr);
  } else {
    xnn_run_transpose_nd_x8(src, dwh_buffer.get(), 5, original_shape, dwh_perm,
                            /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS,
                            /*threadpool=*/nullptr);
  }

  const size_t dst_groups = DivideRoundUp(src_shape.o, dst_group_size * 4);
  const size_t src_groups = DivideRoundUp(src_shape.i, 4);

  const void* src_ptr = nullptr;
  const size_t padded_dst = dst_groups * dst_group_size * 4;
  const size_t padded_src = src_groups * 4;

  std::unique_ptr<T[]> padded_buffer;
  if (src_shape.o == padded_dst &&
      static_cast<size_t>(src_shape.i) == padded_src) {
    // No padding needed.
    src_ptr = dwh_buffer.get();
  } else {
    // Padding the src buffer.
    padded_buffer.reset(new T[padded_dst * padded_src * src_shape.h *
                                  src_shape.w * src_shape.d +
                              XNN_EXTRA_BYTES / sizeof(T)]);
    const size_t prepadding[] = {0, 0, 0, 0, 0};
    const size_t post_padding[] = {padded_dst - src_shape.o, 0, 0, 0,
                                   padded_src - src_shape.i};
    if constexpr (std::is_same_v<T, float>) {
      xnn_run_constant_pad_nd_x32(
          /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*num_dims=*/5, original_shape,
          prepadding, post_padding, dwh_buffer.get(), padded_buffer.get(),
          &pad_value, /*threadpool=*/nullptr);
    } else if constexpr (std::is_same_v<T, half>) {
      xnn_run_constant_pad_nd_x16(
          /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*num_dims=*/5, original_shape,
          prepadding, post_padding, dwh_buffer.get(), padded_buffer.get(),
          &pad_value, /*threadpool=*/nullptr);
    } else {
      xnn_run_constant_pad_nd_x8(
          /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*num_dims=*/5, original_shape,
          prepadding, post_padding, dwh_buffer.get(), padded_buffer.get(),
          &pad_value, /*threadpool=*/nullptr);
    }
    src_ptr = padded_buffer.get();
  }
  // clang-format off
  const size_t dimensions[6] = {
      dst_groups, static_cast<size_t>(dst_group_size), 4,
      static_cast<size_t>(src_shape.h * src_shape.w * src_shape.d),
      src_groups, 4
  };
  // clang-format on
  if constexpr (std::is_same_v<T, float>) {
    xnn_run_transpose_nd_x32(
        src_ptr, dst, /*num_dims=*/6, dimensions, reshape_order.data(),
        /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*threadpool=*/nullptr);
  } else if constexpr (std::is_same_v<T, half>) {
    xnn_run_transpose_nd_x16(
        src_ptr, dst, /*num_dims=*/6, dimensions, reshape_order.data(),
        /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*threadpool=*/nullptr);
  } else {
    xnn_run_transpose_nd_x8(
        src_ptr, dst, /*num_dims=*/6, dimensions, reshape_order.data(),
        /*flags=*/XNN_FLAG_DONT_SPIN_WORKERS, /*threadpool=*/nullptr);
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOHWIOGroupI4O4(const Tensor<OHWI, S>& weights,
                                      int out_group_size, absl::Span<T> dst,
                                      T pad_value) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);
  const int df_index = weights.shape.h * weights.shape.w * weights.shape.i;

  T* ptr = dst.data();
  const auto* src = weights.Data();
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d_group = 0; d_group < out_group_size; ++d_group) {
            for (int s_ch = 4 * s; s_ch < 4 * s + 4; ++s_ch) {
              int d_ch = 4 * (d * out_group_size + d_group);
              int f_index =
                  ((d_ch * weights.shape.h + y) * weights.shape.w + x) *
                      weights.shape.i +
                  s_ch;
              *ptr++ = s_ch < weights.shape.i && d_ch < weights.shape.o
                           ? src[f_index]
                           : pad_value;
              ++d_ch;
              f_index += df_index;
              *ptr++ = s_ch < weights.shape.i && d_ch < weights.shape.o
                           ? src[f_index]
                           : pad_value;
              ++d_ch;
              f_index += df_index;
              *ptr++ = s_ch < weights.shape.i && d_ch < weights.shape.o
                           ? src[f_index]
                           : pad_value;
              ++d_ch;
              f_index += df_index;
              *ptr++ = s_ch < weights.shape.i && d_ch < weights.shape.o
                           ? src[f_index]
                           : pad_value;
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOICustomSpatialI4O4(
    const Tensor<OHWI, S>& weights, const std::vector<int>& spatial_remap,
    absl::Span<T> dst, T pad_value) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  const auto* src = weights.Data();
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int y = 0; y < weights.shape.h; ++y) {
        for (int x = 0; x < weights.shape.w; ++x) {
          const int kernel_index = spatial_remap[y * weights.shape.w + x];
          const int kernel_index_x = kernel_index % weights.shape.w;
          const int kernel_index_y = kernel_index / weights.shape.w;
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              const int s_ch = s * 4 + i;
              const int d_ch = d * 4 + j;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index = weights.shape.LinearIndex(
                    {d_ch, kernel_index_y, kernel_index_x, s_ch});
                dst[counter++] = src[f_index];
              } else {
                dst[counter++] = pad_value;
              }
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOICustomSpatialI4O4(
    const Tensor<OHWDI, S>& weights, const std::vector<int>& spatial_remap,
    absl::Span<T> dst, T pad_value) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  const auto* src = weights.Data();
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int z = 0; z < weights.shape.d; ++z) {
        for (int y = 0; y < weights.shape.h; ++y) {
          for (int x = 0; x < weights.shape.w; ++x) {
            int kernel_index =
                spatial_remap[(z * weights.shape.h + y) * weights.shape.w + x];
            const int kernel_index_x = kernel_index % weights.shape.w;
            kernel_index /= weights.shape.w;
            const int kernel_index_y = kernel_index % weights.shape.h;
            const int kernel_index_z = kernel_index / weights.shape.h;
            for (int i = 0; i < 4; ++i) {
              for (int j = 0; j < 4; ++j) {
                const int s_ch = s * 4 + i;
                const int d_ch = d * 4 + j;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                  const int f_index = weights.shape.LinearIndex(
                      {d_ch, kernel_index_y, kernel_index_x, kernel_index_z,
                       s_ch});
                  dst[counter++] = src[f_index];
                } else {
                  dst[counter++] = pad_value;
                }
              }
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOICustomSpatialO4I4(
    const Tensor<OHWI, S>& weights, const std::vector<int>& spatial_remap,
    absl::Span<T> dst, T pad_value) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  const auto* src = weights.Data();
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int y = 0; y < weights.shape.h; ++y) {
        for (int x = 0; x < weights.shape.w; ++x) {
          const int kernel_index = spatial_remap[y * weights.shape.w + x];
          const int kernel_index_x = kernel_index % weights.shape.w;
          const int kernel_index_y = kernel_index / weights.shape.w;
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index = weights.shape.LinearIndex(
                    {d_ch, kernel_index_y, kernel_index_x, s_ch});
                dst[counter++] = src[f_index];
              } else {
                dst[counter++] = pad_value;
              }
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOICustomSpatialO4I4(
    const Tensor<OHWDI, S>& weights, const std::vector<int>& spatial_remap,
    absl::Span<T> dst, T pad_value) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  const auto* src = weights.Data();
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int z = 0; z < weights.shape.d; ++z) {
        for (int y = 0; y < weights.shape.h; ++y) {
          for (int x = 0; x < weights.shape.w; ++x) {
            int kernel_index =
                spatial_remap[(z * weights.shape.h + y) * weights.shape.w + x];
            const int kernel_index_x = kernel_index % weights.shape.w;
            kernel_index /= weights.shape.w;
            const int kernel_index_y = kernel_index % weights.shape.h;
            const int kernel_index_z = kernel_index / weights.shape.h;
            for (int i = 0; i < 4; ++i) {
              for (int j = 0; j < 4; ++j) {
                const int s_ch = s * 4 + j;
                const int d_ch = d * 4 + i;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                  const int f_index = weights.shape.LinearIndex(
                      {d_ch, kernel_index_y, kernel_index_x, kernel_index_z,
                       s_ch});
                  dst[counter++] = src[f_index];
                } else {
                  dst[counter++] = pad_value;
                }
              }
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToISpatialOI4O4UnalignedIO(const Tensor<OHWI, S>& weights,
                                                absl::Span<T> dst) {
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  int counter = 0;
  for (int src_s = 0; src_s < src_slices; ++src_s) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
          const int src_ch_count = std::min(4, weights.shape.i - src_s * 4);
          const int dst_ch_count = std::min(4, weights.shape.o - dst_s * 4);
          for (int j = 0; j < src_ch_count; ++j) {
            for (int i = 0; i < dst_ch_count; ++i) {
              const int s_ch = src_s * 4 + j;
              const int d_ch = dst_s * 4 + i;
              const int f_index = weights.shape.LinearIndex({d_ch, y, x, s_ch});
              dst[counter++] = weights.data[f_index];
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToCustom(const Tensor<OHWI, S>& weights,
                              std::vector<std::pair<Axis, int>> group_sizes,
                              const std::vector<int>& spatial_remap,
                              absl::Span<T> dst, T pad_value) {
  std::vector<std::pair<Axis, int>> coords;
  coords.reserve(group_sizes.size());
  OHWI total_size = OHWI(1, 1, 1, 1);
  for (auto& [axis, size] : group_sizes) {
    coords.push_back({axis, 0});
    const int prev_size = total_size.get(axis);
    if (size <= 0) {
      size = DivideRoundUp(weights.shape.get(axis), prev_size);
    }
    total_size.set(axis, prev_size * size);
  }
  for (int i = 0; i < total_size.DimensionsProduct(); ++i) {
    OHWI src_coords = OHWI(0, 0, 0, 0);
    OHWI prev_sizes = OHWI(1, 1, 1, 1);
    for (size_t group = 0; group < group_sizes.size(); ++group) {
      const auto& [axis, group_size] = group_sizes[group];
      if (coords[group].second == group_size) {
        coords[group].second = 0;
        ++coords[group + 1].second;
      }
      const int coord_id = coords[group].second;
      const int prev_value = src_coords.get(axis);
      const int prev_size = prev_sizes.get(axis);
      src_coords.set(axis, coord_id * prev_size + prev_value);
      prev_sizes.set(axis, group_size * prev_size);
    }
    if (!spatial_remap.empty()) {
      const int spatial_linear = src_coords.h * weights.shape.w + src_coords.w;
      int new_spatial_linear = spatial_remap[spatial_linear];
      src_coords.w = new_spatial_linear % weights.shape.w;
      src_coords.h = new_spatial_linear / weights.shape.w;
    }
    if (src_coords.o < weights.shape.o && src_coords.h < weights.shape.h &&
        src_coords.w < weights.shape.w && src_coords.i < weights.shape.i) {
      dst[i] = weights.data[weights.shape.LinearIndex(
          {src_coords.o, src_coords.h, src_coords.w, src_coords.i})];
    } else {
      dst[i] = pad_value;
    }
    ++coords[0].second;
  }
}

template <DataType S, typename T>
void RearrangeWeightsToCustom(const Tensor<OHWDI, S>& weights,
                              std::vector<std::pair<Axis, int>> group_sizes,
                              const std::vector<int>& spatial_remap,
                              absl::Span<T> dst, T pad_value) {
  std::vector<std::pair<Axis, int>> coords;
  coords.reserve(group_sizes.size());
  OHWDI total_size = OHWDI(1, 1, 1, 1, 1);
  for (auto& [axis, size] : group_sizes) {
    coords.push_back({axis, 0});
    const int prev_size = total_size.get(axis);
    if (size <= 0) {
      size = DivideRoundUp(weights.shape.get(axis), prev_size);
    }
    total_size.set(axis, prev_size * size);
  }
  for (int i = 0; i < total_size.DimensionsProduct(); ++i) {
    OHWDI src_coords = OHWDI(0, 0, 0, 0, 0);
    OHWDI prev_sizes = OHWDI(1, 1, 1, 1, 1);
    for (size_t group = 0; group < group_sizes.size(); ++group) {
      const auto& [axis, group_size] = group_sizes[group];
      if (coords[group].second == group_size) {
        coords[group].second = 0;
        ++coords[group + 1].second;
      }
      const int coord_id = coords[group].second;
      const int prev_value = src_coords.get(axis);
      const int prev_size = prev_sizes.get(axis);
      src_coords.set(axis, coord_id * prev_size + prev_value);
      prev_sizes.set(axis, group_size * prev_size);
    }
    if (!spatial_remap.empty()) {
      const int spatial_linear =
          (src_coords.d * weights.shape.h + src_coords.h) * weights.shape.w +
          src_coords.w;
      int new_spatial_linear = spatial_remap[spatial_linear];
      src_coords.w = new_spatial_linear % weights.shape.w;
      new_spatial_linear /= weights.shape.w;
      src_coords.h = new_spatial_linear % weights.shape.h;
      src_coords.d = new_spatial_linear / weights.shape.h;
    }
    if (src_coords.o < weights.shape.o && src_coords.h < weights.shape.h &&
        src_coords.w < weights.shape.w && src_coords.d < weights.shape.d &&
        src_coords.i < weights.shape.i) {
      dst[i] = weights.data[weights.shape.LinearIndex(
          {src_coords.o, src_coords.h, src_coords.w, src_coords.d,
           src_coords.i})];
    } else {
      dst[i] = pad_value;
    }
    ++coords[0].second;
  }
}

// Parses the 4 msb/lsb of value, and returns it in the 4 lsb.
inline uint8_t GetUint4FromUint8(uint8_t value, bool msb) {
  return msb ? (value >> 4) : (value & 0xF);
}

// Note that this assumes we shift by 8u
void Uint4PackedkOSpatialIOGroupI4O4(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims) {
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, dst_group_size);
  const int df_index =
      swap_dims ? 1 : weights.shape.h * weights.shape.w * weights.shape.i;

  for (int i = 0; i < weights.shape.o; ++i) {
    weights_sum_i[i] = 0;
  }

  const uint8_t double_pad = pad_value | pad_value << 4;
  pad_value ^= 0x8;  // Shift pad_value by 8u to cancel out when val is shifted
  uint8_t* ptr = dst.data();
  const uint8_t* src = weights.Data();
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d_group = 0; d_group < dst_group_size; ++d_group) {
            const int d_ch = 4 * (d * dst_group_size + d_group);
            const int d_ch_diff = weights.shape.o - d_ch;
            for (int s_ch = 4 * s; s_ch < 4 * s + 4; ++s_ch) {
              if (s_ch >= weights.shape.i) {
                *ptr++ = double_pad;
                *ptr++ = double_pad;
                continue;
              }
              int f_index =
                  swap_dims
                      ? ((s_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.o +
                            d_ch
                      : ((d_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.i +
                            s_ch;

              const uint8_t val0 =
                  d_ch_diff > 0
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              f_index += df_index;
              const uint8_t val1 =
                  d_ch_diff > 1
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              f_index += df_index;
              const uint8_t val2 =
                  d_ch_diff > 2
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              f_index += df_index;
              const uint8_t val3 =
                  d_ch_diff > 3
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              *ptr++ = ((val1 << 4) | val0) ^ 0x88;  // Shift val0 & val1 by 8u
              *ptr++ = ((val3 << 4) | val2) ^ 0x88;

              if (d_ch < weights_sum_i.size())
                weights_sum_i[d_ch + 0] += (val0 ^ 0x8) - 8;
              if (d_ch + 1 < weights_sum_i.size())
                weights_sum_i[d_ch + 1] += (val1 ^ 0x8) - 8;
              if (d_ch + 2 < weights_sum_i.size())
                weights_sum_i[d_ch + 2] += (val2 ^ 0x8) - 8;
              if (d_ch + 3 < weights_sum_i.size())
                weights_sum_i[d_ch + 3] += (val3 ^ 0x8) - 8;
            }
          }
        }
      }
    }
  }
}

// Note that this assumes we shift by 8u
void Uint4Packedk2DYIsSpatialIOAndXIsOGroupI4O4(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims) {
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, dst_group_size);
  const int df_index =
      swap_dims ? 1 : weights.shape.h * weights.shape.w * weights.shape.i;

  for (int i = 0; i < weights.shape.o; ++i) {
    weights_sum_i[i] = 0;
  }

  const uint8_t double_pad = pad_value | pad_value << 4;
  pad_value ^= 0x8;  // Shift pad_value by 8u to cancel out when val is shifted
  uint8_t* ptr = dst.data();
  const uint8_t* src = weights.Data();
  for (int y = 0; y < weights.shape.h; ++y) {
    for (int x = 0; x < weights.shape.w; ++x) {
      for (int s = 0; s < src_slices; ++s) {
        for (int d = 0; d < dst_groups; ++d) {
          for (int d_group = 0; d_group < dst_group_size; ++d_group) {
            const int d_ch = 4 * (d * dst_group_size + d_group);
            const int d_ch_diff = weights.shape.o - d_ch;
            for (int s_ch = 4 * s; s_ch < 4 * s + 4; ++s_ch) {
              if (s_ch >= weights.shape.i) {
                *ptr++ = double_pad;
                *ptr++ = double_pad;
                continue;
              }
              int f_index =
                  swap_dims
                      ? ((s_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.o +
                            d_ch
                      : ((d_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.i +
                            s_ch;

              const uint8_t val0 =
                  d_ch_diff > 0
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              f_index += df_index;
              const uint8_t val1 =
                  d_ch_diff > 1
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              f_index += df_index;
              const uint8_t val2 =
                  d_ch_diff > 2
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              f_index += df_index;
              const uint8_t val3 =
                  d_ch_diff > 3
                      ? GetUint4FromUint8(src[f_index >> 1], f_index & 1)
                      : pad_value;
              *ptr++ = ((val1 << 4) | val0) ^ 0x88;  // Shift val0 & val1 by 8u
              *ptr++ = ((val3 << 4) | val2) ^ 0x88;

              if (d_ch < weights_sum_i.size())
                weights_sum_i[d_ch + 0] += (val0 ^ 0x8) - 8;
              if (d_ch + 1 < weights_sum_i.size())
                weights_sum_i[d_ch + 1] += (val1 ^ 0x8) - 8;
              if (d_ch + 2 < weights_sum_i.size())
                weights_sum_i[d_ch + 2] += (val2 ^ 0x8) - 8;
              if (d_ch + 3 < weights_sum_i.size())
                weights_sum_i[d_ch + 3] += (val3 ^ 0x8) - 8;
            }
          }
        }
      }
    }
  }
}

// Parses the 2 msb/lsb of value, and returns it in the 2 lsb.
inline uint8_t GetUint2FromUint8(uint8_t value, int bit_offset) {
  return (value >> bit_offset) & 0x3;
}

// Note that this assumes we shift by 2u
void Uint2Packedk2DYIsSpatialIOAndXIsOGroupI4O4(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims) {
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, dst_group_size);
  const int df_index =
      swap_dims ? 1 : weights.shape.h * weights.shape.w * weights.shape.i;

  for (int i = 0; i < weights.shape.o; ++i) {
    weights_sum_i[i] = 0;
  }

  const uint8_t quad_pad =
      pad_value | pad_value << 2 | pad_value << 4 | pad_value << 6;
  pad_value ^= 0x2;  // Shift pad_value by 2u to cancel out when val is shifted
  uint8_t* ptr = dst.data();
  const uint8_t* src = weights.Data();
  for (int y = 0; y < weights.shape.h; ++y) {
    for (int x = 0; x < weights.shape.w; ++x) {
      for (int s = 0; s < src_slices; ++s) {
        for (int d = 0; d < dst_groups; ++d) {
          for (int d_group = 0; d_group < dst_group_size; ++d_group) {
            const int d_ch = 4 * (d * dst_group_size + d_group);
            const int d_ch_diff = weights.shape.o - d_ch;
            for (int s_ch = 4 * s; s_ch < 4 * s + 4; ++s_ch) {
              if (s_ch >= weights.shape.i) {
                *ptr++ = quad_pad;
                continue;
              }
              int f_index =
                  swap_dims
                      ? ((s_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.o +
                            d_ch
                      : ((d_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.i +
                            s_ch;

              const uint8_t val0 =
                  d_ch_diff > 0
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              f_index += df_index;
              const uint8_t val1 =
                  d_ch_diff > 1
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              f_index += df_index;
              const uint8_t val2 =
                  d_ch_diff > 2
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              f_index += df_index;
              const uint8_t val3 =
                  d_ch_diff > 3
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              *ptr++ = ((val3 << 6) | (val2 << 4) | (val1 << 2) | val0) ^
                       0xAA;  // Shift val0 & val1 by 2u

              if (d_ch < weights_sum_i.size())
                weights_sum_i[d_ch + 0] += (val0 ^ 0x2) - 2;
              if (d_ch + 1 < weights_sum_i.size())
                weights_sum_i[d_ch + 1] += (val1 ^ 0x2) - 2;
              if (d_ch + 2 < weights_sum_i.size())
                weights_sum_i[d_ch + 2] += (val2 ^ 0x2) - 2;
              if (d_ch + 3 < weights_sum_i.size())
                weights_sum_i[d_ch + 3] += (val3 ^ 0x2) - 2;
            }
          }
        }
      }
    }
  }
}

// Note that this assumes we shift by 2u
void Uint2PackedkOSpatialIOGroupI4O4(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims) {
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, dst_group_size);
  const int df_index =
      swap_dims ? 1 : weights.shape.h * weights.shape.w * weights.shape.i;

  for (int i = 0; i < weights.shape.o; ++i) {
    weights_sum_i[i] = 0;
  }

  const uint8_t quad_pad =
      pad_value | pad_value << 2 | pad_value << 4 | pad_value << 6;
  pad_value ^= 0x2;  // Shift pad_value by 2u to cancel out when val is shifted
  uint8_t* ptr = dst.data();
  const uint8_t* src = weights.Data();
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d_group = 0; d_group < dst_group_size; ++d_group) {
            const int d_ch = 4 * (d * dst_group_size + d_group);
            const int d_ch_diff = weights.shape.o - d_ch;
            for (int s_ch = 4 * s; s_ch < 4 * s + 4; ++s_ch) {
              if (s_ch >= weights.shape.i) {
                *ptr++ = quad_pad;
                continue;
              }
              int f_index =
                  swap_dims
                      ? ((s_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.o +
                            d_ch
                      : ((d_ch * weights.shape.h + y) * weights.shape.w + x) *
                                weights.shape.i +
                            s_ch;

              const uint8_t val0 =
                  d_ch_diff > 0
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              f_index += df_index;
              const uint8_t val1 =
                  d_ch_diff > 1
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              f_index += df_index;
              const uint8_t val2 =
                  d_ch_diff > 2
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              f_index += df_index;
              const uint8_t val3 =
                  d_ch_diff > 3
                      ? GetUint2FromUint8(src[f_index >> 2], (f_index & 3) * 2)
                      : pad_value;
              *ptr++ = ((val3 << 6) | (val2 << 4) | (val1 << 2) | val0) ^
                       0xAA;  // Shift val0 & val1 by 2u

              if (d_ch < weights_sum_i.size())
                weights_sum_i[d_ch + 0] += (val0 ^ 0x2) - 2;
              if (d_ch + 1 < weights_sum_i.size())
                weights_sum_i[d_ch + 1] += (val1 ^ 0x2) - 2;
              if (d_ch + 2 < weights_sum_i.size())
                weights_sum_i[d_ch + 2] += (val2 ^ 0x2) - 2;
              if (d_ch + 3 < weights_sum_i.size())
                weights_sum_i[d_ch + 3] += (val3 ^ 0x2) - 2;
            }
          }
        }
      }
    }
  }
}
}  // namespace

std::unique_ptr<half[]> ConvertF32F16(const std::vector<float>& src) {
  return ConvertF32F16(src.data(), src.size());
}

#ifdef __aarch64__
void RearrangeWeightsToOHWIOGroupI4O4(
    const Tensor<OHWI, DataType::FLOAT32>& weights, int out_group_size,
    __fp16* dst) {
  const float* src = weights.data.data();
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);
  const int df_index = weights.shape.h * weights.shape.w * weights.shape.i;
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d_group = 0; d_group < out_group_size; ++d_group) {
            const int d_ch = 4 * (d * out_group_size + d_group);
            const int d_ch_diff = weights.shape.o - d_ch;
            for (int s_ch = 4 * s; s_ch < 4 * s + 4; ++s_ch) {
              const int f_index =
                  ((d_ch * weights.shape.h + y) * weights.shape.w + x) *
                      weights.shape.i +
                  s_ch;
              *dst++ =
                  s_ch < weights.shape.i && d_ch_diff > 0 ? src[f_index] : 0;
              *dst++ = s_ch < weights.shape.i && d_ch_diff > 1
                           ? src[f_index + df_index]
                           : 0;
              *dst++ = s_ch < weights.shape.i && d_ch_diff > 2
                           ? src[f_index + 2 * df_index]
                           : 0;
              *dst++ = s_ch < weights.shape.i && d_ch_diff > 3
                           ? src[f_index + 3 * df_index]
                           : 0;
            }
          }
        }
      }
    }
  }
}
#endif  // __aarch64__

unsigned int GetTotalElementsCountForLayout(
    const WeightsDescription& weight_desc, const OHWDI& shape) {
  switch (weight_desc.layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
    case WeightsLayout::kOSpatialIOGroupO4I4:
    case WeightsLayout::kOISpatialOGroupI4O4:
    case WeightsLayout::kOISpatialOGroupO4I4:
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4: {
      const int o_group_size = weight_desc.GetOutputGroupSize();
      size_t i_alignment = 4;
      size_t o_alignment = 4 * o_group_size;
      return AlignByN(shape.i, i_alignment) * AlignByN(shape.o, o_alignment) *
             shape.h * shape.w * shape.d;
    }
    case WeightsLayout::kOICustomSpatialI4O4:
    case WeightsLayout::kOICustomSpatialO4I4: {
      unsigned int i_aligned = AlignByN(shape.i, 4);
      unsigned int o_aligned = AlignByN(shape.o, 4);
      return i_aligned * o_aligned * weight_desc.spatial_remap.size();
    }
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4: {
      unsigned int i_slices = DivideRoundUp(shape.i, 4);
      unsigned int o_slices = DivideRoundUp(shape.o, 4);
      unsigned int o_aligned =
          AlignByN(o_slices, weight_desc.GetOutputGroupSize());
      return i_slices * o_aligned * 4 * 4 * shape.h * shape.w * shape.d;
    }
    case WeightsLayout::kISpatialOI4O4UnalignedIO: {
      return shape.DimensionsProduct();
    }
    case WeightsLayout::kCustomGroups: {
      OHWDI total_size = OHWDI(1, 1, 1, 1, 1);
      for (auto [axis, size] : weight_desc.group_sizes) {
        const int prev_size = total_size.get(axis);
        if (size <= 0) {
          size = DivideRoundUp(shape.get(axis), prev_size);
        }
        total_size.set(axis, prev_size * size);
      }
      return total_size.DimensionsProduct();
    }
    case WeightsLayout::kUnknown:
      return -1;
  }
  ABSL_LOG(FATAL) << "Unknown weights layout";
  return -1;
}

unsigned int GetTotalElementsCountForLayout(
    const WeightsDescription& weight_desc, const OHWI& shape) {
  const OHWDI ohwdi_shape = OHWDI(shape.o, shape.h, shape.w, 1, shape.i);
  return GetTotalElementsCountForLayout(weight_desc, ohwdi_shape);
}

uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWI& shape) {
  const OHWDI ohwdi_shape = OHWDI(shape.o, shape.h, shape.w, 1, shape.i);
  return Get2dResourceSize(weight_desc, ohwdi_shape);
}

uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWDI& shape) {
  const int dst_slices = DivideRoundUp(shape.o, 4);
  const int src_slices = DivideRoundUp(shape.i, 4);

  if (weight_desc.layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weight_desc.layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    const int dst_slices_aligned =
        AlignByN(dst_slices, weight_desc.GetOutputGroupSize());
    return uint2(dst_slices_aligned, src_slices * shape.h * shape.w * shape.d);
  } else if (weight_desc.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    const int dst_groups =
        DivideRoundUp(dst_slices, weight_desc.GetOutputGroupSize());
    return uint2(weight_desc.GetOutputGroupSize() * 4,
                 dst_groups * src_slices * shape.h * shape.w * shape.d);
  } else {
    return uint2(0, 0);
  }
}

// weights.data needs an extra XNN_EXTRA_BYTES/sizeof(float) bytes reserved
void RearrangeWeights(const Tensor<OHWI, DataType::FLOAT32>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst) {
  if (!weights.spanned_data.empty() &&
      weights.spanned_data.size() == weights.shape.DimensionsProduct()) {
    const_cast<Tensor<OHWI, DataType::FLOAT32>&>(weights)
        .MoveDataForCpuRearrange(weights.shape.DimensionsProduct() +
                                 XNN_EXTRA_BYTES / sizeof(float));
  }
  ABSL_CHECK_GE(weights.data.size(), weights.shape.DimensionsProduct() +
                                         XNN_EXTRA_BYTES / sizeof(float));
  const unsigned int flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  float* f32_ptr = reinterpret_cast<float*>(dst.data());
  half* f16_ptr = reinterpret_cast<half*>(dst.data());
  std::vector<size_t> reshape_order;
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  switch (dst_weight_desc.layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      // kOSpatialIOGroupI4O4 is the most common layout. As such, we have
      // some optimized paths. If we are using Aarch64 on F16, use intrinsics.
#ifdef __aarch64__
      if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToOHWIOGroupI4O4(weights, dst_group_size,
                                         reinterpret_cast<__fp16*>(dst.data()));
        return;
      }
#endif  // __aarch64__
      if (weights.shape.o % (dst_group_size * 4) != 0 ||
          weights.shape.i % 4 != 0) {  // Need to pad
        // If we need to use both XNNPACK transpose and pad, benchmarks have
        // shown it to be slightly slower than RearrangeWeightsToOHWIOGroupI4O4.
        // Since this is the most common layout, we keep a custom call.
        if (dst_weight_desc.type == DataType::FLOAT32) {
          RearrangeWeightsToOHWIOGroupI4O4(weights, dst_group_size,
                                           absl::MakeSpan(f32_ptr, flt_count),
                                           0.0f);
        } else if (dst_weight_desc.type == DataType::FLOAT16) {
          RearrangeWeightsToOHWIOGroupI4O4(weights, dst_group_size,
                                           absl::MakeSpan(f16_ptr, flt_count),
                                           half(0.0f));
        }
        return;
      }
      reshape_order = {0, 3, 4, 1, 5, 2};
      break;
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      reshape_order = {3, 4, 0, 1, 5, 2};
      break;
    case WeightsLayout::kOSpatialIOGroupO4I4:
      reshape_order = {0, 3, 4, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupO4I4:
      reshape_order = {0, 4, 3, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupI4O4:
      reshape_order = {0, 4, 3, 1, 5, 2};
      break;
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
      reshape_order = {5, 3, 4, 0, 1, 2};
      break;
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      reshape_order = {2, 3, 4, 0, 1, 5};
      break;
    case WeightsLayout::kOICustomSpatialI4O4: {
      if (dst_weight_desc.type == DataType::FLOAT32) {
        RearrangeWeightsToOICustomSpatialI4O4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f32_ptr, flt_count), 0.0f);
      } else if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToOICustomSpatialI4O4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f16_ptr, flt_count), half(0.0f));
      }
      return;
    }
    case WeightsLayout::kOICustomSpatialO4I4: {
      if (dst_weight_desc.type == DataType::FLOAT32) {
        RearrangeWeightsToOICustomSpatialO4I4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f32_ptr, flt_count), 0.0f);
      } else if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToOICustomSpatialO4I4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f16_ptr, flt_count), half(0.0f));
      }
      return;
    }
    case WeightsLayout::kISpatialOI4O4UnalignedIO: {
      if (dst_weight_desc.type == DataType::FLOAT32) {
        RearrangeWeightsToISpatialOI4O4UnalignedIO(
            weights, absl::MakeSpan(f32_ptr, flt_count));
      } else if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToISpatialOI4O4UnalignedIO(
            weights, absl::MakeSpan(f16_ptr, flt_count));
      }
      return;
    }
    case WeightsLayout::kCustomGroups:
      if (dst_weight_desc.type == DataType::FLOAT32) {
        RearrangeWeightsToCustom(weights, dst_weight_desc.group_sizes,
                                 dst_weight_desc.spatial_remap,
                                 absl::MakeSpan(f32_ptr, flt_count), 0.0f);
      } else if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToCustom(
            weights, dst_weight_desc.group_sizes, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f16_ptr, flt_count), half(0.0f));
      }
      return;
    case WeightsLayout::kUnknown:
      return;
  }
  if (dst_weight_desc.type == DataType::FLOAT32) {
    Reshape(weights.Data(), weights.shape, dst_group_size, reshape_order,
            /*pad_value=*/0.0f, f32_ptr);
  } else if (dst_weight_desc.type == DataType::FLOAT16) {
    Reshape(
        ConvertF32F16(weights.Data(), weights.shape.DimensionsProduct()).get(),
        weights.shape, dst_group_size, reshape_order,
        /*pad_value=*/half(0.0f), f16_ptr);
  }
}

// weights.data needs an extra XNN_EXTRA_BYTES/sizeof(float) bytes reserved
void RearrangeWeights(const Tensor<OHWDI, DataType::FLOAT32>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst) {
  if (!weights.spanned_data.empty() &&
      weights.spanned_data.size() == weights.shape.DimensionsProduct()) {
    const_cast<Tensor<OHWDI, DataType::FLOAT32>&>(weights)
        .MoveDataForCpuRearrange(weights.shape.DimensionsProduct() +
                                 XNN_EXTRA_BYTES / sizeof(float));
  }
  ABSL_CHECK_GE(weights.data.size(), weights.shape.DimensionsProduct() +
                                          XNN_EXTRA_BYTES / sizeof(float));
  const unsigned int flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  float* f32_ptr = reinterpret_cast<float*>(dst.data());
  half* f16_ptr = reinterpret_cast<half*>(dst.data());
  std::vector<size_t> reshape_order;
  switch (dst_weight_desc.layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      reshape_order = {0, 3, 4, 1, 5, 2};
      break;
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      reshape_order = {3, 4, 0, 1, 5, 2};
      break;
    case WeightsLayout::kOSpatialIOGroupO4I4:
      reshape_order = {0, 3, 4, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupO4I4:
      reshape_order = {0, 4, 3, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupI4O4:
      reshape_order = {0, 4, 3, 1, 5, 2};
      break;
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
      reshape_order = {5, 3, 4, 0, 1, 2};
      break;
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      reshape_order = {2, 3, 4, 0, 1, 5};
      break;
    case WeightsLayout::kOICustomSpatialI4O4: {
      if (dst_weight_desc.type == DataType::FLOAT32) {
        RearrangeWeightsToOICustomSpatialI4O4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f32_ptr, flt_count), 0.0f);
      } else if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToOICustomSpatialI4O4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f16_ptr, flt_count), half(0.0f));
      }
      return;
    }
    case WeightsLayout::kOICustomSpatialO4I4: {
      if (dst_weight_desc.type == DataType::FLOAT32) {
        RearrangeWeightsToOICustomSpatialO4I4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f32_ptr, flt_count), 0.0f);
      } else if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToOICustomSpatialO4I4(
            weights, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f16_ptr, flt_count), half(0.0f));
      }
      return;
    }
    case WeightsLayout::kCustomGroups:
      if (dst_weight_desc.type == DataType::FLOAT32) {
        RearrangeWeightsToCustom(weights, dst_weight_desc.group_sizes,
                                 dst_weight_desc.spatial_remap,
                                 absl::MakeSpan(f32_ptr, flt_count), 0.0f);
      } else if (dst_weight_desc.type == DataType::FLOAT16) {
        RearrangeWeightsToCustom(
            weights, dst_weight_desc.group_sizes, dst_weight_desc.spatial_remap,
            absl::MakeSpan(f16_ptr, flt_count), half(0.0f));
      }
      return;
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      ABSL_CHECK(false) << "Not implemented";
      return;
    case WeightsLayout::kUnknown:
      return;
  }
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  if (dst_weight_desc.type == DataType::FLOAT32) {
    Reshape(weights.Data(), weights.shape, dst_group_size, reshape_order,
            /*pad_value=*/0.0f, f32_ptr);
  } else if (dst_weight_desc.type == DataType::FLOAT16) {
    Reshape(
        ConvertF32F16(weights.Data(), weights.shape.DimensionsProduct()).get(),
        weights.shape, dst_group_size, reshape_order,
        /*pad_value=*/half(0.0f), f16_ptr);
  }
}

// weights.data needs an extra XNN_EXTRA_BYTES/sizeof(int8_t) bytes reserved
void RearrangeWeights(const Tensor<OHWI, DataType::INT8>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst) {
  if (!weights.spanned_data.empty() &&
      weights.spanned_data.size() == weights.shape.DimensionsProduct()) {
    const_cast<Tensor<OHWI, DataType::INT8>&>(weights).MoveDataForCpuRearrange(
        weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(int8_t));
  }
  ABSL_CHECK_GE(weights.data.size(), weights.shape.DimensionsProduct() +
                                          XNN_EXTRA_BYTES / sizeof(int8_t));
  const unsigned int elements_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  int8_t* i8_ptr = reinterpret_cast<int8_t*>(dst.data());
  auto dst_span = absl::MakeSpan(i8_ptr, elements_count);
  std::vector<size_t> reshape_order;
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  switch (dst_weight_desc.layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      if (weights.shape.o % (dst_group_size * 4) != 0 ||
          weights.shape.i % 4 != 0) {  // Need to pad
        RearrangeWeightsToOHWIOGroupI4O4(weights, dst_group_size, dst_span,
                                         static_cast<int8_t>(0));
        return;
      }
      reshape_order = {0, 3, 4, 1, 5, 2};
      break;
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      reshape_order = {3, 4, 0, 1, 5, 2};
      break;
    case WeightsLayout::kOSpatialIOGroupO4I4:
      reshape_order = {0, 3, 4, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupO4I4:
      reshape_order = {0, 4, 3, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupI4O4:
      reshape_order = {0, 4, 3, 1, 5, 2};
      break;
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
      reshape_order = {5, 3, 4, 0, 1, 2};
      break;
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      reshape_order = {2, 3, 4, 0, 1, 5};
      break;
    case WeightsLayout::kOICustomSpatialI4O4: {
      RearrangeWeightsToOICustomSpatialI4O4(weights,
                                            dst_weight_desc.spatial_remap,
                                            dst_span, static_cast<int8_t>(0));
      return;
    }
    case WeightsLayout::kOICustomSpatialO4I4: {
      RearrangeWeightsToOICustomSpatialO4I4(weights,
                                            dst_weight_desc.spatial_remap,
                                            dst_span, static_cast<int8_t>(0));
      return;
    }
    case WeightsLayout::kCustomGroups: {
      RearrangeWeightsToCustom(weights, dst_weight_desc.group_sizes,
                               dst_weight_desc.spatial_remap, dst_span,
                               static_cast<int8_t>(0));
      return;
    }
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      ABSL_CHECK(false) << "Not implemented";
      return;
    case WeightsLayout::kUnknown:
      return;
  }
  Reshape(weights.Data(), weights.shape, dst_group_size, reshape_order,
          /*pad_value=*/static_cast<int8_t>(0), i8_ptr);
}

// weights.data needs an extra XNN_EXTRA_BYTES/sizeof(uint8_t) bytes reserved
void RearrangeWeights(const Tensor<OHWI, DataType::UINT8>& weights,
                      const WeightsDescription& dst_weight_desc,
                      absl::Span<uint8_t> dst, uint8_t pad_value) {
  if (!weights.spanned_data.empty() &&
      weights.spanned_data.size() == weights.shape.DimensionsProduct()) {
    const_cast<Tensor<OHWI, DataType::UINT8>&>(weights).MoveDataForCpuRearrange(
        weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  }
  ABSL_CHECK_GE(weights.data.size(), weights.shape.DimensionsProduct() +
                                          XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<size_t> reshape_order;
  const int dst_group_size = dst_weight_desc.GetOutputGroupSize();
  switch (dst_weight_desc.layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      if (weights.shape.o % (dst_group_size * 4) != 0 ||
          weights.shape.i % 4 != 0) {  // Need to pad
        RearrangeWeightsToOHWIOGroupI4O4(weights, dst_group_size, dst,
                                         pad_value);
        return;
      }
      reshape_order = {0, 3, 4, 1, 5, 2};
      break;
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      reshape_order = {3, 4, 0, 1, 5, 2};
      break;
    case WeightsLayout::kOSpatialIOGroupO4I4:
      reshape_order = {0, 3, 4, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupO4I4:
      reshape_order = {0, 4, 3, 1, 2, 5};
      break;
    case WeightsLayout::kOISpatialOGroupI4O4:
      reshape_order = {0, 4, 3, 1, 5, 2};
      break;
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
      reshape_order = {5, 3, 4, 0, 1, 2};
      break;
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      reshape_order = {2, 3, 4, 0, 1, 5};
      break;
    case WeightsLayout::kOICustomSpatialI4O4: {
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap, dst, pad_value);
      return;
    }
    case WeightsLayout::kOICustomSpatialO4I4: {
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap, dst, pad_value);
      return;
    }
    case WeightsLayout::kCustomGroups: {
      RearrangeWeightsToCustom(weights, dst_weight_desc.group_sizes,
                               dst_weight_desc.spatial_remap, dst, pad_value);
      return;
    }
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      ABSL_CHECK(false) << "Not implemented";
      return;
    case WeightsLayout::kUnknown:
      return;
  }

  Reshape(weights.Data(), weights.shape, dst_group_size, reshape_order,
          /*pad_value=*/pad_value, reinterpret_cast<uint8_t*>(dst.data()));
}

void RearrangeWeightsInt8AsUint8(const Tensor<OHWI, DataType::INT8>& weights,
                                 const WeightsDescription& dst_weight_desc,
                                 absl::Span<uint8_t> dst, int shift_value,
                                 unsigned int pad_value) {
  Tensor<OHWI, DataType::UINT8> weights_ui8;
  weights_ui8.shape = weights.shape;
  weights_ui8.data.resize(weights_ui8.shape.DimensionsProduct() +
                          XNN_EXTRA_BYTES / sizeof(uint8_t));
  const int8_t* src = weights.Data();
  for (int i = 0; i < weights.shape.DimensionsProduct(); ++i) {
    weights_ui8.data[i] = src[i] + shift_value;
  }
  RearrangeWeights(weights_ui8, dst_weight_desc, dst,
                   static_cast<uint8_t>(pad_value));
}

void RearrangeWeightsInt8AsUint4(const Tensor<OHWI, DataType::INT8>& weights,
                                 const WeightsDescription& dst_weight_desc,
                                 absl::Span<uint8_t> dst, int shift_value,
                                 unsigned int pad_value) {
  std::vector<uint8_t> weights_ui8(dst.size() * 2);
  RearrangeWeightsInt8AsUint8(weights, dst_weight_desc,
                              absl::MakeSpan(weights_ui8), shift_value,
                              pad_value);
  for (int i = 0; i < dst.size(); ++i) {
    uint32_t v0 = weights_ui8[i * 2 + 0];
    uint32_t v1 = weights_ui8[i * 2 + 1];
    dst[i] = (v1 << 4) | v0;
  }
}

void RearrangeWeightsInt8AsUint2(const Tensor<OHWI, DataType::INT8>& weights,
                                 const WeightsDescription& dst_weight_desc,
                                 absl::Span<uint8_t> dst, int shift_value,
                                 unsigned int pad_value) {
  std::vector<uint8_t> weights_ui8(dst.size() * 4);
  RearrangeWeightsInt8AsUint8(weights, dst_weight_desc,
                              absl::MakeSpan(weights_ui8), shift_value,
                              pad_value);
  for (int i = 0; i < dst.size(); ++i) {
    uint32_t v0 = weights_ui8[i * 4 + 0];
    uint32_t v1 = weights_ui8[i * 4 + 1];
    uint32_t v2 = weights_ui8[i * 4 + 2];
    uint32_t v3 = weights_ui8[i * 4 + 3];
    dst[i] = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0;
  }
}

void RearrangeWeightsInt4(const Tensor<OHWI, DataType::INT8>& weights_i4,
                          const WeightsDescription& dst_weight_desc,
                          absl::Span<uint8_t> dst) {
  WeightsDescription weight_desc = dst_weight_desc;
  weight_desc.type = DataType::INT8;
  const int elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_i4.shape);

  std::vector<uint8_t> weights_data(elements_count +
                                    XNN_EXTRA_BYTES / sizeof(uint8_t));
  RearrangeWeights(weights_i4, weight_desc, absl::MakeSpan(weights_data));

  int8_t* src_ptr = (int8_t*)weights_data.data();
  for (int i = 0; i < elements_count; i += 2) {
    int8_t iv0 = src_ptr[i];
    int8_t iv1 = src_ptr[i + 1];
    uint32_t v0 = iv0 >= 0 ? iv0 : 16 + iv0;
    uint32_t v1 = iv1 >= 0 ? iv1 : 16 + iv1;
    dst[i / 2] = (v1 << 4) | v0;
  }
}

void RearrangeWeightsUint2(const Tensor<OHWI, DataType::UINT8>& weights_i2,
                           const WeightsDescription& dst_weight_desc,
                           absl::Span<uint8_t> dst) {
  WeightsDescription weight_desc = dst_weight_desc;
  weight_desc.type = DataType::UINT8;
  const int elements_count =
      GetTotalElementsCountForLayout(weight_desc, weights_i2.shape);

  std::vector<uint8_t> weights_data(elements_count +
                                    XNN_EXTRA_BYTES / sizeof(uint8_t));
  RearrangeWeights(weights_i2, weight_desc, absl::MakeSpan(weights_data));

  uint8_t* src_ptr = (uint8_t*)weights_data.data();
  for (int i = 0; i < elements_count; i += 4) {
    uint32_t v0 = src_ptr[i];
    uint32_t v1 = src_ptr[i + 1];
    uint32_t v2 = src_ptr[i + 2];
    uint32_t v3 = src_ptr[i + 3];
    dst[i / 4] = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0;
  }
}

Tensor<Linear, DataType::INT32> GetWeightsAccumulatedInputChannels(
    const Tensor<OHWI, DataType::INT8>& weights) {
  Tensor<Linear, DataType::INT32> weights_sum_i;
  weights_sum_i.shape = Linear(weights.shape.o);
  weights_sum_i.data.resize(weights_sum_i.shape.DimensionsProduct());
  const int8_t* src = weights.Data();
  for (int o = 0; o < weights.shape.o; ++o) {
    int sum_i = 0;
    for (int h = 0; h < weights.shape.h; ++h) {
      for (int w = 0; w < weights.shape.w; ++w) {
        for (int i = 0; i < weights.shape.i; ++i) {
          sum_i += src[weights.shape.LinearIndex({o, h, w, i})];
        }
      }
    }
    weights_sum_i.data[o] = sum_i;
  }
  return weights_sum_i;
}

absl::Status RearrangeWeightsUInt4Packed(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims) {
  switch (dst_weight_desc.layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      Uint4PackedkOSpatialIOGroupI4O4(weights, dst_weight_desc, dst,
                                      weights_sum_i, pad_value, swap_dims);
      return absl::OkStatus();
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      Uint4Packedk2DYIsSpatialIOAndXIsOGroupI4O4(
          weights, dst_weight_desc, dst, weights_sum_i, pad_value, swap_dims);
      return absl::OkStatus();
    default:
      return absl::UnimplementedError("Unsupported layout");
  }
}

absl::Status RearrangeWeightsUInt2Packed(
    const Tensor<OHWI, DataType::UINT8>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst,
    absl::Span<int32_t> weights_sum_i, unsigned int pad_value, bool swap_dims) {
  switch (dst_weight_desc.layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      Uint2PackedkOSpatialIOGroupI4O4(weights, dst_weight_desc, dst,
                                      weights_sum_i, pad_value, swap_dims);
      return absl::OkStatus();
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      Uint2Packedk2DYIsSpatialIOAndXIsOGroupI4O4(
          weights, dst_weight_desc, dst, weights_sum_i, pad_value, swap_dims);
      return absl::OkStatus();
    default:
      return absl::UnimplementedError("Unsupported layout");
  }
}

std::vector<TensorDescriptor> GetTensorDescriptorsForWeightsLayout(
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& weights_desc) {
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);
  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  std::vector<TensorDescriptor> weights_tensors;
  if (weights_desc.layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weights_desc.layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    weights_tensors.resize(4);
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    for (int i = 0; i < 4; ++i) {
      weights_tensors[i] = TensorDescriptor(
          weights_desc.type, TensorStorageType::TEXTURE_2D, Layout::HW);
      weights_tensors[i].SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
      weights_tensors[i].UploadDataRaw(absl::MakeConstSpan(
          weights_data.data() +
              tex_size.x * tex_size.y * 4 * SizeOf(weights_desc.type) * i,
          weights_tensors[i].GetMemorySizeInBytes()));
    }
  } else {
    weights_tensors.resize(1);
    weights_tensors[0] = TensorDescriptor(
        weights_desc.type, TensorStorageType::BUFFER, Layout::LINEAR);
    weights_tensors[0].SetBHWCShape(BHWC(1, 1, 1, flt_count));
    weights_tensors[0].UploadDataRaw(absl::MakeConstSpan(weights_data));
  }
  return weights_tensors;
}

TensorDescriptor GetTensorDescriptorForWeightsLayout(
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights,
    const WeightsDescription& weights_desc) {
  const int elements_per_8bits = 8 / SizeInBitsOf(weights_desc.type);
  std::vector<uint8_t> data(
      GetTotalElementsCountForLayout(weights_desc, weights.shape) /
      elements_per_8bits);
  if (weights_desc.type == DataType::UINT8) {
    RearrangeWeightsInt8AsUint8(weights, weights_desc, absl::MakeSpan(data),
                                128, 128u);
  } else if (weights_desc.type == DataType::UINT4) {
    RearrangeWeightsInt8AsUint4(weights, weights_desc, absl::MakeSpan(data), 8,
                                8u);
  } else if (weights_desc.type == DataType::UINT2) {
    RearrangeWeightsInt8AsUint2(weights, weights_desc, absl::MakeSpan(data), 2,
                                2u);
  }

  TensorDescriptor weights_td;
  if (weights_desc.IsLinearLayout()) {
    weights_td = TensorDescriptor(DataType::UINT8, TensorStorageType::BUFFER,
                                  Layout::LINEAR);
    weights_td.SetBHWCShape(BHWC(1, 1, 1, data.size()));
  } else {
    DataType texture_type = DataType::UINT32;
    if (weights_desc.type == DataType::UINT4) {
      texture_type = DataType::UINT16;
    } else if (weights_desc.type == DataType::UINT2) {
      texture_type = DataType::UINT8;
    }
    weights_td = TensorDescriptor(texture_type, TensorStorageType::TEXTURE_2D,
                                  Layout::HW);
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    tex_size.x /= 4;  // because we store 16 elements per pixel
    weights_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }
  weights_td.UploadDataRaw(absl::MakeConstSpan(data));

  return weights_td;
}

std::vector<TensorDescriptor> GetTensorDescriptorsForWeightsLayout(
    const OHWI& weights_shape, const WeightsDescription& weights_desc) {
  TensorDescriptor weights_td;
  if (weights_desc.layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weights_desc.layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights_shape);
    weights_td = TensorDescriptor(weights_desc.type,
                                  TensorStorageType::TEXTURE_2D, Layout::HW);
    weights_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 4));
    return std::vector<TensorDescriptor>(4, weights_td);
  } else if (weights_desc.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights_shape);
    DataType texture_type;
    if (weights_desc.type == DataType::UINT8) {
      texture_type = DataType::UINT32;
    } else if (weights_desc.type == DataType::UINT4) {
      texture_type = DataType::UINT16;
    } else if (weights_desc.type == DataType::UINT2) {
      texture_type = DataType::UINT8;
    } else {
      // TODO: b/378522761 - Support other data types.
      ABSL_LOG(FATAL) << absl::StrCat(
          "Weights conversion to k2DYIsSpatialIOAndXIsOGroupI4O4 layout with "
          "data "
          "type ",
          ToString(weights_desc.type), " is unsupported.");
    }
    tex_size.x /= SizeInBitsOf(texture_type) / SizeInBitsOf(weights_desc.type);
    weights_td = TensorDescriptor(texture_type, TensorStorageType::TEXTURE_2D,
                                  Layout::HW);
    weights_td.SetBHWCShape(BHWC(1, tex_size.y, tex_size.x, 1));
    return {weights_td};
  } else {
    weights_td = TensorDescriptor(weights_desc.type, TensorStorageType::BUFFER,
                                  Layout::LINEAR);
    size_t elements_count =
        GetTotalElementsCountForLayout(weights_desc, weights_shape);
    // For Int4 or UInt4 weights, we would store them in packed format: two
    // values are stored in one element (one byte).
    if (weights_desc.type == DataType::INT4 ||
        weights_desc.type == DataType::UINT4) {
      elements_count = DivideRoundUp(elements_count, 2);
    } else if (weights_desc.type == DataType::INT2 ||
               weights_desc.type == DataType::UINT2) {
      elements_count = DivideRoundUp(elements_count, 4);
    }
    weights_td.SetBHWCShape(BHWC(1, 1, 1, elements_count));
    return {weights_td};
  }
}

}  // namespace ml_drift
