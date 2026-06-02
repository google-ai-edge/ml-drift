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

#include "ml_drift/common/util.h"

#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

RuntimeShape::RuntimeShape(const absl::Span<const int32_t> dims_data)
    : size_(dims_data.size()) {
  ABSL_CHECK_LE(size_, kMaxDimensions);
  std::memcpy(dims_, dims_data.data(), sizeof(int32_t) * size_);
}

int RuntimeShape::DimensionsCount() const { return size_; }

int32_t RuntimeShape::Dims(int i) const {
  ABSL_CHECK_GE(i, 0);
  ABSL_CHECK_LT(i, size_);
  return dims_[i];
}

absl::Span<const int32_t> RuntimeShape::DimsData() const {
  return absl::MakeConstSpan(dims_, size_);
}

int64_t RuntimeShape::FlatSize() const {
  int64_t flat_size = 1;
  for (int i = 0; i < size_; ++i) {
    flat_size *= dims_[i];
  }
  return flat_size;
}

RuntimeShape GetTensorShape(const std::vector<int32_t>& data) {
  for (int i = 0; i < data.size(); ++i) {
    ABSL_CHECK_GE(data[i], 0);
  }
  return RuntimeShape(data);
}

int MatchingFlatSize(const RuntimeShape& shape,
                     const RuntimeShape& check_shape_0) {
  ABSL_CHECK_EQ(shape.DimensionsCount(), check_shape_0.DimensionsCount());
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    ABSL_CHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return shape.FlatSize();
}

size_t ReducedOutputOffset(const absl::Span<const int32_t> dims,
                           const absl::Span<const int32_t> index,
                           const absl::Span<const int32_t> axis) {
  if (dims.empty()) {
    return 0;
  }
  size_t offset = 0;
  for (int idx = 0; idx < dims.size(); ++idx) {
    bool is_axis = false;
    for (int axis_idx = 0; axis_idx < axis.size(); ++axis_idx) {
      if (idx == axis[axis_idx]) {
        is_axis = true;
        break;
      }
    }
    if (!is_axis) {
      offset = offset * static_cast<size_t>(dims[idx]) +
               static_cast<size_t>(index[idx]);
    }
  }
  return offset;
}

void UnpackPackedIntToInt8(const int8_t* src_buffer, int num_elements,
                           int bit_width, int8_t* dst_buffer) {
  assert(bit_width == 2 || bit_width == 4);
  if (bit_width == 4) {
    // num_elements means the number of elements regardless of packed or
    // unpacked. For example, 3 elements means both
    //   1) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
    //      stored in src_buffer[0] and src_buffer[1] (i = 0..1)
    //   2) Unpacked: 3 int8's = 3 bytes.
    //.     stored in dst_buffer[0], dst_buffer[1] and dst_buffer[2] (j = 0..2)
    for (int i = 0; i < num_elements / 2; i++) {
      int8_t byte = src_buffer[i];
      // Shift left first so that sign is properly extended when shifted right
      int8_t lower = static_cast<int8_t>(byte << 4) >> 4;
      int8_t higher = byte >> 4;
      dst_buffer[2 * i] = lower;
      dst_buffer[2 * i + 1] = higher;
    }

    // If the buffer size is odd, extract the final lower nibble.
    if (num_elements % 2 != 0) {
      dst_buffer[num_elements - 1] =
          static_cast<int8_t>(src_buffer[num_elements / 2] << 4) >> 4;
    }
  } else if (bit_width == 2) {
    for (int i = 0; i < num_elements / 4; i++) {
      int8_t byte = src_buffer[i];
      // Shift left first so that sign is properly extended when shifted right
      int8_t val1 = static_cast<int8_t>(byte << 6) >> 6;
      int8_t val2 = static_cast<int8_t>((byte << 4) & 0xFF) >> 6;
      int8_t val3 = static_cast<int8_t>((byte << 2) & 0xFF) >> 6;
      int8_t val4 = byte >> 6;
      dst_buffer[4 * i] = val1;
      dst_buffer[4 * i + 1] = val2;
      dst_buffer[4 * i + 2] = val3;
      dst_buffer[4 * i + 3] = val4;
    }

    // Handle the remaining elements.
    int remaining_elements = num_elements % 4;
    if (remaining_elements > 0) {
      int8_t byte = src_buffer[num_elements / 4];
      for (int i = 0; i < remaining_elements; i++) {
        dst_buffer[num_elements - remaining_elements + i] =
            static_cast<int8_t>((byte << (6 - 2 * i)) & 0xFF) >> 6;
      }
    }
  }
}

void UnpackDenseInt4IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer) {
  for (int i = 0; i < num_elements / 2; i++) {
    int8_t byte = src_buffer[i];
    // Shift left first so that sign is properly extended when shifted right
    int8_t lower = static_cast<int8_t>(byte << 4) >> 4;
    int8_t higher = byte >> 4;
    dst_buffer[2 * i] = lower;
    dst_buffer[2 * i + 1] = higher;
  }

  // If the buffer size is odd, extract the final lower nibble.
  if (num_elements % 2 != 0) {
    dst_buffer[num_elements - 1] =
        static_cast<int8_t>(src_buffer[num_elements / 2] << 4) >> 4;
  }
}

void UnpackDenseInt2IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer) {
  for (int i = 0; i < num_elements / 4; i++) {
    int8_t byte = src_buffer[i];
    // Shift left first so that sign is properly extended when shifted right
    int8_t val1 = static_cast<int8_t>(byte << 6) >> 6;
    int8_t val2 = static_cast<int8_t>((byte << 4) & 0xFF) >> 6;
    int8_t val3 = static_cast<int8_t>((byte << 2) & 0xFF) >> 6;
    int8_t val4 = byte >> 6;
    dst_buffer[4 * i] = val1;
    dst_buffer[4 * i + 1] = val2;
    dst_buffer[4 * i + 2] = val3;
    dst_buffer[4 * i + 3] = val4;
  }

  // Handle the remaining elements.
  int remaining_elements = num_elements % 4;
  if (remaining_elements > 0) {
    int8_t byte = src_buffer[num_elements / 4];
    for (int i = 0; i < remaining_elements; i++) {
      dst_buffer[num_elements - remaining_elements + i] =
          static_cast<int8_t>((byte << (6 - 2 * i)) & 0xFF) >> 6;
    }
  }
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info) {
  constexpr float kHalfEpsilon = 9.7656e-4;
  switch (precision) {
    case CalculationsPrecision::F32:
      return gpu_info.opencl_info.supports_fp32_rtn ? FLT_EPSILON
                                                    : 4 * FLT_EPSILON;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      return gpu_info.opencl_info.supports_fp16_rtn ? kHalfEpsilon
                                                    : 4 * kHalfEpsilon;
  }
}

float GetEpsilon(DataType data_type, const GpuInfo& gpu_info) {
  constexpr float kHalfEpsilon = 9.7656e-4;
  switch (data_type) {
    case DataType::FLOAT32:
      return gpu_info.opencl_info.supports_fp32_rtn ? FLT_EPSILON
                                                    : 4 * FLT_EPSILON;
    case DataType::FLOAT16:
      return gpu_info.opencl_info.supports_fp16_rtn ? kHalfEpsilon
                                                    : 4 * kHalfEpsilon;
    case DataType::UNKNOWN:
    case DataType::FLOAT64:
    case DataType::BFLOAT16:
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::UINT16:
    case DataType::INT16:
    case DataType::UINT32:
    case DataType::INT32:
    case DataType::UINT64:
    case DataType::INT64:
    case DataType::BOOL:
    case DataType::INT4:
    case DataType::UINT4:
    case DataType::INT3:
    case DataType::UINT3:
    case DataType::INT2:
    case DataType::UINT2:
    case DataType::INT1:
    case DataType::UINT1:
      break;
  }
  return 0.0f;
}

std::vector<uint16_t> FloatToBFloat(const std::vector<float>& src) {
  std::vector<uint16_t> dst(src.size());
  for (int i = 0; i < src.size(); ++i) {
    uint32_t src_ui32 = *((uint32_t*)&src[i]);
    if ((src_ui32 & 0x00018000) == 0x00018000) {
      src_ui32 += 0x00008000;  // round towards even
    }
    src_ui32 >>= 16;
    dst[i] = static_cast<uint16_t>(src_ui32);
  }
  return dst;
}

std::vector<float> BFloatToFloat(const std::vector<uint16_t>& src) {
  std::vector<float> dst(src.size());
  for (int i = 0; i < src.size(); ++i) {
    uint32_t src_ui32 = static_cast<uint32_t>(src[i]);
    src_ui32 <<= 16;
    dst[i] = *((float*)&src_ui32);
  }
  return dst;
}

BHWC GetShapeForPackedType(const BHWC& shape, PackedType packed_type) {
  BHWC dst_shape = shape;
  if (packed_type == PackedType::kUint8W4C4 ||
      packed_type == PackedType::kInt8W4C4) {
    dst_shape.w = DivideRoundUp(dst_shape.w, 4);
  } else if (packed_type == PackedType::kUint8C16 ||
             packed_type == PackedType::kInt8C16) {
    dst_shape.c = DivideRoundUp(dst_shape.c, 4);
  } else if (packed_type == PackedType::kUint4C32 ||
             packed_type == PackedType::kInt4C32) {
    dst_shape.c = DivideRoundUp(dst_shape.c, 8);
  }
  return dst_shape;
}

}  // namespace ml_drift
