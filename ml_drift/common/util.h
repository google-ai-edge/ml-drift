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

#ifndef ML_DRIFT_COMMON_UTIL_H_
#define ML_DRIFT_COMMON_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class RuntimeShape {
 public:
  // The maximum number of dimensions supported.
  static constexpr int kMaxDimensions = 5;

  RuntimeShape() : size_(0) {}

  explicit RuntimeShape(absl::Span<const int32_t> dims_data);

  bool operator==(const RuntimeShape& other) const;

  int DimensionsCount() const;

  int32_t Dims(int i) const;

  absl::Span<const int32_t> DimsData() const;

  int64_t FlatSize() const;

 private:
  int size_;
  int32_t dims_[kMaxDimensions];
};

struct DequantizationParams {
  float scale;
  int32_t zero_point;
};

struct PerChannelDequantizationParams {
  const float* scale;
  const int32_t* zero_point;
  int32_t quantized_dimension;
};

struct BlockwiseDequantizationParams {
  const float* scale;
  const int32_t* zero_point;
  int32_t quantized_dimension;
  int32_t group_size;
};

RuntimeShape GetTensorShape(const std::vector<int32_t>& data);

int MatchingFlatSize(const RuntimeShape& shape,
                     const RuntimeShape& check_shape_0);

template <typename IndexType>
bool NextIndex(const absl::Span<const int32_t> dims, IndexType* current) {
  if (dims.empty()) {
    return false;
  }
  int carry = 1;
  for (int idx = dims.size() - 1; idx >= 0; --idx) {
    IndexType current_val = current[idx] + carry;
    ABSL_CHECK_GE(dims[idx], current_val);
    if (dims[idx] == current_val) {
      current[idx] = 0;
    } else {
      current[idx] = current_val;
      carry = 0;
      break;
    }
  }
  return (carry == 0);
}

size_t ReducedOutputOffset(absl::Span<const int32_t> dims,
                           absl::Span<const int32_t> index,
                           absl::Span<const int32_t> axis);

template <typename InputT, typename OutputT>
void Dequantize(const DequantizationParams& op_params,
                const RuntimeShape& input_shape, const InputT* input_data,
                const RuntimeShape& output_shape, OutputT* output_data) {
  int32_t zero_point = op_params.zero_point;
  const float scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const int32_t val = input_data[i];
    const OutputT result = static_cast<OutputT>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

template <typename T>
void PerChannelDequantize(const PerChannelDequantizationParams& op_params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const RuntimeShape& output_shape,
                          float* output_data) {
  MatchingFlatSize(input_shape, output_shape);

  const int32_t* zero_point = op_params.zero_point;
  const float* scale = op_params.scale;
  const int32_t quantized_dimension = op_params.quantized_dimension;
  const int32_t num_dims = input_shape.DimensionsCount();
  absl::Span<const int32_t> dims_data = input_shape.DimsData();
  std::vector<int> current_dim(num_dims, 0);

  do {
    size_t offset = ReducedOutputOffset(dims_data, current_dim, {});
    const int channel = current_dim[quantized_dimension];
    const int32_t val = input_data[offset];
    const float result =
        static_cast<float>(scale[channel] * (val - zero_point[channel]));
    output_data[offset] = result;
  } while (NextIndex(dims_data, current_dim.data()));
}

template <typename T>
void BlockwiseDequantize(const BlockwiseDequantizationParams& op_params,
                         const RuntimeShape& input_shape, const T* input_data,
                         const RuntimeShape& output_shape, float* output_data) {
  MatchingFlatSize(input_shape, output_shape);

  const int32_t* zero_point = op_params.zero_point;
  const float* scale = op_params.scale;
  const int32_t quantized_dimension = op_params.quantized_dimension;
  const int32_t num_dims = input_shape.DimensionsCount();
  absl::Span<const int32_t> dims_data = input_shape.DimsData();
  std::vector<int> current_dim(num_dims, 0);

  const int group_size = op_params.group_size;
  const int group_dimension = num_dims - 1;
  const int group_index = dims_data[group_dimension] / group_size;

  std::vector<int32_t> scale_dims_data(num_dims, 1);
  scale_dims_data[quantized_dimension] = dims_data[quantized_dimension];
  scale_dims_data[group_dimension] = group_index;

  std::vector<int> scale_dim(num_dims, 0);

  do {
    size_t offset = ReducedOutputOffset(dims_data, current_dim, {});
    // Construct scale_dim: (o, 0, 0, g)
    scale_dim[quantized_dimension] = current_dim[quantized_dimension];
    scale_dim[group_dimension] = current_dim[group_dimension] / group_size;
    size_t scale_offset = ReducedOutputOffset(scale_dims_data, scale_dim, {});
    const int32_t val = input_data[offset];
    const float result = static_cast<float>(scale[scale_offset] *
                                            (val - zero_point[scale_offset]));
    output_data[offset] = result;
  } while (NextIndex(dims_data, current_dim.data()));
}

void UnpackPackedIntToInt8(const int8_t* src_buffer, int num_elements,
                           int bit_width, int8_t* dst_buffer);

void UnpackDenseInt4IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer);

void UnpackDenseInt2IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer);

// @param n must be non negative
// @param divisor must be greater than zero
template <typename T, typename N>
T DivideRoundUp(T n, N divisor) {
  const T div = static_cast<T>(divisor);
  const T q = n / div;
  return n % div == 0 ? q : q + 1;
}

template <>
inline uint3 DivideRoundUp(uint3 n, uint3 divisor) {
  return uint3(DivideRoundUp(n.x, divisor.x), DivideRoundUp(n.y, divisor.y),
               DivideRoundUp(n.z, divisor.z));
}

// @param number or its components must be greater than zero
// @param n must be greater than zero
template <typename T, typename N>
T AlignByN(T number, N n) {
  return DivideRoundUp(number, n) * n;
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info);

float GetEpsilon(DataType data_type, const GpuInfo& gpu_info);

std::vector<uint16_t> FloatToBFloat(const std::vector<float>& src);

std::vector<float> BFloatToFloat(const std::vector<uint16_t>& src);

BHWC GetShapeForPackedType(const BHWC& shape, PackedType packed_type);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_UTIL_H_
