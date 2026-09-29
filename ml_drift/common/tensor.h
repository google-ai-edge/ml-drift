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

#ifndef ML_DRIFT_COMMON_TENSOR_H_
#define ML_DRIFT_COMMON_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace internal_tensor {

// Meta function given element type returns a type for Tensor data container.
template <DataType Type>
struct StorageType;

template <>
struct StorageType<DataType::kFloat32> {
  using value = std::vector<float>;
  using value_span = absl::Span<float>;
  using value_type = float;
};

template <>
struct StorageType<DataType::kFloat16> {
  using value = std::vector<half>;
  using value_span = absl::Span<half>;
  using value_type = half;
};

template <>
struct StorageType<DataType::kBfloat16> {
  using value = std::vector<uint16_t>;
  using value_span = absl::Span<uint16_t>;
  using value_type = uint16_t;
};

template <>
struct StorageType<DataType::kInt32> {
  using value = std::vector<int32_t>;
  using value_span = absl::Span<int32_t>;
  using value_type = int32_t;
};

template <>
struct StorageType<DataType::kInt16> {
  using value = std::vector<int16_t>;
  using value_span = absl::Span<int16_t>;
  using value_type = int16_t;
};

template <>
struct StorageType<DataType::kInt2> {
  using value = std::vector<int8_t>;
  using value_span = absl::Span<int8_t>;
  using value_type = int8_t;
};

template <>
struct StorageType<DataType::kInt4> {
  using value = std::vector<int8_t>;
  using value_span = absl::Span<int8_t>;
  using value_type = int8_t;
};

template <>
struct StorageType<DataType::kInt8> {
  using value = std::vector<int8_t>;
  using value_span = absl::Span<int8_t>;
  using value_type = int8_t;
};

template <>
struct StorageType<DataType::kUint32> {
  using value = std::vector<uint32_t>;
  using value_span = absl::Span<uint32_t>;
  using value_type = uint32_t;
};

template <>
struct StorageType<DataType::kUint16> {
  using value = std::vector<uint16_t>;
  using value_span = absl::Span<uint16_t>;
  using value_type = uint16_t;
};

template <>
struct StorageType<DataType::kUint8> {
  using value = std::vector<uint8_t>;
  using value_span = absl::Span<uint8_t>;
  using value_type = uint8_t;
};

template <>
struct StorageType<DataType::kBool> {
  using value = std::vector<uint8_t>;
  using value_span = absl::Span<uint8_t>;
  using value_type = bool;
};

}  // namespace internal_tensor

template <typename ShapeT, DataType Type>
struct Tensor {
  using ShapeType = ShapeT;

  constexpr static DataType kType = Type;

  using TensorStorageType = typename internal_tensor::StorageType<Type>::value;
  using SpannedStorageType =
      typename internal_tensor::StorageType<Type>::value_span;
  using ValueType = typename internal_tensor::StorageType<Type>::value_type;
  // Opaque id of a tensor.
  int64_t id = -1;

  ShapeType shape;

  TensorStorageType data;
  // TODO: b/389755820 - Support variant<data, span>;
  SpannedStorageType spanned_data;

  const ValueType* Data() const {
    return !data.empty() ? data.data() : spanned_data.data();
  }

  ValueType* Data() {
    return !data.empty() ? data.data() : spanned_data.data();
  }

  // Returns the number of elements in the underlying storage buffer.
  // For standard data types like FLOAT32, INT32 or INT8, this is equal to the
  // number of elements in the tensor (`shape.DimensionsProduct()`).
  // For packed data types like INT4, this returns the size of the backing
  // buffer (e.g., in bytes if the storage is `int8_t`). This size may include
  // padding. For example, a tensor with 3 INT4 elements would be stored in a
  // 2-byte buffer, and `size()` would return 2.
  size_t size() const {
    return !data.empty() ? data.size() : spanned_data.size();
  }

  void MoveDataForCpuRearrange(size_t size) {
    data.resize(size);
    if (!spanned_data.empty()) {
      std::memcpy(data.data(), spanned_data.data(),
             spanned_data.size() * sizeof(ValueType));
      spanned_data = {};
    }
  }

  bool empty() const { return data.empty() && spanned_data.empty(); }

  ValueType Get(size_t index) const { return *(Data() + index); }
  void Set(size_t index, const ValueType& value) { *(Data() + index) = value; }

  // Explicitly disable type conversions
  template <typename Other>
  void Set(size_t index, const Other& value) = delete;
};

// TensorRef is a reference to another tensor. If an object should never hold
// tensor data, then TensorRef should be used instead.
template <typename ShapeT>
struct TensorRef {
  using ShapeType = ShapeT;

  DataType type = DataType::kUnknown;

  ShapeT shape;

  // Opaque reference to a tensor. Upstream component is responsible for
  // resolving this reference into an actual tensor.
  int64_t ref = -1;

  // Specifies if the tensor should be a variable input tensor that must be an
  // output as well as an input to the graph.
  bool is_variable_input = false;
};

template <typename ShapeT, DataType Type>
constexpr DataType Tensor<ShapeT, Type>::kType;

template <typename ShapeT, DataType Type>
Tensor<ShapeT, Type> MakeZeroTensor(const ShapeT& shape) {
  Tensor<ShapeT, Type> tensor;
  tensor.shape = shape;
  const size_t vector_element_size =
      sizeof(typename Tensor<ShapeT, Type>::TensorStorageType::value_type);
  tensor.data = typename Tensor<ShapeT, Type>::TensorStorageType(
      shape.DimensionsProduct() * SizeOf(Type) / vector_element_size, 0);
  return tensor;
}

using TensorBool = Tensor<BHWC, DataType::kBool>;
using TensorFloat32 = Tensor<BHWC, DataType::kFloat32>;
using TensorFloat16 = Tensor<BHWC, DataType::kFloat16>;
using TensorInt32 = Tensor<BHWC, DataType::kInt32>;
using Tensor5DFloat32 = Tensor<BHWDC, DataType::kFloat32>;

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TENSOR_H_
