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

#ifndef ML_DRIFT_COMMON_DATA_TYPE_H_
#define ML_DRIFT_COMMON_DATA_TYPE_H_

#include <stddef.h>

#include <string>

namespace ml_drift {

enum class DataType {
  UNKNOWN = 0,
  FLOAT16 = 1,
  FLOAT32 = 2,
  FLOAT64 = 3,
  BFLOAT16 = 4,
  UINT8 = 5,
  INT8 = 6,
  UINT16 = 7,
  INT16 = 8,
  UINT32 = 9,
  INT32 = 10,
  UINT64 = 11,
  INT64 = 12,
  BOOL = 13,
  INT4 = 14,
  UINT4 = 15,
  INT3 = 16,
  UINT3 = 17,
  INT2 = 18,
  UINT2 = 19,
  INT1 = 20,
  UINT1 = 21,
};

enum class PackedType {
  kUnknown = 0,
  kUint8C4 = 1,
  kInt8C4 = 2,
  kUint8W4C4 = 3,
  kInt8W4C4 = 4,
  kInt8C16 = 5,
  kUint8C16 = 6,
  kInt4C32 = 7,
  kUint4C32 = 8,
};

// Converts PackedType to a string representation.
std::string ToString(PackedType type);
// Returns true if the PackedType is signed.
bool IsSigned(PackedType type);
// Returns the size of the PackedType in bits.
size_t GetTypeSizeInBits(PackedType type);
// Converts PackedType to a spatial tensor type.
DataType ToSpatialTensorType(PackedType type);

// Returns the size of the DataType in bytes.
size_t SizeOf(DataType data_type);
// Returns the size of the DataType in bits.
size_t SizeInBitsOf(DataType data_type);
// Returns true if the DataType is a float type.
bool IsFloatType(DataType data_type);
// Returns true if the DataType is signed.
bool IsSigned(DataType type);

// Converts DataType to a string representation.
std::string ToString(DataType data_type);

// Converts DataType to an OpenCL data type string.
std::string ToCLDataType(DataType data_type, int vec_size = 1);

// Converts DataType to a WebGPU data type string.
std::string ToWebGpuDataType(DataType data_type, int vec_size = 1);

// Converts DataType to a Metal data type string.
std::string ToMetalDataType(DataType data_type, int vec_size = 1);

// Converts DataType to a Metal texture type.
DataType ToMetalTextureType(DataType data_type);

// Converts DataType to a GLSL shader data type string.
// When add_precision enabled it will add:
//   highp for INT32/UINT32/FLOAT32
//   mediump for INT16/UINT16/FLOAT16(if explicit_fp16 not enabled)
//   lowp for INT8/UINT8
std::string ToGlslShaderDataType(DataType data_type, int vec_size = 1,
                                 bool add_precision = false,
                                 bool explicit_fp16 = false);

// Converts DataType to a WebGPU type string.
std::string ToWebGpuType(DataType data_type, int vec_size, bool explicit_fp16);

// Converts DataType to a UCL data type string.
std::string ToUclDataType(DataType data_type, int vec_size = 1);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_DATA_TYPE_H_
