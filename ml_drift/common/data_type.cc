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

#include "ml_drift/common/data_type.h"

#include <stddef.h>

#include <string>

#include "absl/strings/str_cat.h"

namespace ml_drift {
namespace {
std::string ToGlslType(const std::string& scalar_type,
                       const std::string& vec_type, int vec_size) {
  return vec_size == 1 ? scalar_type : absl::StrCat(vec_type, vec_size);
}

std::string GetGlslPrecisionModifier(DataType data_type) {
  switch (data_type) {
    case DataType::UINT8:
    case DataType::INT8:
      return "lowp ";
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
      return "mediump ";
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return "highp ";
    case DataType::BOOL:
      return "";
    default:
      return "";
  }
}
}  // namespace

size_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::BOOL:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT3:
    case DataType::INT3:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
      return 1;
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::BFLOAT16:
      return 2;
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return 4;
    case DataType::FLOAT64:
    case DataType::INT64:
    case DataType::UINT64:
      return 8;
    case DataType::UNKNOWN:
      return 0;
  }
  return 0;
}

size_t SizeInBitsOf(DataType data_type) {
  switch (data_type) {
    case DataType::UINT8:
    case DataType::INT8:
      return 8;
    case DataType::BOOL:
      return 1;
    case DataType::UINT4:
    case DataType::INT4:
      return 4;
    case DataType::UINT3:
    case DataType::INT3:
      return 3;
    case DataType::UINT2:
    case DataType::INT2:
      return 2;
    case DataType::UINT1:
    case DataType::INT1:
      return 1;
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::BFLOAT16:
      return 16;
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return 32;
    case DataType::FLOAT64:
    case DataType::INT64:
    case DataType::UINT64:
      return 64;
    case DataType::UNKNOWN:
      return 0;
  }
  return 0;
}

bool IsFloatType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      return true;
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::BOOL:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT3:
    case DataType::INT3:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::INT64:
    case DataType::UINT64:
    case DataType::UNKNOWN:
      return false;
  }
  return false;
}

bool IsSigned(DataType data_type) {
  switch (data_type) {
    case DataType::UINT64:
    case DataType::UINT32:
    case DataType::UINT16:
    case DataType::UINT8:
    case DataType::UINT4:
    case DataType::UINT3:
    case DataType::UINT2:
    case DataType::UINT1:
      return false;
    case DataType::INT64:
    case DataType::INT32:
    case DataType::INT16:
    case DataType::INT8:
    case DataType::INT4:
    case DataType::INT3:
    case DataType::INT2:
    case DataType::INT1:
      return true;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      return false;
    case DataType::BOOL:
    case DataType::UNKNOWN:
      return false;
  }
  return false;
}

std::string ToString(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT16:
      return "float16";
    case DataType::FLOAT32:
      return "float32";
    case DataType::FLOAT64:
      return "float64";
    case DataType::BFLOAT16:
      return "bfloat16";
    case DataType::INT16:
      return "int16";
    case DataType::INT32:
      return "int32";
    case DataType::INT64:
      return "int64";
    case DataType::INT8:
      return "int8";
    case DataType::INT4:
      return "int4";
    case DataType::INT3:
      return "int3";
    case DataType::INT2:
      return "int2";
    case DataType::INT1:
      return "int1";
    case DataType::UINT16:
      return "uint16";
    case DataType::UINT32:
      return "uint32";
    case DataType::UINT64:
      return "uint64";
    case DataType::UINT8:
      return "uint8";
    case DataType::UINT4:
      return "uint4";
    case DataType::UINT3:
      return "uint3";
    case DataType::UINT2:
      return "uint2";
    case DataType::UINT1:
      return "uint1";
    case DataType::BOOL:
      return "bool";
    case DataType::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

std::string ToCLDataType(DataType data_type, int vec_size) {
  const std::string postfix = vec_size == 1 ? "" : std::to_string(vec_size);
  switch (data_type) {
    case DataType::FLOAT16:
      return "half" + postfix;
    case DataType::FLOAT32:
      return "float" + postfix;
    case DataType::FLOAT64:
      return "double" + postfix;
    case DataType::INT16:
      return "short" + postfix;
    case DataType::INT32:
      return "int" + postfix;
    case DataType::INT64:
      return "long" + postfix;
    case DataType::INT8:
      return "char" + postfix;
    case DataType::UINT16:
      return "ushort" + postfix;
    case DataType::UINT32:
      return "uint" + postfix;
    case DataType::UINT64:
      return "ulong" + postfix;
    case DataType::UINT8:
      return "uchar" + postfix;
    case DataType::BOOL:
      return "bool" + postfix;
    case DataType::UNKNOWN:
    case DataType::BFLOAT16:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT3:
    case DataType::INT3:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
      return "unknown";
  }
  return "undefined";
}

std::string ToWebGpuDataType(DataType data_type, int vec_size) {
  std::string scalar;
  // Tracks how many of each value are packed into the wgsl datatype. For
  // example, wgsl does not have an int8 type, so we need to pack 4 int8 values
  // into an i32.
  int num_packed = 0;
  switch (data_type) {
    case DataType::FLOAT16:
      scalar = "f16";
      num_packed = 1;
      break;
    case DataType::FLOAT32:
      scalar = "f32";
      num_packed = 1;
      break;
    case DataType::INT32:
      scalar = "i32";
      num_packed = 1;
      break;
    case DataType::INT8:
      scalar = "i32";
      num_packed = 4;
      break;
    case DataType::UINT32:
      scalar = "u32";
      num_packed = 1;
      break;
    case DataType::UINT8:
      scalar = "u32";
      num_packed = 4;
      break;
    case DataType::BOOL:
      scalar = "bool";
      num_packed = 1;
      break;
    case DataType::BFLOAT16:
    case DataType::UINT64:
    case DataType::UINT16:
    case DataType::INT64:
    case DataType::INT16:
    case DataType::FLOAT64:
    case DataType::UNKNOWN:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT3:
    case DataType::INT3:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
      return "unknown";
  }
  if (vec_size == 1) {
    return scalar;
  }
  if (vec_size % num_packed != 0) {
    return "undefined";
  }
  return absl::StrCat("vec", vec_size / num_packed, "<", scalar, ">");
}

std::string ToMetalDataType(DataType data_type, int vec_size) {
  const std::string postfix = vec_size == 1 ? "" : std::to_string(vec_size);
  switch (data_type) {
    case DataType::FLOAT16:
      return "half" + postfix;
    case DataType::FLOAT32:
      return "float" + postfix;
    case DataType::FLOAT64:
      return "double" + postfix;
    case DataType::INT16:
      return "short" + postfix;
    case DataType::INT32:
      return "int" + postfix;
    case DataType::INT64:
      return "long" + postfix;
    case DataType::INT8:
      return "char" + postfix;
    case DataType::BFLOAT16:
      return "bfloat" + postfix;
    case DataType::UINT16:
      return "ushort" + postfix;
    case DataType::UINT32:
      return "uint" + postfix;
    case DataType::UINT64:
      return "ulong" + postfix;
    case DataType::UINT8:
      return "uchar" + postfix;
    case DataType::BOOL:
      return "bool" + postfix;
    case DataType::UNKNOWN:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT3:
    case DataType::INT3:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
      return "unknown";
  }
  return "undefined";
}

DataType ToMetalTextureType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
    case DataType::INT32:
    case DataType::INT16:
    case DataType::UINT32:
    case DataType::UINT16:
      return data_type;
    case DataType::INT8:
      return DataType::INT16;
    case DataType::BFLOAT16:
    case DataType::UINT8:
    case DataType::BOOL:
      return DataType::UINT16;
    default:
      return DataType::UNKNOWN;
  }
}

std::string ToGlslShaderDataType(DataType data_type, int vec_size,
                                 bool add_precision, bool explicit_fp16) {
  const std::string precision_modifier =
      add_precision ? GetGlslPrecisionModifier(data_type) : "";
  switch (data_type) {
    case DataType::FLOAT16:
      if (explicit_fp16) {
        return ToGlslType("float16_t", "f16vec", vec_size);
      } else {
        return precision_modifier + ToGlslType("float", "vec", vec_size);
      }
    case DataType::FLOAT32:
      return precision_modifier + ToGlslType("float", "vec", vec_size);
    case DataType::FLOAT64:
      return precision_modifier + ToGlslType("double", "dvec", vec_size);
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::INT64:
      return precision_modifier + ToGlslType("int", "ivec", vec_size);
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::UINT64:
      return precision_modifier + ToGlslType("uint", "uvec", vec_size);
    case DataType::BOOL:
      return ToGlslType("bool", "bvec", vec_size);
    case DataType::UNKNOWN:
    case DataType::BFLOAT16:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT3:
    case DataType::INT3:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
      return "unknown";
  }
  return "unknown";
}

std::string ToWebGpuType(DataType data_type, int vec_size, bool explicit_fp16) {
  std::string type_name;
  if (data_type == DataType::FLOAT32) {
    type_name = "f32";
  } else if (data_type == DataType::FLOAT16) {
    type_name = explicit_fp16 ? "f16" : "f32";
  } else if (data_type == DataType::INT32 || data_type == DataType::INT16 ||
             data_type == DataType::INT8) {
    type_name = "i32";
  } else if (data_type == DataType::UINT32 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT8) {
    type_name = "u32";
  } else if (data_type == DataType::BOOL) {
    type_name = "bool";
  } else {
    return "no_type";
  }
  if (vec_size != 1) {
    type_name = absl::StrCat("vec", vec_size, "<", type_name, ">");
  }
  return type_name;
}

std::string ToUclDataType(DataType data_type, int vec_size) {
  const std::string postfix = vec_size == 1 ? "" : std::to_string(vec_size);
  switch (data_type) {
    case DataType::FLOAT16:
      return "half" + postfix;
    case DataType::FLOAT32:
      return "float" + postfix;
    case DataType::FLOAT64:
      return "double" + postfix;
    case DataType::INT16:
      return "short" + postfix;
    case DataType::INT32:
      return "int" + postfix;
    case DataType::INT64:
      return "long" + postfix;
    case DataType::INT8:
      return "char" + postfix;
    case DataType::BFLOAT16:
      return "bfloat" + postfix;
    case DataType::UINT16:
      return "ushort" + postfix;
    case DataType::UINT32:
      return "uint" + postfix;
    case DataType::UINT64:
      return "ulong" + postfix;
    case DataType::UINT8:
      return "uchar" + postfix;
    case DataType::BOOL:
      return "bool" + postfix;
    case DataType::UNKNOWN:
    case DataType::UINT4:
    case DataType::INT4:
    case DataType::UINT3:
    case DataType::INT3:
    case DataType::UINT2:
    case DataType::INT2:
    case DataType::UINT1:
    case DataType::INT1:
      return "unknown";
  }
  return "undefined";
}

std::string ToString(PackedType type) {
  switch (type) {
    case PackedType::kUnknown:
      return "unknown";
    case PackedType::kUint8C4:
      return "uint8c4";
    case PackedType::kInt8C4:
      return "int8c4";
    case PackedType::kUint8W4C4:
      return "uint8w4c4";
    case PackedType::kInt8W4C4:
      return "int8w4c4";
    case PackedType::kInt8C16:
      return "int8c16";
    case PackedType::kUint8C16:
      return "uint8c16";
    case PackedType::kInt4C32:
      return "int4c32";
    case PackedType::kUint4C32:
      return "uint4c32";
  }
}

bool IsSigned(PackedType type) {
  switch (type) {
    case PackedType::kUnknown:
    case PackedType::kUint8C4:
    case PackedType::kUint8W4C4:
    case PackedType::kUint8C16:
    case PackedType::kUint4C32:
      return false;
    case PackedType::kInt8C4:
    case PackedType::kInt8W4C4:
    case PackedType::kInt8C16:
    case PackedType::kInt4C32:
      return true;
  }
}

size_t GetTypeSizeInBits(PackedType type) {
  switch (type) {
    case PackedType::kUnknown:
      return 0;
    case PackedType::kUint8C4:
    case PackedType::kUint8W4C4:
    case PackedType::kUint8C16:
    case PackedType::kInt8C4:
    case PackedType::kInt8W4C4:
    case PackedType::kInt8C16:
      return 8;
    case PackedType::kInt4C32:
    case PackedType::kUint4C32:
      return 4;
  }
}

DataType ToSpatialTensorType(PackedType type) {
  switch (type) {
    case PackedType::kUnknown:
      return DataType::UNKNOWN;
    case PackedType::kUint8C4:
      return DataType::UINT8;
    case PackedType::kUint8W4C4:
      return DataType::UINT32;
    case PackedType::kUint8C16:
      return DataType::UINT32;
    case PackedType::kInt8C4:
      return DataType::INT8;
    case PackedType::kInt8W4C4:
      return DataType::INT32;
    case PackedType::kInt8C16:
      return DataType::INT32;
    case PackedType::kInt4C32:
      return DataType::INT32;
    case PackedType::kUint4C32:
      return DataType::UINT32;
  }
}

}  // namespace ml_drift
