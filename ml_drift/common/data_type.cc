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
    case DataType::kUint8:
    case DataType::kInt8:
      return "lowp ";
    case DataType::kFloat16:
    case DataType::kInt16:
    case DataType::kUint16:
      return "mediump ";
    case DataType::kFloat32:
    case DataType::kInt32:
    case DataType::kUint32:
      return "highp ";
    case DataType::kBool:
      return "";
    default:
      return "";
  }
}
}  // namespace

size_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::kUint8:
    case DataType::kInt8:
    case DataType::kBool:
    case DataType::kUint4:
    case DataType::kInt4:
    case DataType::kUint3:
    case DataType::kInt3:
    case DataType::kUint2:
    case DataType::kInt2:
    case DataType::kUint1:
    case DataType::kInt1:
      return 1;
    case DataType::kFloat16:
    case DataType::kInt16:
    case DataType::kUint16:
    case DataType::kBfloat16:
      return 2;
    case DataType::kFloat32:
    case DataType::kInt32:
    case DataType::kUint32:
      return 4;
    case DataType::kFloat64:
    case DataType::kInt64:
    case DataType::kUint64:
      return 8;
    case DataType::kUnknown:
      return 0;
  }
  return 0;
}

size_t SizeInBitsOf(DataType data_type) {
  switch (data_type) {
    case DataType::kUint8:
    case DataType::kInt8:
      return 8;
    case DataType::kBool:
      return 1;
    case DataType::kUint4:
    case DataType::kInt4:
      return 4;
    case DataType::kUint3:
    case DataType::kInt3:
      return 3;
    case DataType::kUint2:
    case DataType::kInt2:
      return 2;
    case DataType::kUint1:
    case DataType::kInt1:
      return 1;
    case DataType::kFloat16:
    case DataType::kInt16:
    case DataType::kUint16:
    case DataType::kBfloat16:
      return 16;
    case DataType::kFloat32:
    case DataType::kInt32:
    case DataType::kUint32:
      return 32;
    case DataType::kFloat64:
    case DataType::kInt64:
    case DataType::kUint64:
      return 64;
    case DataType::kUnknown:
      return 0;
  }
  return 0;
}

bool IsFloatType(DataType data_type) {
  switch (data_type) {
    case DataType::kFloat16:
    case DataType::kBfloat16:
    case DataType::kFloat32:
    case DataType::kFloat64:
      return true;
    case DataType::kUint8:
    case DataType::kInt8:
    case DataType::kBool:
    case DataType::kUint4:
    case DataType::kInt4:
    case DataType::kUint3:
    case DataType::kInt3:
    case DataType::kUint2:
    case DataType::kInt2:
    case DataType::kUint1:
    case DataType::kInt1:
    case DataType::kInt16:
    case DataType::kUint16:
    case DataType::kInt32:
    case DataType::kUint32:
    case DataType::kInt64:
    case DataType::kUint64:
    case DataType::kUnknown:
      return false;
  }
  return false;
}

bool IsSigned(DataType data_type) {
  switch (data_type) {
    case DataType::kUint64:
    case DataType::kUint32:
    case DataType::kUint16:
    case DataType::kUint8:
    case DataType::kUint4:
    case DataType::kUint3:
    case DataType::kUint2:
    case DataType::kUint1:
      return false;
    case DataType::kInt64:
    case DataType::kInt32:
    case DataType::kInt16:
    case DataType::kInt8:
    case DataType::kInt4:
    case DataType::kInt3:
    case DataType::kInt2:
    case DataType::kInt1:
      return true;
    case DataType::kFloat16:
    case DataType::kBfloat16:
    case DataType::kFloat32:
    case DataType::kFloat64:
      return false;
    case DataType::kBool:
    case DataType::kUnknown:
      return false;
  }
  return false;
}

std::string ToString(DataType data_type) {
  switch (data_type) {
    case DataType::kFloat16:
      return "float16";
    case DataType::kFloat32:
      return "float32";
    case DataType::kFloat64:
      return "float64";
    case DataType::kBfloat16:
      return "bfloat16";
    case DataType::kInt16:
      return "int16";
    case DataType::kInt32:
      return "int32";
    case DataType::kInt64:
      return "int64";
    case DataType::kInt8:
      return "int8";
    case DataType::kInt4:
      return "int4";
    case DataType::kInt3:
      return "int3";
    case DataType::kInt2:
      return "int2";
    case DataType::kInt1:
      return "int1";
    case DataType::kUint16:
      return "uint16";
    case DataType::kUint32:
      return "uint32";
    case DataType::kUint64:
      return "uint64";
    case DataType::kUint8:
      return "uint8";
    case DataType::kUint4:
      return "uint4";
    case DataType::kUint3:
      return "uint3";
    case DataType::kUint2:
      return "uint2";
    case DataType::kUint1:
      return "uint1";
    case DataType::kBool:
      return "bool";
    case DataType::kUnknown:
      return "unknown";
  }
  return "undefined";
}

std::string ToCLDataType(DataType data_type, int vec_size) {
  const std::string postfix = vec_size == 1 ? "" : std::to_string(vec_size);
  switch (data_type) {
    case DataType::kFloat16:
      return "half" + postfix;
    case DataType::kFloat32:
      return "float" + postfix;
    case DataType::kFloat64:
      return "double" + postfix;
    case DataType::kInt16:
      return "short" + postfix;
    case DataType::kInt32:
      return "int" + postfix;
    case DataType::kInt64:
      return "long" + postfix;
    case DataType::kInt8:
      return "char" + postfix;
    case DataType::kUint16:
      return "ushort" + postfix;
    case DataType::kUint32:
      return "uint" + postfix;
    case DataType::kUint64:
      return "ulong" + postfix;
    case DataType::kUint8:
      return "uchar" + postfix;
    case DataType::kBool:
      return "bool" + postfix;
    case DataType::kUnknown:
    case DataType::kBfloat16:
    case DataType::kUint4:
    case DataType::kInt4:
    case DataType::kUint3:
    case DataType::kInt3:
    case DataType::kUint2:
    case DataType::kInt2:
    case DataType::kUint1:
    case DataType::kInt1:
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
    case DataType::kFloat16:
      scalar = "f16";
      num_packed = 1;
      break;
    case DataType::kFloat32:
      scalar = "f32";
      num_packed = 1;
      break;
    case DataType::kInt32:
      scalar = "i32";
      num_packed = 1;
      break;
    case DataType::kInt8:
      scalar = "i32";
      num_packed = 4;
      break;
    case DataType::kUint32:
      scalar = "u32";
      num_packed = 1;
      break;
    case DataType::kUint8:
      scalar = "u32";
      num_packed = 4;
      break;
    case DataType::kBool:
      scalar = "bool";
      num_packed = 1;
      break;
    case DataType::kBfloat16:
    case DataType::kUint64:
    case DataType::kUint16:
    case DataType::kInt64:
    case DataType::kInt16:
    case DataType::kFloat64:
    case DataType::kUnknown:
    case DataType::kUint4:
    case DataType::kInt4:
    case DataType::kUint3:
    case DataType::kInt3:
    case DataType::kUint2:
    case DataType::kInt2:
    case DataType::kUint1:
    case DataType::kInt1:
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
    case DataType::kFloat16:
      return "half" + postfix;
    case DataType::kFloat32:
      return "float" + postfix;
    case DataType::kFloat64:
      return "double" + postfix;
    case DataType::kInt16:
      return "short" + postfix;
    case DataType::kInt32:
      return "int" + postfix;
    case DataType::kInt64:
      return "long" + postfix;
    case DataType::kInt8:
      return "char" + postfix;
    case DataType::kBfloat16:
      return "bfloat" + postfix;
    case DataType::kUint16:
      return "ushort" + postfix;
    case DataType::kUint32:
      return "uint" + postfix;
    case DataType::kUint64:
      return "ulong" + postfix;
    case DataType::kUint8:
      return "uchar" + postfix;
    case DataType::kBool:
      return "bool" + postfix;
    case DataType::kUnknown:
    case DataType::kUint4:
    case DataType::kInt4:
    case DataType::kUint3:
    case DataType::kInt3:
    case DataType::kUint2:
    case DataType::kInt2:
    case DataType::kUint1:
    case DataType::kInt1:
      return "unknown";
  }
  return "undefined";
}

DataType ToMetalTextureType(DataType data_type) {
  switch (data_type) {
    case DataType::kFloat32:
    case DataType::kFloat16:
    case DataType::kInt32:
    case DataType::kInt16:
    case DataType::kUint32:
    case DataType::kUint16:
      return data_type;
    case DataType::kInt8:
      return DataType::kInt16;
    case DataType::kBfloat16:
    case DataType::kUint8:
    case DataType::kBool:
      return DataType::kUint16;
    default:
      return DataType::kUnknown;
  }
}

std::string ToGlslShaderDataType(DataType data_type, int vec_size,
                                 bool add_precision, bool explicit_fp16) {
  const std::string precision_modifier =
      add_precision ? GetGlslPrecisionModifier(data_type) : "";
  switch (data_type) {
    case DataType::kFloat16:
      if (explicit_fp16) {
        return ToGlslType("float16_t", "f16vec", vec_size);
      } else {
        return precision_modifier + ToGlslType("float", "vec", vec_size);
      }
    case DataType::kFloat32:
      return precision_modifier + ToGlslType("float", "vec", vec_size);
    case DataType::kFloat64:
      return precision_modifier + ToGlslType("double", "dvec", vec_size);
    case DataType::kInt8:
    case DataType::kInt16:
    case DataType::kInt32:
    case DataType::kInt64:
      return precision_modifier + ToGlslType("int", "ivec", vec_size);
    case DataType::kUint8:
    case DataType::kUint16:
    case DataType::kUint32:
    case DataType::kUint64:
      return precision_modifier + ToGlslType("uint", "uvec", vec_size);
    case DataType::kBool:
      return ToGlslType("bool", "bvec", vec_size);
    case DataType::kUnknown:
    case DataType::kBfloat16:
    case DataType::kUint4:
    case DataType::kInt4:
    case DataType::kUint3:
    case DataType::kInt3:
    case DataType::kUint2:
    case DataType::kInt2:
    case DataType::kUint1:
    case DataType::kInt1:
      return "unknown";
  }
  return "unknown";
}

std::string ToWebGpuType(DataType data_type, int vec_size, bool explicit_fp16) {
  std::string type_name;
  if (data_type == DataType::kFloat32) {
    type_name = "f32";
  } else if (data_type == DataType::kFloat16) {
    type_name = explicit_fp16 ? "f16" : "f32";
  } else if (data_type == DataType::kInt32 || data_type == DataType::kInt16 ||
             data_type == DataType::kInt8) {
    type_name = "i32";
  } else if (data_type == DataType::kUint32 || data_type == DataType::kUint16 ||
             data_type == DataType::kUint8) {
    type_name = "u32";
  } else if (data_type == DataType::kBool) {
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
    case DataType::kFloat16:
      return "half" + postfix;
    case DataType::kFloat32:
      return "float" + postfix;
    case DataType::kFloat64:
      return "double" + postfix;
    case DataType::kInt16:
      return "short" + postfix;
    case DataType::kInt32:
      return "int" + postfix;
    case DataType::kInt64:
      return "long" + postfix;
    case DataType::kInt8:
      return "char" + postfix;
    case DataType::kBfloat16:
      return "bfloat" + postfix;
    case DataType::kUint16:
      return "ushort" + postfix;
    case DataType::kUint32:
      return "uint" + postfix;
    case DataType::kUint64:
      return "ulong" + postfix;
    case DataType::kUint8:
      return "uchar" + postfix;
    case DataType::kBool:
      return "bool" + postfix;
    case DataType::kUnknown:
    case DataType::kUint4:
    case DataType::kInt4:
    case DataType::kUint3:
    case DataType::kInt3:
    case DataType::kUint2:
    case DataType::kInt2:
    case DataType::kUint1:
    case DataType::kInt1:
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
      return DataType::kUnknown;
    case PackedType::kUint8C4:
      return DataType::kUint8;
    case PackedType::kUint8W4C4:
      return DataType::kUint32;
    case PackedType::kUint8C16:
      return DataType::kUint32;
    case PackedType::kInt8C4:
      return DataType::kInt8;
    case PackedType::kInt8W4C4:
      return DataType::kInt32;
    case PackedType::kInt8C16:
      return DataType::kInt32;
    case PackedType::kInt4C32:
      return DataType::kInt32;
    case PackedType::kUint4C32:
      return DataType::kUint32;
  }
}

}  // namespace ml_drift
