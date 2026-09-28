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

#include "ml_drift/gl/gl_texture_helper.h"

#include "ml_drift/common/data_type.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {

GLenum ToTextureFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::kInt8:
    case DataType::kUint8:
      return normalized ? GL_RGBA : GL_RGBA_INTEGER;
    case DataType::kBool:
      return GL_RGBA_INTEGER;
    case DataType::kBfloat16:
    case DataType::kUint16:
    case DataType::kUint32:
    case DataType::kInt16:
    case DataType::kInt32:
      return GL_RGBA_INTEGER;
    case DataType::kFloat16:
    case DataType::kFloat32:
      return GL_RGBA;
    default:
      return 0;
  }
}

GLenum ToTextureInternalFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::kUint8:
      return normalized ? GL_RGBA8 : GL_RGBA8UI;
    case DataType::kBool:
      return GL_RGBA8UI;
    case DataType::kInt8:
      return normalized ? GL_RGBA8_SNORM : GL_RGBA8I;
    case DataType::kBfloat16:
    case DataType::kUint16:
      return GL_RGBA16UI;
    case DataType::kUint32:
      return GL_RGBA32UI;
    case DataType::kInt16:
      return GL_RGBA16I;
    case DataType::kInt32:
      return GL_RGBA32I;
    case DataType::kFloat16:
      return GL_RGBA16F;
    case DataType::kFloat32:
      return GL_RGBA32F;
    default:
      return 0;
  }
}

GLenum ToTextureDataType(DataType type) {
  switch (type) {
    case DataType::kUint8:
      return GL_UNSIGNED_BYTE;
    case DataType::kBool:
      return GL_UNSIGNED_BYTE;
    case DataType::kInt8:
      return GL_BYTE;
    case DataType::kBfloat16:
    case DataType::kUint16:
      return GL_UNSIGNED_SHORT;
    case DataType::kUint32:
      return GL_UNSIGNED_INT;
    case DataType::kInt16:
      return GL_SHORT;
    case DataType::kInt32:
      return GL_INT;
    case DataType::kFloat16:
      return GL_HALF_FLOAT;
    case DataType::kFloat32:
      return GL_FLOAT;
    default:
      return 0;
  }
}

}  // namespace gl
}  // namespace ml_drift
