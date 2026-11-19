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

#include "ml_drift/cl/tensor_type_util.h"

#include "ml_drift/common/api_cl_gl_vk.h"
#include "ml_drift/common/api_common.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace cl {

ObjectType ToObjectType(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::kImageBuffer:
    case TensorStorageType::kBuffer:
      return ObjectType::kOpenClBuffer;
    case TensorStorageType::kSingleTexture2D:
    case TensorStorageType::kTexture2D:
    case TensorStorageType::kTextureArray:
    case TensorStorageType::kTexture3D:
      return ObjectType::kOpenClTexture;
    default:
      return ObjectType::kUnknown;
  }
}

DataLayout ToDataLayout(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::kBuffer:
      return DataLayout::kDHWC4;
    case TensorStorageType::kImageBuffer:
      return DataLayout::kDHWC4;
    case TensorStorageType::kSingleTexture2D:
      return DataLayout::kBHWC;
    case TensorStorageType::kTexture2D:
      return DataLayout::kHDWC4;
    case TensorStorageType::kTextureArray:
      return DataLayout::kDHWC4;
    case TensorStorageType::kTexture3D:
      return DataLayout::kDHWC4;
    default:
      return DataLayout::kUnknown;
  }
}

TensorStorageType ToTensorStorageType(ObjectType object_type,
                                      DataLayout data_layout) {
  switch (object_type) {
    case ObjectType::kOpenClBuffer:
      return TensorStorageType::kBuffer;
    case ObjectType::kOpenClTexture:
      switch (data_layout) {
        case DataLayout::kBHWC:
          return TensorStorageType::kSingleTexture2D;
        case DataLayout::kDHWC4:
          return TensorStorageType::kTextureArray;
        case DataLayout::kHDWC4:
          return TensorStorageType::kTexture2D;
        default:
          return TensorStorageType::kUnknown;
      }
    default:
      return TensorStorageType::kUnknown;
  }
}

}  // namespace cl
}  // namespace ml_drift
