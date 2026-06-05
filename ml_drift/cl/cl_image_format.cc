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

#include "ml_drift/cl/cl_image_format.h"

#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/data_type.h"

namespace ml_drift {
namespace cl {

cl_channel_order ToChannelOrder(int num_channels) {
  switch (num_channels) {
    case 1:
      return CL_R;
    case 2:
      return CL_RG;
    case 3:
      return CL_RGB;
    case 4:
      return CL_RGBA;
    default:
      return -1;
  }
}

cl_channel_type DataTypeToChannelType(DataType type, bool normalized) {
  switch (type) {
    case DataType::FLOAT32:
      return CL_FLOAT;
    case DataType::FLOAT16:
      return CL_HALF_FLOAT;
    case DataType::INT8:
      return normalized ? CL_SNORM_INT8 : CL_SIGNED_INT8;
    case DataType::UINT8:
      return normalized ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
    case DataType::INT16:
      return normalized ? CL_SNORM_INT16 : CL_SIGNED_INT16;
    case DataType::BFLOAT16:
    case DataType::UINT16:
      return normalized ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
    case DataType::INT32:
      return CL_SIGNED_INT32;
    case DataType::UINT32:
      return CL_UNSIGNED_INT32;
    case DataType::BOOL:
      return CL_UNSIGNED_INT8;
    default:
      return CL_FLOAT;
  }
}

}  // namespace cl
}  // namespace ml_drift
