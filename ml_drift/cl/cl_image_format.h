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

#ifndef ML_DRIFT_CL_CL_IMAGE_FORMAT_H_
#define ML_DRIFT_CL_CL_IMAGE_FORMAT_H_

#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/data_type.h"

namespace ml_drift {
namespace cl {

cl_channel_order ToChannelOrder(int num_channels);

cl_channel_type DataTypeToChannelType(DataType type, bool normalized = false);

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_IMAGE_FORMAT_H_
