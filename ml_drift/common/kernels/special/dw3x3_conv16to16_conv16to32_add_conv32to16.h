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

#ifndef ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV16TO16_CONV16TO32_ADD_CONV32TO16_H_
#define ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV16TO16_CONV16TO32_ADD_CONV32TO16_H_

#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

// This operation can replace few sequences that slightly different
// Input: 2 tensors
//   src_0 - 16 channels
//   src_1 - 32 or 8 channels
// Output:
//   dst_0 - 32 channels
//   dst_1 - 16 channels
// This operation replace following sequence of operations:
//   - Depthwise with kernel 3x3 (src_0 -> interm_0)
//   - Convolution 1x1 16 to 16 channels (+PReLU) (interm_0 -> interm_1)
//   - Convolution 1x1 16 to 32 channels (interm_1 -> interm_2)
//   - if (src_1 cnannels == 8) Pad src_1 with zeroes to 32 channels
//   - Add (+PReLU) (interm_2, src_1 -> dst_0)
//   - Convolution 1x1 32 to 16 channels (+PReLU) (dst_0 -> dst_1)
GPUOperation CreateDW3x3Conv16To16Conv16To32AddConv32To16(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv_0, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv_1, const PReLUAttributes& prelu1,
    const Convolution2DAttributes& conv_2, const PReLUAttributes& prelu2,
    bool full);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV16TO16_CONV16TO32_ADD_CONV32TO16_H_
