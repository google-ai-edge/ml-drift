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

#ifndef ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV32TO16_DW3X3_CONV16TO16_ADD_CONV16TO32_H_
#define ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV32TO16_DW3X3_CONV16TO16_ADD_CONV16TO32_H_

#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

// Input:
//   src_0 - 32 channels
//   src_1 - 16 channels
// Output:
//   dst_0 - 32 channels
// This operation replace following sequence of operations:
//   - Depthwise with kernel 3x3 (src_0 -> interm_0)
//   - Convolution 1x1 32 to 16 channels (interm_0 -> interm_1)
//   - Depthwise with kernel 3x3 (src_1 -> interm_2)
//   - Convolution 1x1 16 to 16 channels (interm_2 -> interm_3)
//   - Add (+PReLU) (interm_1, input_3 -> input_4)
//   - Convolution 1x1 16 to 32 channels (+PReLU) (input_4 -> dst_0)
GPUOperation CreateDW3x3Conv32To16DW3x3Conv16To16AddConv16To32(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr_0,
    const Convolution2DAttributes& conv32to16,
    const DepthwiseConvolution2DAttributes& dw_attr_1,
    const Convolution2DAttributes& conv16to16, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv16to32, const PReLUAttributes& prelu1);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV32TO16_DW3X3_CONV16TO16_ADD_CONV16TO32_H_
