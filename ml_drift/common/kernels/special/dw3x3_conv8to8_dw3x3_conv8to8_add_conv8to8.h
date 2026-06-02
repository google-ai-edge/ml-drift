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

#ifndef ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV8TO8_DW3X3_CONV8TO8_ADD_CONV8TO8_H_
#define ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV8TO8_DW3X3_CONV8TO8_ADD_CONV8TO8_H_

#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

// Input: 2 equal tensors with channels = 8 both.
// Output: 1 tensor equals to input tensors
// This operation replace following sequence of operations:
//   - Depthwise with kernel 3x3 (input_0 -> interm_0)
//   - Convolution 1x1 8 to 8 channels (interm_0 -> interm_1)
//   - Depthwise with kernel 3x3 (input_1 -> interm_2)
//   - Convolution 1x1 8 to 8 channels (interm_2 -> interm_3)
//   - Add (+PReLU) (interm_1, input_3 -> input_4)
//   - Convolution 1x1 8 to 8 channels (+PReLU) (input_4 -> output_0)
GPUOperation CreateDW3x3Conv8To8DW3x3Conv8To8AddConv8To8(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr_0,
    const Convolution2DAttributes& conv8to8_0,
    const DepthwiseConvolution2DAttributes& dw_attr_1,
    const Convolution2DAttributes& conv8to8_1, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv8to8_2, const PReLUAttributes& prelu1);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPECIAL_DW3X3_CONV8TO8_DW3X3_CONV8TO8_ADD_CONV8TO8_H_
