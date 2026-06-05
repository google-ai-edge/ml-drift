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

#ifndef ML_DRIFT_COMMON_KERNELS_SPECIAL_CONV2X2_MAX_POOL2X2_H_
#define ML_DRIFT_COMMON_KERNELS_SPECIAL_CONV2X2_MAX_POOL2X2_H_

#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

// Input: 1 tensor with channels = 8.
// Output: 2 tensors
//   1 (conv output) - 16 channels, 2x smaller in XY than input
//   2 (pooling output)- 8 channels, 2x smaller in XY than input
// This operation replace following sequence of operations:
//   - Convolution 2x2 8 to 16 channels (+PReLU) (input -> output_0)
//   - MaxPooling 2x2 8 to 8 channels (input -> output_1)
GPUOperation CreateConv2x2MaxPool2x2(const OperationDef& definition,
                                     const Convolution2DAttributes& conv_attr,
                                     const PReLUAttributes& prelu_attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPECIAL_CONV2X2_MAX_POOL2X2_H_
