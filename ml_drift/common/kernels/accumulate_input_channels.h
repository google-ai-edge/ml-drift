// Copyright 2025 The ML Drift Authors.
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

#ifndef ML_DRIFT_COMMON_KERNELS_ACCUMULATE_INPUT_CHANNELS_H_
#define ML_DRIFT_COMMON_KERNELS_ACCUMULATE_INPUT_CHANNELS_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

// This GPU operation performs a reduction sum over the input channels of the
// input tensor. The input tensor is expected to be in the layout of raw OHWI,
// which is the layout of weights stored in TFLite FlatBuffer.
//
// @param definition The operation definition, which is supposed to contain one
//   src tensor and one dst tensor:
//   * Src tensor's BHWC shape is expected to be (1, 1, 1, input_channels *
//     output_channels). It merges the input channels and output channels into
//     the C-dimension, because the src tensor is expected to be in the layout
//     of raw OHWI.
//   * Dst tensor's BHWC shape is expected to be (1, 1, 1, output_channels).
// @param src_shape The shape of the input tensor, which is supposed to be
//   OHWI(output_channels, 1, 1, input_channels).
// @return The GPU operation.
//
GPUOperation CreateAccumulateInputChannels(const OperationDef& definition,
                                           const OHWI& input_shape,
                                           const DataType& input_data_type);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_ACCUMULATE_INPUT_CHANNELS_H_
