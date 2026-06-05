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

#ifndef ML_DRIFT_COMMON_TRANSFORMATIONS_MERGE_PADDING_WITH_H_
#define ML_DRIFT_COMMON_TRANSFORMATIONS_MERGE_PADDING_WITH_H_

#include <memory>

#include "ml_drift/common/model_transformer.h"

namespace ml_drift {

// Merges a Pad node with a following Pooling node.
std::unique_ptr<SequenceTransformation> NewMergePaddingWithPooling();

// Merges a Pad node with a following Convolution2D node.
std::unique_ptr<SequenceTransformation> NewMergePaddingWithConvolution2D();

// Merges a Pad node with a following DepthwiseConvolution2D node.
std::unique_ptr<SequenceTransformation>
NewMergePaddingWithDepthwiseConvolution();

// This transform requires Add operation support of unequal tensors on input.
// Padding should be with zeroes, and only appended in Z axis.
// Also input tensor channels should be divisible by 4(aligned).
// It should replace following pattern:
// 1) some tensor padded with zeroes in Z dim, for example from 24 to 32
//   channels
// 2) than this tensor used only in Add operation and Add operation
//   adds this useless zeroes on 24-32 channels.
// It removes this useless addition
// by using Add with unequal tensors on input. Instead of filling with zeroes
// and adding this part in Add operation, Add operation makes additional check
// for this tensor:
//   if (channels < src_channels) {
//     result += tensor_from_pad_operation.data[index];
//   }
std::unique_ptr<NodeTransformation> NewMergePaddingWithAdd();

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TRANSFORMATIONS_MERGE_PADDING_WITH_H_
