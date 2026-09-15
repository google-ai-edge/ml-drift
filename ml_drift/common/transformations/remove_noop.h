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

#ifndef ML_DRIFT_COMMON_TRANSFORMATIONS_REMOVE_NOOP_H_
#define ML_DRIFT_COMMON_TRANSFORMATIONS_REMOVE_NOOP_H_

#include <memory>

#include "ml_drift/common/model_transformer.h"

namespace ml_drift {

// Removes a Concat node if it only has a single input.
std::unique_ptr<SequenceTransformation> NewRemoveSingleInputConcat();

// Removes an Add node if it only has a single input.
std::unique_ptr<SequenceTransformation> NewRemoveSingleInputAdd();

// Removes an Upsampling node if the input and output shapes are the same.
std::unique_ptr<SequenceTransformation> NewRemoveDegenerateUpsampling();

// Removes a Reshape node if the input and output shapes are the same.
std::unique_ptr<NodeTransformation> NewRemoveIdentityReshape();

// Merges two consecutive Reshape nodes into a single Reshape node.
std::unique_ptr<SequenceTransformation> NewMergeConsecutiveReshapes();

// Removes a StridedSlice node if it does not change the input tensor.
std::unique_ptr<NodeTransformation> NewRemoveIdentityStridedSlice();

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TRANSFORMATIONS_REMOVE_NOOP_H_
