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

#include "ml_drift/common/transformations/model_transformations.h"

#include <memory>

#include "ml_drift/common/custom_transformations.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_transformer.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/transformations/add_quant_adjustments.h"
#include "ml_drift/common/transformations/fuse_add_to_conv.h"
#include "ml_drift/common/transformations/fuse_mul_to_conv.h"
#include "ml_drift/common/transformations/global_pooling_to_reduce_op.h"
#include "ml_drift/common/transformations/make_fully_connected.h"
#include "ml_drift/common/transformations/make_padding.h"
#include "ml_drift/common/transformations/merge_padding_with.h"
#include "ml_drift/common/transformations/remove_noop.h"

namespace ml_drift {
namespace {

bool ApplyGeneralTransformations(ModelTransformer* transformer) {
  // whenever any of these transforms return false, that means that a graph
  // is in the broken state and processing should not continue.
  return transformer->Apply("add_quant_adjustments",
                            NewAddQuantAdjustments().get()) &&
         transformer->Apply("remove_degenerate_upsampling",
                            NewRemoveDegenerateUpsampling().get()) &&
         transformer->Apply("remove_single_input_add",
                            NewRemoveSingleInputAdd().get()) &&
         transformer->Apply("remove_single_input_concat",
                            NewRemoveSingleInputConcat().get()) &&
         transformer->Apply("remove_identity_reshape",
                            NewRemoveIdentityReshape().get()) &&
         transformer->Apply("remove_identity_strided_slice",
                            NewRemoveIdentityStridedSlice().get()) &&
         transformer->Apply("make_padding_from_concat",
                            NewMakePaddingFromConcat().get()) &&
         transformer->Apply("make_fully_connected_from_convolution",
                            NewMakeFullyConnectedFromConvolution().get()) &&
         transformer->Apply("merge_padding_with_convolution",
                            NewMergePaddingWithConvolution2D().get()) &&
         transformer->Apply("merge_padding_with_pooling",
                            NewMergePaddingWithPooling().get()) &&
         transformer->Apply("merge_padding_with_depthwise_convolution",
                            NewMergePaddingWithDepthwiseConvolution().get()) &&
         transformer->Apply("merge_convolution_with_mul",
                            NewMergeConvolutionWithMul().get()) &&
         transformer->Apply("merge_convolution_with_add",
                            NewMergeConvolutionWithAdd().get()) &&
         transformer->Apply("merge_add_with_convolution",
                            NewMergeAddWithConvolution().get()) &&
         transformer->Apply("merge_mul_with_convolution",
                            NewMergeMulWithConvolution().get()) &&
         transformer->Apply("global_pooling_to_reduce_op",
                            NewGlobalPoolingToReduceOp().get());
}

bool ApplyGpuSpecificTransformations(ModelTransformer* transformer) {
  // whenever any of these transforms return false, that means that a graph
  // is in the broken state and processing should not continue.
  // NewMergePaddingWithAdd requires support of Add op with unequal shapes.
  return transformer->Apply("merge_padding_with_add",
                            NewMergePaddingWithAdd().get());
}

}  // namespace

absl::Status ApplyGpuModelTransformations(GraphFloat32* graph) {
  ModelTransformer transformer(graph);
  if (!ApplyCustomTransformations(&transformer)) {
    return absl::InternalError("Graph custom transformations failed");
  }
  if (!ApplyGeneralTransformations(&transformer)) {
    return absl::InternalError("Graph general transformations failed");
  }
  if (!ApplyGpuSpecificTransformations(&transformer)) {
    return absl::InternalError("Graph GPU specific transformations failed");
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
