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

#include "ml_drift/common/selectors/special_selector.h"

#include <set>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/mean_stddev_normalization_parser.h"
#include "ml_drift/common/kernels/mish.h"
#include "ml_drift/common/kernels/reduce_parser.h"
#include "ml_drift/common/kernels/special/concat_conv.h"
#include "ml_drift/common/kernels/special/conv_pointwise.h"
#include "ml_drift/common/kernels/special/dw7x7_conv2to6_concat_conv8to8.h"
#include "ml_drift/common/kernels/special/thin_local_memory_fuser.h"
#include "ml_drift/common/kernels/special/thin_pointwise_fuser.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
absl::Status GPUSubgraphFromGraph(
    const ModelHints& hints, const GpuInfo& gpu_info, const GraphFloat32& graph,
    NodeId first_node_id, const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryDW7x7Conv2To6ConcatConv8to8(gpu_info, graph, first_node_id,
                                     consumed_nodes, new_consumed_nodes,
                                     model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryThinPointwiseFuser(gpu_info, graph, first_node_id, consumed_nodes,
                            new_consumed_nodes, model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryThinLocalMemoryFuser(gpu_info, graph, first_node_id, consumed_nodes,
                              new_consumed_nodes, model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  if (TryFusedPointwiseConv(graph, first_node_id, consumed_nodes,
                            new_consumed_nodes, model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  if (TryMeanStdDevNormalization(gpu_info, graph, first_node_id, consumed_nodes,
                                 new_consumed_nodes, model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  if (TryMish(gpu_info, graph, first_node_id, consumed_nodes,
              new_consumed_nodes, model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  if (TryAddThenReduce(gpu_info, graph, first_node_id, consumed_nodes,
                       new_consumed_nodes, model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryConcatConv(gpu_info, graph, first_node_id, consumed_nodes,
                    new_consumed_nodes, model_builder)
          .ok()) {
    return absl::OkStatus();
  }
  return absl::NotFoundError("No special combination.");
}

}  // namespace ml_drift
