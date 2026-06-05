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

#ifndef ML_DRIFT_COMMON_KERNELS_MEAN_STDDEV_NORMALIZATION_PARSER_H_
#define ML_DRIFT_COMMON_KERNELS_MEAN_STDDEV_NORMALIZATION_PARSER_H_

#include <set>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
// Tries to fuse a MeanStdDevNormalization subgraph.
// MeanStdDevNormalization fusion works with this patterns
//       input
//       /    \
//      |    mean
//       \    /
//     subtraction
//       /    \
//      |      |
//      |    square
//      |      |
//      |     mean
//      |      |
//      |     add
//      |      |
//      |    rsqrt
//      |      |
//       \    /
//    multiplication
//          |
//        output
//
// or
//
//        input_tensor
//       /     |      \
//    square  mean     |
//     |      /   \   /
//    mean square  sub
//      \   /       |
//       sub        |
//        |         |
//       add        |
//        |         |
//      rsqrt       |
//        |         |
//     mul_ones     |
//        \        /
//      multiplication
//            |
//          output
absl::Status TryMeanStdDevNormalization(const GpuInfo& gpu_info,
                                        const GraphFloat32& graph,
                                        NodeId first_node_id,
                                        const std::set<NodeId>& consumed_nodes,
                                        std::set<NodeId>* new_consumed_nodes,
                                        GpuModelBuilder* model_builder);

// Tries to fuse a LayerNormalization subgraph.
// LayerNormalization fusion works with this subgraph
//    input_tensor
//      /  /   \
//     /  |    mean0
//    /    \   /   \
//   |    sq_diff   |
//   |       |      |
//   |     mean1    |
//   |       |      |
//   |     add0     |
//   |       |      |
//   |     rsqrt    |
//   |       |      |
//    \    mul0    /
//     \   /   \  /
//      mul1    mul2
//       |       |
//        \     sub
//         \   /
//          add1
//           |
//     output_tensor
absl::Status TryLayerNormalization(const GpuInfo& gpu_info,
                                   const GraphFloat32& graph,
                                   NodeId first_node_id,
                                   const std::set<NodeId>& consumed_nodes,
                                   std::set<NodeId>* new_consumed_nodes,
                                   GpuModelBuilder* model_builder);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_MEAN_STDDEV_NORMALIZATION_PARSER_H_
