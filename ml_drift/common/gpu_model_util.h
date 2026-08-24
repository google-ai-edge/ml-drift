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

#ifndef ML_DRIFT_COMMON_GPU_MODEL_UTIL_H_
#define ML_DRIFT_COMMON_GPU_MODEL_UTIL_H_

#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
class OpSelector {
 public:
  OpSelector() = default;
  virtual ~OpSelector() = default;
  virtual absl::Status GPUOperationFromNode(const OperationDef& op_def,
                                            const std::vector<Value*>& inputs,
                                            const std::vector<Value*>& outputs,
                                            const Node& node,
                                            GpuModelBuilder* model_builder) = 0;
  virtual absl::Status GPUSubgraphFromGraph(
      const GraphFloat32& graph, NodeId first_node_id,
      const std::set<NodeId>& consumed_nodes,
      std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
    return absl::UnimplementedError("Not implemented GPUSubgraphFromGraph.");
  }
};

absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, GpuModel* gpu_model);

// Keep GraphToGpuModel with OpSelector parameter separately to exclude adding
// DefaultOpSelector to binary.
absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, GpuModel* gpu_model,
                             OpSelector* op_selector);

absl::Status GraphToGpuModelWithWeightsConversion(
    const GraphFloat32& graph, const CreateGpuModelInfo& create_info,
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    GpuModel* gpu_weights_conversion_model,
    absl::flat_hash_map<ValueId, ValueId>* weights_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_info);

absl::Status GraphToGpuModelWithWeightsConversion(
    const GraphFloat32& graph, const CreateGpuModelInfo& create_info,
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    GpuModel* gpu_weights_conversion_model,
    absl::flat_hash_map<ValueId, ValueId>* weights_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_info,
    OpSelector* op_selector);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_MODEL_UTIL_H_
