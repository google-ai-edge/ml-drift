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

#ifndef ML_DRIFT_WEBGPU_KERNELS_REGISRTY_H_
#define ML_DRIFT_WEBGPU_KERNELS_REGISRTY_H_

#include <set>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/kernels/reduce_parser.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/selectors/operation_selector.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
namespace webgpu {

class WebGpuOpSelector : public OpSelector {
 public:
  WebGpuOpSelector(const CreateGpuModelInfo& create_info,
                   const GpuInfo& gpu_info)
      : create_info_(create_info), gpu_info_(gpu_info) {}
  ~WebGpuOpSelector() override = default;
  absl::Status GPUOperationFromNode(const OperationDef& op_def,
                                    const std::vector<Value*>& inputs,
                                    const std::vector<Value*>& outputs,
                                    const Node& node,
                                    GpuModelBuilder* model_builder) override {
    return ml_drift::GPUOperationFromNode(gpu_info_, op_def, create_info_,
                                          inputs, outputs, node, model_builder);
  }

  absl::Status GPUSubgraphFromGraph(const GraphFloat32& graph,
                                    NodeId first_node_id,
                                    const std::set<NodeId>& consumed_nodes,
                                    std::set<NodeId>* new_consumed_nodes,
                                    GpuModelBuilder* model_builder) override {
    return TryAddThenReduce(gpu_info_, graph, first_node_id, consumed_nodes,
                            new_consumed_nodes, model_builder);
  }

 private:
  const CreateGpuModelInfo& create_info_;
  const GpuInfo& gpu_info_;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_KERNELS_REGISRTY_H_
