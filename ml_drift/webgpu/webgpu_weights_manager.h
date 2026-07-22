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

#ifndef ML_DRIFT_WEBGPU_WEBGPU_WEIGHTS_MANAGER_H_
#define ML_DRIFT_WEBGPU_WEBGPU_WEIGHTS_MANAGER_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/webgpu/environment.h"

namespace ml_drift {
namespace webgpu {

class WebGpuWeightsManager : public WeightsManager {
 public:
  WebGpuWeightsManager() = default;
  ~WebGpuWeightsManager() = default;

  // Computes the batches of operations required to prepare weights.
  // Groups operations into batches to respect GPU limits and optimize execution
  // according to the specified schedule strategy.
  absl::StatusOr<std::vector<std::vector<WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(const Environment& env,
                                  const ScheduleStrategy schedule_strategy,
                                  size_t total_shared_tensor_size);

  // Prepares weights for a single batch of operations.
  // Executes the operations specified in op_infos and returns a map of
  // prepared tensors.
  absl::StatusOr<
      absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
  PrepareWeightsInBatch(const Environment& env,
                        std::vector<WeightsPrepOperationInfo>& op_infos);

  // Prepares all weights by grouping operations into batches and executing
  // them. Uses GetBatchesForWeightsPreparation to determine batches and calls
  // PrepareWeightsInBatch for each.
  absl::StatusOr<
      absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
  PrepareWeightsInBatches(const Environment& env,
                          ScheduleStrategy schedule_strategy,
                          size_t total_shared_tensor_size);
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_WEBGPU_WEIGHTS_MANAGER_H_
