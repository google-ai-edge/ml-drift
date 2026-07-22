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

#ifndef ML_DRIFT_METAL_METAL_WEIGHTS_MANAGER_H_
#define ML_DRIFT_METAL_METAL_WEIGHTS_MANAGER_H_

#import <Metal/Metal.h>

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/metal/environment.h"

namespace ml_drift {
namespace metal {

class MetalWeightsManager : public WeightsManager {
 public:
  MetalWeightsManager() = default;
  ~MetalWeightsManager() = default;

  // Returns the batches for weights preparation.
  absl::StatusOr<
      std::vector<std::vector<WeightsManager::WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(Environment& env,
                                  const ScheduleStrategy schedule_strategy,
                                  size_t total_shared_tensor_size);

  // Prepares the weights in one batch gotten from
  // GetBatchesForWeightsPreparation.
  absl::StatusOr<
      absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
  PrepareWeightsInBatch(
      Environment& env,
      std::vector<WeightsManager::WeightsPrepOperationInfo>& op_infos);

  // Prepares the weights in batches for GPU backend. This is only being used
  // when GPU weights preparation is enabled. This effectively combines
  // GetBatchesForWeightsPreparation and PrepareWeightsInBatch into a single
  // function.
  absl::StatusOr<
      absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
  PrepareWeightsInBatches(Environment& env, ScheduleStrategy schedule_strategy,
                          size_t total_shared_tensor_size);
};

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_METAL_WEIGHTS_MANAGER_H_
