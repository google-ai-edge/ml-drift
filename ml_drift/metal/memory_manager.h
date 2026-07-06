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

#ifndef ML_DRIFT_METAL_MEMORY_MANAGER_H_
#define ML_DRIFT_METAL_MEMORY_MANAGER_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/metal/buffer.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {

class MemoryManager {
 public:
  using ModelId = uint32_t;
  using Key = std::pair<ModelId, ValueId>;

  absl::StatusOr<ModelId> AllocateMemory(
      const Environment& env, const GpuModel& gpu_model,
      const ExternalTensorsInfo& external_tensors,
      std::vector<std::unique_ptr<MetalSpatialTensor>>&
          external_mutable_tensors);
  MetalSpatialTensor* GetSpatialTensor(Key key);
  // Can be used only with ids from mutable_tensors in external_tensors
  // Must be called after initialization and before execution
  absl::Status SetExternalTensor(const Key& key,
                                 MetalSpatialTensor* tensor_ptr);
  uint64_t GetIntermediateTensorsSize() const;
  uint64_t GetConstantTensorsSize() const;

  // Adds allocated internal constant GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddConstantsToResidencySet(id<MTLResidencySet> residency_set) const;

  // Adds allocated intermediate GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddIntermediatesToResidencySet(id<MTLResidencySet> residency_set) const;

  // Adds allocated external immutable GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddExternalImmutableToResidencySet(
      id<MTLResidencySet> residency_set) const;

  // Adds allocated external mutable GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddExternalMutableToResidencySet(
      id<MTLResidencySet> residency_set) const;

 private:
  enum class TensorMemoryType {
    kStrongShape,
    kBuffer,
    kConst,
    kExternal
  };
  absl::Status CollectUsageInformation(ModelId model_id,
                                       const GpuModel& gpu_model,
                                       const GpuInfo& gpu_info,
                                       std::map<ValueId, int2>* buffer_usages,
                                       std::map<ValueId, int2>* texture_usages);
  absl::StatusOr<std::vector<std::unique_ptr<MetalSpatialTensor>>>
  AllocateExternalTensors(ModelId model_id,
                          const ExternalTensorsInfo& external_tensors,
                          const Environment& env);
  absl::Status AllocateMemoryForConstTensors(ModelId model_id,
                                             const GpuModel& gpu_model,
                                             const Environment& env);
  absl::Status AllocateMemoryForBuffers(
      ModelId model_id, const Environment& env,
      const std::map<ValueId, int2>& buffer_usages);
  absl::Status AllocateMemoryForTextures(
      ModelId model_id, const Environment& env,
      const std::map<ValueId, int2>& texture_usages);
  TensorMemoryType GetTensorMemoryType(const GpuInfo& gpu_info, Key key);

  absl::flat_hash_map<Key, MetalSpatialTensor*, absl::Hash<Key>,
                      std::equal_to<Key>>
      external_immutable_tensors_;
  absl::flat_hash_map<Key, MetalSpatialTensor*, absl::Hash<Key>,
                      std::equal_to<Key>>
      external_mutable_tensors_;

  absl::flat_hash_map<Key, TensorDescriptor, absl::Hash<Key>,
                      std::equal_to<Key>>
      tensors_descs_;
  std::map<Key, MetalSpatialTensor> const_tensors_;

  std::vector<std::unique_ptr<Buffer>> shared_buffers_;
  absl::flat_hash_map<Key, MetalSpatialTensor, absl::Hash<Key>,
                      std::equal_to<Key>>
      value_id_to_buffer_;

  std::vector<std::unique_ptr<MetalSpatialTensor>> shared_texture_tensors_;
  absl::flat_hash_map<Key, MetalSpatialTensor, absl::Hash<Key>,
                      std::equal_to<Key>>
      value_id_to_texture_;
  ModelId next_model_id_ = 0;
};

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_MEMORY_MANAGER_H_
