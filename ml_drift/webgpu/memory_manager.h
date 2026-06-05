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

#ifndef ML_DRIFT_WEBGPU_MEMORY_MANAGER_H_
#define ML_DRIFT_WEBGPU_MEMORY_MANAGER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/spatial_tensor.h"

namespace ml_drift {
namespace webgpu {

class MemoryManager {
 public:
  using ModelId = uint32_t;
  using Key = std::pair<ModelId, ValueId>;

  MemoryManager() = default;
  ~MemoryManager() = default;

  absl::StatusOr<ModelId> AllocateMemory(
      const Environment& environment, const GpuModel& gpu_model,
      const ExternalTensorsInfo& external_tensors,
      std::vector<std::unique_ptr<SpatialTensor>>& external_mutable_tensors);
  SpatialTensor* GetSpatialTensor(Key key);
  absl::Status SetExternalTensor(Key key, SpatialTensor* tensor);
  const TensorDescriptor& GetTensorDescriptor(Key key);
  // for profiling and memory statistics
  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;
  uint64_t GetSizeOfMemoryAllocatedForConstTensors() const;

 private:
  enum class TensorMemoryType {
    kExternal,
    kBuffer,
    kTexture,
    kConst,
  };
  TensorMemoryType GetTensorMemoryType(Key key);
  absl::Status CollectUsageInformation(
      const GpuModel& gpu_model,
      std::map<ValueId, int2>* buffer_usages,
      std::map<ValueId, int2>* texture_usages);
  absl::Status AllocateMemoryForConstTensors(const Environment& environment,
                                             const GpuModel& gpu_model);
  absl::Status AllocateMemoryForBuffers(
      const Environment& environment,
      const std::map<ValueId, int2>& buffer_usages);
  absl::Status AllocateMemoryForTextures(
      const Environment& environment,
      const std::map<ValueId, int2>& texture_usages);

  absl::flat_hash_map<Key, TensorDescriptor> tensor_descriptors_;
  absl::flat_hash_map<Key, std::unique_ptr<SpatialTensor>> const_tensors_;
  std::vector<std::unique_ptr<Buffer>> shared_buffers_;
  absl::flat_hash_map<Key, std::unique_ptr<SpatialTensor>> value_id_to_buffer_;
  std::vector<std::unique_ptr<SpatialTensor>> shared_texture_tensors_;
  absl::flat_hash_map<Key, std::unique_ptr<SpatialTensor>> value_id_to_texture_;
  absl::flat_hash_map<Key, SpatialTensor*> external_immutable_tensors_;
  absl::flat_hash_map<Key, SpatialTensor*> external_mutable_tensors_;
  ModelId next_model_id_ = 0;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_MEMORY_MANAGER_H_
