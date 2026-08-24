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

#ifndef ML_DRIFT_CL_MEMORY_MANAGER_H_
#define ML_DRIFT_CL_MEMORY_MANAGER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {

class MemoryManager {
 public:
  using ModelId = uint32_t;
  using Key = std::pair<ModelId, ValueId>;

  absl::StatusOr<ModelId> AllocateMemory(
      const GpuModel& gpu_model, const GpuInfo& gpu_info,
      const ExternalTensorsInfo& external_tensors,
      std::vector<std::unique_ptr<Tensor>>& external_mutable_tensors,
      CLContext* context);

  Tensor* GetTensor(Key key);
  absl::Status SetExternalTensor(const Key& key, Tensor* tensor_ptr);

  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;
  uint64_t GetConstantTensorsSize() const;
  uint64_t GetExternalTensorsSize() const;

 private:
  enum class TensorMemoryType { kStrongShape, kBuffer, kConst, kExternal };

  absl::StatusOr<std::vector<std::unique_ptr<Tensor>>> AllocateExternalTensors(
      ModelId model_id, const CLContext& context, const GpuModel& gpu_model,
      const ExternalTensorsInfo& external_tensors);

  absl::Status AllocateConstTensors(ModelId model_id, const GpuModel& gpu_model,
                                    CLContext* context);

  absl::Status AllocateBufferBasedTensors(
      ModelId model_id, const std::map<ValueId, int2>& buffer_usages,
      const GpuModel& gpu_model, const GpuInfo& gpu_info, CLContext* context);

  absl::Status AllocateTextureBasedTensors(
      ModelId model_id, const std::map<ValueId, int2>& texture_usages,
      const GpuModel& gpu_model, const GpuInfo& gpu_info, CLContext* context);

  absl::flat_hash_map<Key, Tensor*, absl::Hash<Key>, std::equal_to<Key>>
      external_immutable_tensors_;
  absl::flat_hash_map<Key, Tensor*, absl::Hash<Key>, std::equal_to<Key>>
      external_mutable_tensors_;

  absl::flat_hash_map<Key, TensorDescriptor, absl::Hash<Key>,
                      std::equal_to<Key>>
      tensors_descs_;
  std::map<Key, Tensor> const_tensors_;

  std::vector<std::unique_ptr<Buffer>> shared_buffers_;
  std::vector<std::unique_ptr<Buffer>> sub_buffers_;
  absl::flat_hash_map<Key, Tensor, absl::Hash<Key>, std::equal_to<Key>>
      value_id_to_buffer_;

  std::vector<std::unique_ptr<Tensor>> shared_texture_tensors_;
  absl::flat_hash_map<Key, Tensor, absl::Hash<Key>, std::equal_to<Key>>
      value_id_to_texture_;

  ModelId next_model_id_ = 0;
};

absl::Status GetTotalBufferSizeForTensors(
    const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors,
    const GpuInfo& gpu_info, uint64_t* result);

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_MEMORY_MANAGER_H_
