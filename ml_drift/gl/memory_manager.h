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

#ifndef ML_DRIFT_GL_MEMORY_MANAGER_H_
#define ML_DRIFT_GL_MEMORY_MANAGER_H_

#include <cstdint>
#include <map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_spatial_tensor.h"

namespace ml_drift {
namespace gl {

class MemoryManager {
 public:
  absl::Status AllocateMemory(const GpuModel& gpu_model,
                              const ExternalTensorsInfo& external_tensors = {});

  GlSpatialTensor* GetTensor(ValueId id);

  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;

 private:
  enum class TensorMemoryType { kStrongShape, kBuffer, kConst, kExternal };
  absl::Status AllocateMemoryForConstTensors(const GpuModel& gpu_model);

  absl::Status AllocateMemoryForBuffers(
      const std::map<ValueId, int2>& buffer_usages);

  absl::Status AllocateMemoryForTextures(
      const std::map<ValueId, int2>& texture_usages);

  absl::Status CollectUsageInformation(
      const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors,
      std::map<ValueId, int2>* buffer_usages,
      std::map<ValueId, int2>* texture_usages);

  TensorMemoryType GetTensorMemoryType(
      ValueId id, const ExternalTensorsInfo& external_tensors);

  absl::flat_hash_map<ValueId, TensorDescriptor> tensors_descs_;

  absl::flat_hash_map<ValueId, GlSpatialTensor*> external_immutable_tensors_;

  std::map<ValueId, GlSpatialTensor> const_tensors_;

  std::vector<GlBuffer> shared_buffers_;
  absl::flat_hash_map<ValueId, GlSpatialTensor> value_id_to_buffer_;

  std::vector<GlSpatialTensor> shared_texture_tensors_;
  absl::flat_hash_map<ValueId, GlSpatialTensor> value_id_to_texture_;
};

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_MEMORY_MANAGER_H_
