// Copyright 2024 The ML Drift Authors.
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
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {

class MemoryManager {
 public:
  absl::StatusOr<std::vector<std::unique_ptr<Tensor>>> AllocateMemory(
      const GpuModel& gpu_model, const GpuInfo& gpu_info,
      const ExternalTensorsInfo& external_tensors, CLContext* context);

  Tensor* GetTensor(ValueId id);
  absl::Status SetTensor(const ValueId& tensor_id, Tensor* tensor_ptr);

  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;
  uint64_t GetConstantTensorsSize() const;
  uint64_t GetExternalTensorsSize() const;

  std::unique_ptr<Buffer> shared_buffers_parent_;
  Buffer* shared_buffers_parent_ptr_ = nullptr;

 private:
  absl::StatusOr<std::vector<std::unique_ptr<Tensor>>> AllocateExternalTensors(
      const CLContext& context, const GpuModel& gpu_model,
      const ExternalTensorsInfo& external_tensors);

  absl::Status AllocateConstTensors(const GpuModel& gpu_model,
                                    CLContext* context);

  absl::Status AllocateBufferBasedTensors(
      const std::map<ValueId, int2>& buffer_usages, const GpuModel& gpu_model,
      const GpuInfo& gpu_info, CLContext* context);

  absl::Status AllocateTextureBasedTensors(
      const std::map<ValueId, int2>& texture_usages, const GpuModel& gpu_model,
      const GpuInfo& gpu_info, CLContext* context);

  absl::flat_hash_map<ValueId, Tensor*> external_immutable_tensors_;
  absl::flat_hash_map<ValueId, Tensor*> external_mutable_tensors_;

  std::map<ValueId, Tensor> const_tensors_;

  std::vector<Buffer> shared_buffers_;
  absl::flat_hash_map<ValueId, Tensor> value_id_to_buffer_;

  std::vector<Tensor> shared_texture_tensors_;
  absl::flat_hash_map<ValueId, Tensor> value_id_to_texture_;
};

absl::Status GetTotalBufferSizeForTensors(
    const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors,
    const GpuInfo& gpu_info, uint64_t* result);

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_MEMORY_MANAGER_H_
