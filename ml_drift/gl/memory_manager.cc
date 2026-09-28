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

#include "ml_drift/gl/memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/memory_management.h"
#include "ml_drift/common/memory_management/types.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_spatial_tensor.h"

namespace ml_drift {
namespace gl {
namespace {
void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    (*usage_records)[id].y = task_index;
  }
}

// returns true if actual memory for this storage type will be allocated with
// clCreateBuffer.
bool IsBufferBased(const TensorStorageType& type) {
  return type == TensorStorageType::kBuffer ||
         type == TensorStorageType::kImageBuffer;
}
}  // namespace

absl::Status MemoryManager::CollectUsageInformation(
    const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors,
    std::map<ValueId, int2>* buffer_usages,
    std::map<ValueId, int2>* texture_usages) {
  for (auto input : gpu_model.input_ids_and_refs) {
    ValueId id = input.first;
    const auto& memory_type = GetTensorMemoryType(id, external_tensors);
    if (memory_type == TensorMemoryType::kBuffer) {
      AddUsage(id, 0, buffer_usages);
    } else if (memory_type == TensorMemoryType::kStrongShape) {
      AddUsage(id, 0, texture_usages);
    }
  }

  for (int node_index = 0; node_index < gpu_model.nodes.size(); ++node_index) {
    const auto& node = gpu_model.nodes[node_index];
    auto ids = node.inputs;
    ids.insert(ids.end(), node.outputs.begin(), node.outputs.end());
    for (ValueId id : ids) {
      const auto& memory_type = GetTensorMemoryType(id, external_tensors);
      if (memory_type == TensorMemoryType::kBuffer) {
        AddUsage(id, node_index, buffer_usages);
      } else if (memory_type == TensorMemoryType::kStrongShape) {
        AddUsage(id, node_index, texture_usages);
      }
    }
  }

  for (auto output : gpu_model.output_ids_and_refs) {
    ValueId id = output.first;
    const auto& memory_type = GetTensorMemoryType(id, external_tensors);
    if (memory_type == TensorMemoryType::kBuffer) {
      AddUsage(id, gpu_model.nodes.size(), buffer_usages);
    } else if (memory_type == TensorMemoryType::kStrongShape) {
      AddUsage(id, gpu_model.nodes.size(), texture_usages);
    }
  }
  return absl::OkStatus();
}

MemoryManager::TensorMemoryType MemoryManager::GetTensorMemoryType(
    ValueId id, const ExternalTensorsInfo& external_tensors) {
  if (external_tensors.immutable_tensors.find(id) !=
          external_tensors.immutable_tensors.end() ||
      external_tensors.mutable_tensors.find(id) !=
          external_tensors.mutable_tensors.end()) {
    return TensorMemoryType::kExternal;
  } else if (const_tensors_.find(id) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (IsBufferBased(tensors_descs_[id].GetStorageType())) {
    return TensorMemoryType::kBuffer;
  } else {
    return TensorMemoryType::kStrongShape;
  }
}

absl::Status MemoryManager::AllocateMemory(
    const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors) {
  for (const auto& [id, tensor] : external_tensors.immutable_tensors) {
    auto* gl_spatial_tensor = dynamic_cast<GlSpatialTensor*>(tensor);
    if (!gl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected GLSpatialTensor.");
    }
    external_immutable_tensors_[id] = gl_spatial_tensor;
  }
  tensors_descs_ = gpu_model.tensors;
  ABSL_RETURN_IF_ERROR(AllocateMemoryForConstTensors(gpu_model));
  std::map<ValueId, int2> buffer_usages;
  std::map<ValueId, int2> texture_usages;
  ABSL_RETURN_IF_ERROR(CollectUsageInformation(
      gpu_model, external_tensors, &buffer_usages, &texture_usages));

  ABSL_RETURN_IF_ERROR(AllocateMemoryForBuffers(buffer_usages));
  ABSL_RETURN_IF_ERROR(AllocateMemoryForTextures(texture_usages));
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForConstTensors(
    const GpuModel& gpu_model) {
  for (auto& description : gpu_model.const_tensors) {
    ABSL_RETURN_IF_ERROR(const_tensors_[description.first].CreateFromDescriptor(
        description.second));
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForBuffers(
    const std::map<ValueId, int2>& buffer_usages) {
  std::map<ValueId, int> value_id_to_shared_buffer_tensors;
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& t = tensors_descs_[usage.first];
    size_t buffer_size = t.GetMemorySizeInBytes();
    value_id_to_shared_buffer_tensors[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::kGreedyBest, &buffer_assignment));

  shared_buffers_.resize(buffer_assignment.object_sizes.size());
  for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
    ABSL_RETURN_IF_ERROR(CreateReadWriteBuffer(
        buffer_assignment.object_sizes[i], &shared_buffers_[i]));
  }

  for (auto& usage : buffer_usages) {
    const auto& td = tensors_descs_[usage.first];
    const int buffer_index =
        buffer_assignment
            .object_ids[value_id_to_shared_buffer_tensors[usage.first]];
    auto& tensor = value_id_to_buffer_[usage.first];
    ABSL_RETURN_IF_ERROR(CreateTensorShared(
        shared_buffers_[buffer_index].GetMemoryHandle(), td, &tensor));
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForTextures(
    const std::map<ValueId, int2>& texture_usages) {
  struct TensorDescComparator {
    TensorDescriptor tensor_desc;

    bool operator==(const TensorDescComparator& t) const {
      return tensor_desc == t.tensor_desc &&
             tensor_desc.GetBHWDCShape() == t.tensor_desc.GetBHWDCShape();
    }
  };

  std::vector<TensorUsageRecord<TensorDescComparator>> usage_records;
  std::map<ValueId, uint32_t> remap_from_value_ids;
  for (auto& usage : texture_usages) {
    remap_from_value_ids[usage.first] = usage_records.size();
    usage_records.push_back({{tensors_descs_[usage.first]},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::kEquality, &assignment));

  shared_texture_tensors_.resize(assignment.object_sizes.size());
  for (int i = 0; i < assignment.object_sizes.size(); ++i) {
    ABSL_RETURN_IF_ERROR(CreateTensor(assignment.object_sizes[i].tensor_desc,
                                      &shared_texture_tensors_[i]));
  }
  for (auto& usage : texture_usages) {
    const auto& td = tensors_descs_[usage.first];
    const auto id = assignment.object_ids[remap_from_value_ids[usage.first]];
    ABSL_RETURN_IF_ERROR(
        CreateTensorShared(shared_texture_tensors_[id].GetMemoryPtr(), td,
                           &value_id_to_texture_[usage.first]));
  }
  return absl::OkStatus();
}

uint64_t MemoryManager::GetSizeOfMemoryAllocatedForIntermediateTensors() const {
  uint64_t total_memory = 0;
  for (const auto& t : shared_texture_tensors_) {
    total_memory += t.GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    total_memory += b.GetMemorySizeInBytes();
  }
  return total_memory;
}

GlSpatialTensor* MemoryManager::GetTensor(ValueId id) {
  if (external_immutable_tensors_.find(id) !=
      external_immutable_tensors_.end()) {
    return external_immutable_tensors_[id];
  } else if (const_tensors_.find(id) != const_tensors_.end()) {
    return &const_tensors_[id];
  } else if (value_id_to_buffer_.find(id) != value_id_to_buffer_.end()) {
    return &value_id_to_buffer_[id];
  } else if (value_id_to_texture_.find(id) != value_id_to_texture_.end()) {
    return &value_id_to_texture_[id];
  } else {
    return nullptr;
  }
}

}  // namespace gl
}  // namespace ml_drift
