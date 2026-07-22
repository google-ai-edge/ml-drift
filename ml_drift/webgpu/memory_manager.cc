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

#include "ml_drift/webgpu/memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/hash/hash.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/memory_management.h"
#include "ml_drift/common/memory_management/types.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
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

bool IsBufferBased(const TensorStorageType& type) {
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER;
}

// Returns a vector of T* sorted in increasing memory size.
template <typename T>
std::vector<const T*> GetSortedVector(
    const std::vector<std::unique_ptr<T>>& items) {
  std::vector<const T*> sorted_items;
  sorted_items.reserve(items.size());
  for (const auto& item : items) {
    sorted_items.push_back(item.get());
  }
  absl::c_sort(sorted_items, [](const T* i1, const T* i2) {
    return i1->GetMemorySizeInBytes() < i2->GetMemorySizeInBytes();
  });
  return sorted_items;
}

// Takes the smallest available texture which can hold `td`.
const SpatialTensor* TakeAvailableTexture(
    std::vector<const SpatialTensor*>& available_tensors,
    const TensorDescriptor& td) {
  auto it = absl::c_find_if(available_tensors, [td](const SpatialTensor* t) {
    if (td != t->GetDescriptor()) {
      return false;
    }

    // For a texture to match it must be large enough in every dimension. This
    // will only work if no clamping is used (which is the case for WebGPU).
    BHWDC s1 = td.GetBHWDCShape();
    BHWDC s2 = t->GetDescriptor().GetBHWDCShape();
    return s1.b <= s2.b && s1.h <= s2.h && s1.w <= s2.w && s1.d <= s2.d &&
           s1.c <= s2.c;
  });
  if (it == available_tensors.end()) {
    return nullptr;
  }

  const SpatialTensor* tensor = *it;
  available_tensors.erase(it);
  return tensor;
}

// Takes the smallest available buffer which can hold `size`.
const Buffer* TakeAvailableBuffer(std::vector<const Buffer*>& available_buffers,
                                  size_t size) {
  auto it = absl::c_find_if(available_buffers, [size](const Buffer* b) {
    return b->GetMemorySizeInBytes() >= size;
  });
  if (it == available_buffers.end()) {
    return nullptr;
  }

  const Buffer* buffer = *it;
  available_buffers.erase(it);
  return buffer;
}

uint64_t GetUniqueKey(const TensorDescriptor& tensor_desc) {
  return absl::HashOf(ToStringWithShape(tensor_desc));
}

}  // namespace

MemoryManager::TensorMemoryType MemoryManager::GetTensorMemoryType(Key key) {
  if (external_immutable_tensors_.find(key) !=
      external_immutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (external_mutable_tensors_.find(key) !=
             external_mutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (const_tensors_.find(key) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (IsBufferBased(tensor_descriptors_[key].GetStorageType())) {
    return TensorMemoryType::kBuffer;
  } else {
    return TensorMemoryType::kTexture;
  }
}

absl::StatusOr<MemoryManager::ModelId> MemoryManager::AllocateMemory(
    const Environment& environment, const GpuModel& gpu_model,
    const ExternalTensorsInfo& external_tensors,
    std::vector<std::unique_ptr<SpatialTensor>>& external_mutable_tensors) {
  for (const auto& [k, v] : gpu_model.tensors) {
    tensor_descriptors_[Key(next_model_id_, k)] = v;
  }
  for (const auto& external : external_tensors.immutable_tensors) {
    SpatialTensor* spatial_tensor =
        static_cast<SpatialTensor*>(external.second);
    external_immutable_tensors_[Key(next_model_id_, external.first)] =
        spatial_tensor;
  }
  absl::flat_hash_map<uint64_t, SpatialTensor*> mutable_tensors;
  for (const auto& external : external_tensors.mutable_tensors) {
    Key key = Key(next_model_id_, external.first);
    auto tensor_desc = GetTensorDescriptor(key);
    uint64_t tensor_desc_unique_key = GetUniqueKey(tensor_desc);
    if (mutable_tensors.contains(tensor_desc_unique_key)) {
      external_mutable_tensors_[key] = mutable_tensors[tensor_desc_unique_key];
      continue;
    }
    external_mutable_tensors.push_back(std::make_unique<SpatialTensor>());
    auto* tensor = external_mutable_tensors.back().get();
    ABSL_RETURN_IF_ERROR(
        CreateTensor(environment.device(), tensor_desc, tensor));
    mutable_tensors[tensor_desc_unique_key] = tensor;
    // It will be reset to nullptr by the caller later.
    external_mutable_tensors_[key] = tensor;
  }
  ABSL_RETURN_IF_ERROR(AllocateMemoryForConstTensors(environment, gpu_model));
  std::map<ValueId, int2> buffer_usages;
  std::map<ValueId, int2> texture_usages;
  ABSL_RETURN_IF_ERROR(
      CollectUsageInformation(gpu_model, &buffer_usages, &texture_usages));

  ABSL_RETURN_IF_ERROR(AllocateMemoryForBuffers(environment, buffer_usages));
  ABSL_RETURN_IF_ERROR(AllocateMemoryForTextures(environment, texture_usages));
  return next_model_id_++;
}

const TensorDescriptor& MemoryManager::GetTensorDescriptor(Key key) {
  return tensor_descriptors_.at(key);
}

SpatialTensor* MemoryManager::GetSpatialTensor(Key key) {
  if (const_tensors_.find(key) != const_tensors_.end()) {
    return const_tensors_[key].get();
  } else if (external_mutable_tensors_.find(key) !=
             external_mutable_tensors_.end()) {
    return external_mutable_tensors_[key];
  } else if (external_immutable_tensors_.find(key) !=
             external_immutable_tensors_.end()) {
    return external_immutable_tensors_[key];
  } else if (value_id_to_buffer_.find(key) != value_id_to_buffer_.end()) {
    return value_id_to_buffer_[key].get();
  } else if (value_id_to_texture_.find(key) != value_id_to_texture_.end()) {
    return value_id_to_texture_[key].get();
  } else {
    return nullptr;
  }
}

absl::Status MemoryManager::SetExternalTensor(Key key, SpatialTensor* tensor) {
  auto it = external_mutable_tensors_.find(key);
  if (it == external_mutable_tensors_.end()) {
    return absl::InvalidArgumentError("No external tensor with this id.");
  }
  it->second = tensor;
  return absl::OkStatus();
}

uint64_t MemoryManager::GetSizeOfMemoryAllocatedForIntermediateTensors() const {
  uint64_t total_memory = 0;
  for (const auto& t : shared_texture_tensors_) {
    total_memory += t->GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    total_memory += b->GetMemorySizeInBytes();
  }
  return total_memory;
}

uint64_t MemoryManager::GetSizeOfMemoryAllocatedForConstTensors() const {
  uint64_t total_memory = 0;
  for (const auto& t : const_tensors_) {
    total_memory += t.second->GetMemorySizeInBytes();
  }
  return total_memory;
}

absl::Status MemoryManager::CollectUsageInformation(
    const GpuModel& gpu_model, std::map<ValueId, int2>* buffer_usages,
    std::map<ValueId, int2>* texture_usages) {
  for (auto input : gpu_model.input_ids_and_refs) {
    ValueId id = input.first;
    const auto& memory_type = GetTensorMemoryType(Key(next_model_id_, id));
    if (memory_type == TensorMemoryType::kBuffer) {
      AddUsage(id, 0, buffer_usages);
    } else if (memory_type == TensorMemoryType::kTexture) {
      AddUsage(id, 0, texture_usages);
    }
  }

  for (int node_index = 0; node_index < gpu_model.nodes.size(); ++node_index) {
    const auto& node = gpu_model.nodes[node_index];
    auto ids = node.inputs;
    ids.insert(ids.end(), node.outputs.begin(), node.outputs.end());
    for (ValueId id : ids) {
      const auto& memory_type = GetTensorMemoryType(Key(next_model_id_, id));
      if (memory_type == TensorMemoryType::kBuffer) {
        AddUsage(id, node_index, buffer_usages);
      } else if (memory_type == TensorMemoryType::kTexture) {
        AddUsage(id, node_index, texture_usages);
      }
    }
  }

  for (auto output : gpu_model.output_ids_and_refs) {
    ValueId id = output.first;
    const auto& memory_type = GetTensorMemoryType(Key(next_model_id_, id));
    if (memory_type == TensorMemoryType::kBuffer) {
      AddUsage(id, gpu_model.nodes.size(), buffer_usages);
    } else if (memory_type == TensorMemoryType::kTexture) {
      AddUsage(id, gpu_model.nodes.size(), texture_usages);
    }
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForConstTensors(
    const Environment& environment, const GpuModel& gpu_model) {
  for (const auto& description : gpu_model.const_tensors) {
    auto& tensor = const_tensors_[Key(next_model_id_, description.first)];
    tensor = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(
        tensor->CreateFromDescriptor(environment.device(), description.second));
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForBuffers(
    const Environment& environment,
    const std::map<ValueId, int2>& buffer_usages) {
  std::map<ValueId, int> value_id_to_shared_buffer_tensors;

  std::vector<TensorUsageRecord<size_t>> usage_records;
  for (auto& usage : buffer_usages) {
    const auto& td = GetTensorDescriptor(Key(next_model_id_, usage.first));
    size_t buffer_size = td.GetMemorySizeInBytes();
    value_id_to_shared_buffer_tensors[usage.first] = usage_records.size();
    usage_records.push_back({buffer_size, static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> assignment;
  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::GREEDY_BEST, &assignment));

  std::vector<wgpu::Buffer> buffers(assignment.object_sizes.size());
  std::vector<const Buffer*> available_buffers =
      GetSortedVector(shared_buffers_);
  for (int i = 0; i < assignment.object_sizes.size(); ++i) {
    size_t size = assignment.object_sizes[i];
    const Buffer* buffer = TakeAvailableBuffer(available_buffers, size);
    if (!buffer) {
      shared_buffers_.push_back(std::make_unique<Buffer>(
          CreateBufferStorage(environment.device(), size)));
      buffer = shared_buffers_.back().get();
    }
    buffers[i] = buffer->GetMemoryHandle();
  }

  for (auto& usage : buffer_usages) {
    const auto& td = GetTensorDescriptor(Key(next_model_id_, usage.first));
    const int buffer_index =
        assignment.object_ids[value_id_to_shared_buffer_tensors[usage.first]];
    auto& buffer = value_id_to_buffer_[Key(next_model_id_, usage.first)];
    buffer = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(
        CreateSharedTensor(buffers[buffer_index], td, buffer.get()));
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForTextures(
    const Environment& environment,
    const std::map<ValueId, int2>& texture_usages) {
  std::map<ValueId, uint32_t> remap_from_value_ids;
  struct TensorDescComparator {
    TensorDescriptor tensor_desc;

    bool operator==(const TensorDescComparator& t) const {
      // The BHWDC == operator is slow when doing a large number of comparisons,
      // so use a hand rolled version here.
      const BHWDC& s = tensor_desc.GetBHWDCShape();
      const BHWDC& s2 = t.tensor_desc.GetBHWDCShape();
      return tensor_desc == t.tensor_desc && s.b == s2.b && s.h == s2.h &&
             s.w == s2.w && s.d == s2.d && s.c == s2.c;
    }
  };
  std::vector<TensorUsageRecord<TensorDescComparator>> usage_records;
  for (auto& usage : texture_usages) {
    const auto& td = GetTensorDescriptor(Key(next_model_id_, usage.first));
    remap_from_value_ids[usage.first] = usage_records.size();
    usage_records.push_back({{td},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::EQUALITY, &assignment));

  std::vector<std::pair<wgpu::Texture, wgpu::TextureView>> textures(
      assignment.object_sizes.size());
  std::vector<const SpatialTensor*> available_tensors =
      GetSortedVector(shared_texture_tensors_);
  for (int i = 0; i < assignment.object_sizes.size(); ++i) {
    const auto& td = assignment.object_sizes[i].tensor_desc;
    const SpatialTensor* tensor = TakeAvailableTexture(available_tensors, td);
    if (!tensor) {
      auto& new_tensor =
          shared_texture_tensors_.emplace_back(new SpatialTensor());
      ABSL_RETURN_IF_ERROR(
          CreateTensor(environment.device(), td, new_tensor.get()));
      tensor = new_tensor.get();
    }
    textures[i] = {tensor->GetTextureHandle(), tensor->GetTextureViewHandle()};
  }

  for (auto& usage : texture_usages) {
    const auto& td = GetTensorDescriptor(Key(next_model_id_, usage.first));
    const auto id = assignment.object_ids[remap_from_value_ids[usage.first]];
    const auto& texture = textures[id];
    auto& tensor = value_id_to_texture_[Key(next_model_id_, usage.first)];
    tensor = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(
        CreateSharedTensor(texture.first, texture.second, td, tensor.get()));
  }
  return absl::OkStatus();
}

}  // namespace webgpu
}  // namespace ml_drift
