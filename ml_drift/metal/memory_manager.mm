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

#include "ml_drift/metal/memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/memory_management.h"
#include "ml_drift/common/memory_management/types.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/metal/common.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {
namespace {

// A heuristic to avoid offset assignment of buffers for better memory footprint. Offset assignment
// prevents memory compression when buffers are not used or filled with zeros.
constexpr uint64_t kMaxTotalOffsetSizeToEnableOffsetAssignment = 128L * 1024 * 1024;  // 128 MB

// returns true if actual memory for this storage type is buffer
bool IsBufferBased(const GpuInfo& gpu_info, const TensorStorageType& type) {
  const bool family_apple1 =
      gpu_info.IsApple() && gpu_info.apple_info.IsFamilyApple1();
  if (!family_apple1 && (type == TensorStorageType::TEXTURE_2D ||
                         type == TensorStorageType::SINGLE_TEXTURE_2D)) {
    return true;
  }
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER;
}

void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    // initializing start index(.x) and end index(.y)
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    // updating end index(.y)
    (*usage_records)[id].y = task_index;
  }
}

// Calculates the total size of the assignment.
size_t TotalSize(const ObjectsAssignment<size_t>& assignment,
                 size_t alignment = 1) {
  size_t total_size = 0;
  for (auto object_size : assignment.object_sizes) {
    total_size += AlignByN(object_size, alignment);
  }
  return total_size;
}

// Returns a vector of T* sorted in increasing memory size.
template <typename T>
std::vector<const T*> GetSortedVector(const std::vector<std::unique_ptr<T>>& items) {
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

// Takes the smallest available buffer which can hold `size`.
const Buffer* TakeAvailableBuffer(std::vector<const Buffer*>& available_buffers, size_t size) {
  auto it = absl::c_find_if(available_buffers,
                            [size](const Buffer* b) { return b->GetMemorySizeInBytes() >= size; });
  if (it == available_buffers.end()) {
    return nullptr;
  }

  const Buffer* buffer = *it;
  available_buffers.erase(it);
  return buffer;
}

// Takes the first available texture which has the same descriptor as `td`.
const MetalSpatialTensor* TakeAvailableTexture(
    std::vector<const MetalSpatialTensor*>& available_tensors, const TensorDescriptor& td) {
  auto it = absl::c_find_if(available_tensors, [td](const MetalSpatialTensor* t) {
    if (td != t->GetDescriptor()) {
      return false;
    }

    BHWDC s1 = td.GetBHWDCShape();
    BHWDC s2 = t->GetDescriptor().GetBHWDCShape();
    return s1.b == s2.b && s1.h == s2.h && s1.w == s2.w && s1.d == s2.d && s1.c == s2.c;
  });
  if (it == available_tensors.end()) {
    return nullptr;
  }

  const MetalSpatialTensor* tensor = *it;
  available_tensors.erase(it);
  return tensor;
}

uint64_t GetUniqueKey(const TensorDescriptor& tensor_desc) {
  return absl::HashOf(ToStringWithShape(tensor_desc));
}

}  // namespace

absl::StatusOr<MemoryManager::ModelId> MemoryManager::AllocateMemory(
    const Environment& env, const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors,
    std::vector<std::unique_ptr<MetalSpatialTensor>>& external_mutable_tensors) {
  ModelId model_id = next_model_id_++;
  for (const auto& [id, desc] : gpu_model.tensors) {
    tensors_descs_[Key(model_id, id)] = desc;
  }
  ABSL_ASSIGN_OR_RETURN(external_mutable_tensors,
                        AllocateExternalTensors(model_id, external_tensors, env));
  ABSL_RETURN_IF_ERROR(AllocateMemoryForConstTensors(model_id, gpu_model, env));
  std::map<ValueId, int2> buffer_usages;
  std::map<ValueId, int2> texture_usages;
  ABSL_RETURN_IF_ERROR(
      CollectUsageInformation(model_id, gpu_model, env.GetInfo(), &buffer_usages, &texture_usages));
  ABSL_RETURN_IF_ERROR(AllocateMemoryForBuffers(model_id, env, buffer_usages));
  ABSL_RETURN_IF_ERROR(AllocateMemoryForTextures(model_id, env, texture_usages));
  return model_id;
}

MetalSpatialTensor* MemoryManager::GetSpatialTensor(Key key) {
  if (external_immutable_tensors_.find(key) != external_immutable_tensors_.end()) {
    return external_immutable_tensors_[key];
  } else if (external_mutable_tensors_.find(key) != external_mutable_tensors_.end()) {
    return external_mutable_tensors_[key];
  } else if (const_tensors_.find(key) != const_tensors_.end()) {
    return &const_tensors_[key];
  } else if (value_id_to_buffer_.find(key) != value_id_to_buffer_.end()) {
    return &value_id_to_buffer_[key];
  } else if (value_id_to_texture_.find(key) != value_id_to_texture_.end()) {
    return &value_id_to_texture_[key];
  }
  return nullptr;
}

MemoryManager::TensorMemoryType MemoryManager::GetTensorMemoryType(const GpuInfo& gpu_info,
                                                                   Key key) {
  if (external_immutable_tensors_.find(key) != external_immutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (external_mutable_tensors_.find(key) != external_mutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (const_tensors_.find(key) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (IsBufferBased(gpu_info, tensors_descs_[key].GetStorageType())) {
    return TensorMemoryType::kBuffer;
  } else {
    return TensorMemoryType::kStrongShape;
  }
}

absl::Status MemoryManager::CollectUsageInformation(ModelId model_id, const GpuModel& gpu_model,
                                                    const GpuInfo& gpu_info,
                                                    std::map<ValueId, int2>* buffer_usages,
                                                    std::map<ValueId, int2>* texture_usages) {
  for (auto input : gpu_model.input_ids_and_refs) {
    ValueId id = input.first;
    const auto& memory_type = GetTensorMemoryType(gpu_info, Key(model_id, id));
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
      const auto& memory_type = GetTensorMemoryType(gpu_info, Key(model_id, id));
      if (memory_type == TensorMemoryType::kBuffer) {
        AddUsage(id, node_index, buffer_usages);
      } else if (memory_type == TensorMemoryType::kStrongShape) {
        AddUsage(id, node_index, texture_usages);
      }
    }
  }

  for (auto output : gpu_model.output_ids_and_refs) {
    ValueId id = output.first;
    const auto& memory_type = GetTensorMemoryType(gpu_info, Key(model_id, id));
    if (memory_type == TensorMemoryType::kBuffer) {
      AddUsage(id, gpu_model.nodes.size(), buffer_usages);
    } else if (memory_type == TensorMemoryType::kStrongShape) {
      AddUsage(id, gpu_model.nodes.size(), texture_usages);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<MetalSpatialTensor>>>
MemoryManager::AllocateExternalTensors(ModelId model_id,
                                       const ExternalTensorsInfo& external_tensors,
                                       const Environment& env) {
  for (const auto& [id, tensor] : external_tensors.immutable_tensors) {
    auto* metal_spatial_tensor = dynamic_cast<MetalSpatialTensor*>(tensor);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    external_immutable_tensors_[Key(model_id, id)] = metal_spatial_tensor;
  }

  std::vector<std::unique_ptr<MetalSpatialTensor>> temp_tensors;
  absl::flat_hash_map<uint64_t, MetalSpatialTensor*> mutable_tensors;
  for (const auto& [id, _] : external_tensors.mutable_tensors) {
    auto key = Key(model_id, id);
    auto& tensor_desc = tensors_descs_[key];
    auto tensor_desc_unique_key = GetUniqueKey(tensor_desc);
    if (mutable_tensors.contains(tensor_desc_unique_key)) {
      external_mutable_tensors_[key] = mutable_tensors[tensor_desc_unique_key];
      continue;
    }
    temp_tensors.push_back(std::make_unique<MetalSpatialTensor>());
    auto* metal_spatial_tensor = temp_tensors.back().get();
    ABSL_RETURN_IF_ERROR(CreateTensor(env.device(), tensors_descs_[key], metal_spatial_tensor));
    mutable_tensors[tensor_desc_unique_key] = metal_spatial_tensor;
    // It will be reset to nullptr by the caller later.
    external_mutable_tensors_[key] = metal_spatial_tensor;
  }
  return std::move(temp_tensors);
}

absl::Status MemoryManager::AllocateMemoryForConstTensors(ModelId model_id,
                                                          const GpuModel& gpu_model,
                                                          const Environment& env) {
  for (auto& description : gpu_model.const_tensors) {
    ABSL_RETURN_IF_ERROR(const_tensors_[Key(model_id, description.first)].CreateFromDescriptor(
        description.second, env.device()));
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForBuffers(ModelId model_id, const Environment& env,
                                                     const std::map<ValueId, int2>& buffer_usages) {
  // From Apple documentation:
  // For buffers in the device address space, align the offset to the data type
  // consumed by the compute function (which is always less than or equal to 16
  // bytes).
  // For buffers in the constant address space, align the offset to 256
  // bytes in macOS. In iOS, align the offset to the maximum of either the data
  // type consumed by the compute function, or 4 bytes. A 16-byte alignment is
  // safe in iOS if you don't need to consider the data type.
  if (buffer_usages.empty()){
    return absl::OkStatus();
  }
#if defined(TARGET_IOS) || defined(TARGET_TVOS)
  const size_t kConstAlignment = 16;
#elif defined(TARGET_MACOS)
  const size_t kConstAlignment = 256;
#else
  const size_t kConstAlignment = 256;
#endif
  size_t min_common_alignment = kConstAlignment;
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  std::map<ValueId, int> value_id_to_shared_buffer_tensors;
  for (auto& usage : buffer_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const auto& shape = td.GetBHWDCShape();
    const auto& storage_dims = td.GetStorageDims();
    const size_t element_size = SizeOf(td.GetDataType());
    size_t buffer_size;
    if (td.GetStorageType() == TensorStorageType::TEXTURE_2D ||
        td.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D) {
      const size_t row_bytes_alignment = [env.device()
          minimumLinearTextureAlignmentForPixelFormat:DataTypeToPixelFormat(td.GetDataType(),
                                                                            td.GetElementSize(),
                                                                            false)];
      min_common_alignment =
          std::lcm(min_common_alignment, row_bytes_alignment);
      const int vec_size =
          td.GetStorageType() == TensorStorageType::TEXTURE_2D ? 4 : shape.c;
      const size_t bytes_per_row = storage_dims[0] * vec_size * element_size;
      buffer_size =
          AlignByN(bytes_per_row, row_bytes_alignment) * storage_dims[1];
    } else {
      buffer_size = storage_dims[0] * 4 * element_size;
    }
    value_id_to_shared_buffer_tensors[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(buffer_usage_records, MemoryStrategy::GREEDY_BEST,
                                              &buffer_assignment));

  OffsetsAssignment offset_assignment;
  ABSL_RETURN_IF_ERROR(AssignOffsetsToTensors(buffer_usage_records, MemoryStrategy::GREEDY_BY_SIZE,
                                              &offset_assignment, min_common_alignment));

  bool use_offset_assignment = false;
  if (offset_assignment.total_size <= TotalSize(buffer_assignment) &&
      offset_assignment.total_size <= kMaxTotalOffsetSizeToEnableOffsetAssignment &&
      offset_assignment.total_size <= env.GetInfo().GetMaxBufferSize()) {
    use_offset_assignment = true;
  }

  std::vector<const Buffer*> mtl_buffers;
  std::vector<std::unique_ptr<Buffer>> new_buffers;

  if (use_offset_assignment) {
    size_t required_size = offset_assignment.total_size;
    std::vector<const Buffer*> available_buffers = GetSortedVector(shared_buffers_);
    const Buffer* selected_buffer = TakeAvailableBuffer(available_buffers, required_size);
    if (!selected_buffer) {
      new_buffers.push_back(std::make_unique<Buffer>());
      ABSL_RETURN_IF_ERROR(
          CreateBuffer(required_size, /*data=*/nullptr, env.device(), new_buffers.back().get()));
      selected_buffer = new_buffers.back().get();
    }
    mtl_buffers.push_back(selected_buffer);
  } else {
    std::vector<const Buffer*> available_buffers = GetSortedVector(shared_buffers_);
    mtl_buffers.resize(buffer_assignment.object_sizes.size());
    for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
      size_t size = buffer_assignment.object_sizes[i];
      const Buffer* selected_buffer = TakeAvailableBuffer(available_buffers, size);
      if (!selected_buffer) {
        new_buffers.push_back(std::make_unique<Buffer>());
        ABSL_RETURN_IF_ERROR(
            CreateBuffer(size, /*data=*/nullptr, env.device(), new_buffers.back().get()));
        selected_buffer = new_buffers.back().get();
      }
      mtl_buffers[i] = selected_buffer;
    }
  }

  for (auto& buf : new_buffers) {
    shared_buffers_.push_back(std::move(buf));
  }

  for (auto& usage : buffer_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const int buffer_index =
        buffer_assignment
            .object_ids[value_id_to_shared_buffer_tensors[usage.first]];
    auto& tensor = value_id_to_buffer_[Key(model_id, usage.first)];
    uint64_t base_buffer_offset = 0;
    const Buffer* selected_buffer = nullptr;
    if (use_offset_assignment) {
      selected_buffer = mtl_buffers[0];
      base_buffer_offset =
          offset_assignment
              .offsets[value_id_to_shared_buffer_tensors[usage.first]];
    } else {
      selected_buffer = mtl_buffers[buffer_index];
      base_buffer_offset = 0;
    }
    if (td.GetStorageType() == TensorStorageType::TEXTURE_2D ||
        td.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D) {
      size_t row_bytes_alignment = [env.device()
          minimumLinearTextureAlignmentForPixelFormat:DataTypeToPixelFormat(td.GetDataType(),
                                                                            td.GetElementSize(),
                                                                            false)];
      @autoreleasepool {
        ABSL_RETURN_IF_ERROR(CreateTensorSharedImage2DBuffer(
            selected_buffer->GetMemoryPtr(), td, row_bytes_alignment, &tensor, base_buffer_offset));
      }
    } else {
      @autoreleasepool {
        ABSL_RETURN_IF_ERROR(CreateTensorSharedBuffer(selected_buffer->GetMemoryPtr(), td, &tensor,
                                                      base_buffer_offset));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForTextures(
    ModelId model_id, const Environment& env, const std::map<ValueId, int2>& texture_usages) {
  if (texture_usages.empty()) {
    return absl::OkStatus();
  }
  struct TensorDescComparator {
    TensorDescriptor tensor_desc;

    bool operator==(const TensorDescComparator& t) const {
      return tensor_desc == t.tensor_desc &&
             tensor_desc.GetBHWDCShape() == t.tensor_desc.GetBHWDCShape();
    }
  };

  std::vector<TensorUsageRecord<TensorDescComparator>> usage_records;
  std::map<ValueId, ValueId> remap_from_value_ids;
  for (auto& usage : texture_usages) {
    remap_from_value_ids[usage.first] = usage_records.size();
    usage_records.push_back({{tensors_descs_[Key(model_id, usage.first)]},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  ABSL_RETURN_IF_ERROR(
      AssignObjectsToTensors(usage_records, MemoryStrategy::EQUALITY, &assignment));

  std::vector<const MetalSpatialTensor*> mtl_textures(assignment.object_sizes.size(), nullptr);
  std::vector<std::unique_ptr<MetalSpatialTensor>> new_textures;

  std::vector<const MetalSpatialTensor*> available_textures =
      GetSortedVector(shared_texture_tensors_);

  for (int i = 0; i < assignment.object_sizes.size(); ++i) {
    const auto& td = assignment.object_sizes[i].tensor_desc;
    const MetalSpatialTensor* selected_tensor = TakeAvailableTexture(available_textures, td);
    if (selected_tensor) {
      mtl_textures[i] = selected_tensor;
    } else {
      new_textures.push_back(std::make_unique<MetalSpatialTensor>());
      ABSL_RETURN_IF_ERROR(CreateTensor(env.device(), td, new_textures.back().get()));
      mtl_textures[i] = new_textures.back().get();
    }
  }

  for (auto& tex : new_textures) {
    shared_texture_tensors_.push_back(std::move(tex));
  }

  for (auto& usage : texture_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const auto id = assignment.object_ids[remap_from_value_ids[usage.first]];
    ABSL_RETURN_IF_ERROR(
        CreateTensorSharedTexture(mtl_textures[id]->GetTextureHandle(), td,
                                  &value_id_to_texture_[Key(model_id, usage.first)]));
  }
  return absl::OkStatus();
}

uint64_t MemoryManager::GetIntermediateTensorsSize() const {
  uint64_t total_memory = 0;
  for (const auto& t : shared_texture_tensors_) {
    total_memory += t->GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    total_memory += b->GetMemorySizeInBytes();
  }

  return total_memory;
}

uint64_t MemoryManager::GetConstantTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& t : const_tensors_) {
    total_size += t.second.GetMemorySizeInBytes();
  }
  return total_size;
}

absl::Status MemoryManager::SetExternalTensor(const Key& key, MetalSpatialTensor* tensor_ptr) {
  auto it = external_mutable_tensors_.find(key);
  if (it == external_mutable_tensors_.end()) {
    return absl::InvalidArgumentError("No external tensor with this id.");
  }
  external_mutable_tensors_[key] = tensor_ptr;
  return absl::OkStatus();
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddConstantsToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& [key, tensor] : const_tensors_) {
    if (tensor.GetBufferHandle() != nil) {
      [residency_set addAllocation:tensor.GetBufferHandle()];
    }
    if (tensor.GetTextureHandle() != nil) {
      [residency_set addAllocation:tensor.GetTextureHandle()];
    }
  }
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddIntermediatesToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& buffer : shared_buffers_) {
    if (buffer->GetMemoryPtr() != nil) {
      [residency_set addAllocation:buffer->GetMemoryPtr()];
    }
  }

  for (const auto& tensor : shared_texture_tensors_) {
    if (tensor->GetTextureHandle() != nil) {
      [residency_set addAllocation:tensor->GetTextureHandle()];
    }
    if (tensor->GetBufferHandle() != nil) {
      [residency_set addAllocation:tensor->GetBufferHandle()];
    }
  }
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddExternalImmutableToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& [key, tensor] : external_immutable_tensors_) {
    if (tensor != nullptr) {
      if (tensor->GetBufferHandle() != nil) {
        [residency_set addAllocation:tensor->GetBufferHandle()];
      }
      if (tensor->GetTextureHandle() != nil) {
        [residency_set addAllocation:tensor->GetTextureHandle()];
      }
    }
  }
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddExternalMutableToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& [key, tensor] : external_mutable_tensors_) {
    if (tensor != nullptr) {
      if (tensor->GetBufferHandle() != nil) {
        [residency_set addAllocation:tensor->GetBufferHandle()];
      }
      if (tensor->GetTextureHandle() != nil) {
        [residency_set addAllocation:tensor->GetTextureHandle()];
      }
    }
  }
}

}  // namespace metal
}  // namespace ml_drift
