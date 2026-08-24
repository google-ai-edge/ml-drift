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

#include "ml_drift/cl/memory_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/memory_management.h"
#include "ml_drift/common/memory_management/types.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace cl {
namespace {

enum class TensorType { kConst, kExternal, kRuntime };

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
bool IsBufferBased(const GpuInfo& gpu_info, const TensorStorageType& type) {
  const bool image2d_based_buffer =
      (type == TensorStorageType::TEXTURE_2D ||
       type == TensorStorageType::SINGLE_TEXTURE_2D) &&
      gpu_info.opencl_info.IsImage2dFromBufferSupported();
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER || image2d_based_buffer;
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

TensorType GetTensorType(const GpuModel& gpu_model,
                         const ExternalTensorsInfo& external_tensors,
                         const GpuInfo& gpu_info, ValueId id) {
  if (external_tensors.immutable_tensors.find(id) !=
          external_tensors.immutable_tensors.end() ||
      external_tensors.mutable_tensors.find(id) !=
          external_tensors.mutable_tensors.end()) {
    return TensorType::kExternal;
  } else if (gpu_model.const_tensors.find(id) !=
             gpu_model.const_tensors.end()) {
    return TensorType::kConst;
  } else {
    return TensorType::kRuntime;
  }
}

void CollectUsageInformation(const GpuModel& gpu_model, const GpuInfo& gpu_info,
                             const ExternalTensorsInfo& external_tensors,
                             std::map<ValueId, int2>* buffer_usages,
                             std::map<ValueId, int2>* texture_usages) {
  for (auto input : gpu_model.input_ids_and_refs) {
    ValueId id = input.first;
    const auto& memory_type =
        GetTensorType(gpu_model, external_tensors, gpu_info, id);
    if (memory_type == TensorType::kRuntime &&
        IsBufferBased(gpu_info, gpu_model.tensors.at(id).GetStorageType())) {
      AddUsage(id, 0, buffer_usages);
    } else if (memory_type == TensorType::kRuntime) {
      AddUsage(id, 0, texture_usages);
    }
  }

  for (int node_index = 0; node_index < gpu_model.nodes.size(); ++node_index) {
    const auto& node = gpu_model.nodes[node_index];
    auto ids = node.inputs;
    ids.insert(ids.end(), node.outputs.begin(), node.outputs.end());
    for (ValueId id : ids) {
      const auto& memory_type =
          GetTensorType(gpu_model, external_tensors, gpu_info, id);
      if (memory_type == TensorType::kRuntime &&
          IsBufferBased(gpu_info, gpu_model.tensors.at(id).GetStorageType())) {
        AddUsage(id, node_index, buffer_usages);
      } else if (memory_type == TensorType::kRuntime) {
        AddUsage(id, node_index, texture_usages);
      }
    }
  }

  for (auto output : gpu_model.output_ids_and_refs) {
    ValueId id = output.first;
    const auto& memory_type =
        GetTensorType(gpu_model, external_tensors, gpu_info, id);
    if (memory_type == TensorType::kRuntime &&
        IsBufferBased(gpu_info, gpu_model.tensors.at(id).GetStorageType())) {
      AddUsage(id, gpu_model.nodes.size(), buffer_usages);
    } else if (memory_type == TensorType::kRuntime) {
      AddUsage(id, gpu_model.nodes.size(), texture_usages);
    }
  }
}

size_t GetWidthAlignment(const GpuInfo& gpu_info, size_t bytes_per_pixel) {
  size_t width_pixel_alignment = gpu_info.opencl_info.image_pitch_alignment;
  if (gpu_info.IsAdreno() && width_pixel_alignment % bytes_per_pixel == 0) {
    width_pixel_alignment /= bytes_per_pixel;
  }
  return width_pixel_alignment;
}

size_t GetTensorMemorySize(const GpuInfo& gpu_info,
                           const TensorDescriptor& descriptor) {
  if (descriptor.GetStorageType() == TensorStorageType::TEXTURE_2D ||
      descriptor.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D) {
    const size_t bytes_per_pixel =
        SizeOf(descriptor.GetDataType()) * descriptor.GetElementSize();
    const size_t width_pixel_alignment =
        GetWidthAlignment(gpu_info, bytes_per_pixel);
    std::vector<uint64_t> storage_dims = descriptor.GetStorageDims();
    const size_t width = storage_dims[0];
    const size_t height = storage_dims[1];
    const size_t width_aligned = AlignByN(width, width_pixel_alignment);
    return width_aligned * bytes_per_pixel * height;
  } else {
    return descriptor.GetMemorySizeInBytes();
  }
}

bool HasBufferBasedImages(const std::map<ValueId, int2>& buffer_usages,
                          const GpuModel& gpu_model) {
  for (const auto& usage : buffer_usages) {
    const auto& storage_type =
        gpu_model.tensors.at(usage.first).GetStorageType();
    if (storage_type == TensorStorageType::IMAGE_BUFFER ||
        storage_type == TensorStorageType::TEXTURE_2D ||
        storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
      return true;
    }
  }
  return false;
}

absl::Status GetBufferAssignment(
    const std::map<ValueId, int2>& buffer_usages, const GpuModel& gpu_model,
    const GpuInfo& gpu_info,
    std::vector<TensorUsageRecord<size_t>>* buffer_usage_records,
    std::map<ValueId, int>* graph_ids_to_shared_buffer_tensors,
    ObjectsAssignment<size_t>* buffer_assignment,
    OffsetsAssignment* offset_assignment, bool* use_offset_assignment,
    bool* is_sub_buffers_supported) {
  for (const auto& usage : buffer_usages) {
    if (graph_ids_to_shared_buffer_tensors) {
      (*graph_ids_to_shared_buffer_tensors)[usage.first] =
          buffer_usage_records->size();
    }
    buffer_usage_records->push_back(
        {GetTensorMemorySize(gpu_info, gpu_model.tensors.at(usage.first)),
         static_cast<TaskId>(usage.second.x),
         static_cast<TaskId>(usage.second.y)});
  }

  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(
      *buffer_usage_records, MemoryStrategy::GREEDY_BEST, buffer_assignment));

  const bool has_buffer_based_images =
      HasBufferBasedImages(buffer_usages, gpu_model);
  *is_sub_buffers_supported =
      (!has_buffer_based_images && gpu_info.IsCL11OrHigher()) ||
      CanUseSubBufferForImage2d(gpu_info);
  const size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);

  *use_offset_assignment = false;
  if (*is_sub_buffers_supported) {
    ABSL_RETURN_IF_ERROR(AssignOffsetsToTensors(
        *buffer_usage_records, MemoryStrategy::GREEDY_BY_SIZE,
        offset_assignment, base_align_bytes));
    if (offset_assignment->total_size <= TotalSize(*buffer_assignment) &&
        offset_assignment->total_size <= gpu_info.GetMaxBufferSize()) {
      *use_offset_assignment = true;
    }
  }
  return absl::OkStatus();
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
  std::sort(sorted_items.begin(), sorted_items.end(),
            [](const T* i1, const T* i2) {
              return i1->GetMemorySizeInBytes() < i2->GetMemorySizeInBytes();
            });
  return sorted_items;
}

const Buffer* TakeAvailableBuffer(std::vector<const Buffer*>& available_buffers,
                                  size_t size) {
  auto it = std::find_if(
      available_buffers.begin(), available_buffers.end(),
      [size](const Buffer* b) { return b->GetMemorySizeInBytes() >= size; });
  if (it == available_buffers.end()) {
    return nullptr;
  }
  const Buffer* buffer = *it;
  available_buffers.erase(it);
  return buffer;
}

const Tensor* TakeAvailableTexture(
    std::vector<const Tensor*>& available_tensors, const TensorDescriptor& td) {
  auto it = std::find_if(available_tensors.begin(), available_tensors.end(),
                         [td](const Tensor* t) {
                           if (td != t->GetDescriptor()) {
                             return false;
                           }
                           BHWDC s1 = td.GetBHWDCShape();
                           BHWDC s2 = t->GetDescriptor().GetBHWDCShape();
                           return s1.b == s2.b && s1.h == s2.h &&
                                  s1.w == s2.w && s1.d == s2.d && s1.c == s2.c;
                         });
  if (it == available_tensors.end()) {
    return nullptr;
  }
  const Tensor* tensor = *it;
  available_tensors.erase(it);
  return tensor;
}

}  // namespace

absl::StatusOr<MemoryManager::ModelId> MemoryManager::AllocateMemory(
    const GpuModel& gpu_model, const GpuInfo& gpu_info,
    const ExternalTensorsInfo& external_tensors,
    std::vector<std::unique_ptr<Tensor>>& external_mutable_tensors,
    CLContext* context) {
  const ModelId model_id = next_model_id_++;
  for (const auto& [id, tensor_desc] : gpu_model.tensors) {
    tensors_descs_[Key(model_id, id)] = tensor_desc;
  }
  ABSL_ASSIGN_OR_RETURN(
      external_mutable_tensors,
      AllocateExternalTensors(model_id, *context, gpu_model, external_tensors));
  ABSL_RETURN_IF_ERROR(AllocateConstTensors(model_id, gpu_model, context));
  std::map<ValueId, int2> buffer_usages;
  std::map<ValueId, int2> texture_usages;
  CollectUsageInformation(gpu_model, gpu_info, external_tensors, &buffer_usages,
                          &texture_usages);
  ABSL_RETURN_IF_ERROR(AllocateBufferBasedTensors(
      model_id, buffer_usages, gpu_model, gpu_info, context));
  ABSL_RETURN_IF_ERROR(AllocateTextureBasedTensors(
      model_id, texture_usages, gpu_model, gpu_info, context));
  return model_id;
}

uint64_t GetUniqueKey(const TensorDescriptor& tensor_desc) {
  return absl::HashOf(ToStringWithShape(tensor_desc));
}

absl::StatusOr<std::vector<std::unique_ptr<Tensor>>>
MemoryManager::AllocateExternalTensors(
    ModelId model_id, const CLContext& context, const GpuModel& gpu_model,
    const ExternalTensorsInfo& external_tensors) {
  for (const auto& [id, tensor] : external_tensors.immutable_tensors) {
    auto* cl_spatial_tensor = dynamic_cast<Tensor*>(tensor);
    if (!cl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected CLSpatialTensor.");
    }
    external_immutable_tensors_[Key(model_id, id)] = cl_spatial_tensor;
  }

  std::vector<std::unique_ptr<Tensor>> temp_tensors;
  absl::flat_hash_map<uint64_t, Tensor*> mutable_tensors;
  for (const auto& [id, _] : external_tensors.mutable_tensors) {
    const auto& tensor_desc = gpu_model.tensors.at(id);
    uint64_t tensor_desc_unique_key = GetUniqueKey(tensor_desc);
    if (mutable_tensors.contains(tensor_desc_unique_key)) {
      external_mutable_tensors_[Key(model_id, id)] =
          mutable_tensors[tensor_desc_unique_key];
      continue;
    }
    temp_tensors.push_back(std::make_unique<Tensor>());
    auto* cl_spatial_tensor = temp_tensors.back().get();
    ABSL_RETURN_IF_ERROR(CreateTensor(context, tensor_desc, cl_spatial_tensor));
    mutable_tensors[tensor_desc_unique_key] = cl_spatial_tensor;
    external_mutable_tensors_[Key(model_id, id)] = cl_spatial_tensor;
  }
  return std::move(temp_tensors);
}

absl::Status MemoryManager::AllocateConstTensors(ModelId model_id,
                                                 const GpuModel& gpu_model,
                                                 CLContext* context) {
  for (auto& description : gpu_model.const_tensors) {
    ABSL_RETURN_IF_ERROR(
        const_tensors_[Key(model_id, description.first)].CreateFromDescriptor(
            description.second, *context));
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateBufferBasedTensors(
    ModelId model_id, const std::map<ValueId, int2>& buffer_usages,
    const GpuModel& gpu_model, const GpuInfo& gpu_info, CLContext* context) {
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  ObjectsAssignment<size_t> buffer_assignment;
  OffsetsAssignment offset_assignment;
  bool use_offset_assignment;
  bool is_sub_buffers_supported;
  std::map<ValueId, int> value_id_to_shared_buffer_tensors;
  ABSL_RETURN_IF_ERROR(GetBufferAssignment(
      buffer_usages, gpu_model, gpu_info, &buffer_usage_records,
      &value_id_to_shared_buffer_tensors, &buffer_assignment,
      &offset_assignment, &use_offset_assignment, &is_sub_buffers_supported));
  const size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);

  if (buffer_usage_records.empty()) {
    return absl::OkStatus();
  }

  std::vector<const Buffer*> cl_buffers;
  std::vector<std::unique_ptr<Buffer>> new_buffers;
  std::vector<const Buffer*> allocated_buffers;

  if (use_offset_assignment) {
    size_t required_size = offset_assignment.total_size;
    std::vector<const Buffer*> available_buffers =
        GetSortedVector(shared_buffers_);
    const Buffer* selected_buffer =
        TakeAvailableBuffer(available_buffers, required_size);
    if (!selected_buffer) {
      auto parent_buf = std::make_unique<Buffer>();
      Buffer shared_buffer;
      ABSL_RETURN_IF_ERROR(
          CreateReadWriteBuffer(required_size, context, &shared_buffer));
      *parent_buf = std::move(shared_buffer);
      selected_buffer = parent_buf.get();
      new_buffers.push_back(std::move(parent_buf));
    }
    cl_buffers.push_back(selected_buffer);

    allocated_buffers.resize(offset_assignment.offsets.size());
    for (int i = 0; i < offset_assignment.offsets.size(); ++i) {
      ABSL_ASSIGN_OR_RETURN(
          auto sub_buf,
          CreateSubBuffer(*selected_buffer, offset_assignment.offsets[i],
                          buffer_usage_records[i].tensor_size, context));
      sub_buffers_.push_back(std::make_unique<Buffer>(std::move(sub_buf)));
      allocated_buffers[i] = sub_buffers_.back().get();
    }
  } else {
    const size_t total_size = TotalSize(buffer_assignment, base_align_bytes);
    if (is_sub_buffers_supported && total_size <= gpu_info.GetMaxBufferSize()) {
      std::vector<const Buffer*> available_buffers =
          GetSortedVector(shared_buffers_);
      const Buffer* selected_buffer =
          TakeAvailableBuffer(available_buffers, total_size);
      if (!selected_buffer) {
        auto parent_buf = std::make_unique<Buffer>();
        Buffer shared_buffer;
        ABSL_RETURN_IF_ERROR(
            CreateReadWriteBuffer(total_size, context, &shared_buffer));
        *parent_buf = std::move(shared_buffer);
        selected_buffer = parent_buf.get();
        new_buffers.push_back(std::move(parent_buf));
      }
      cl_buffers.push_back(selected_buffer);

      allocated_buffers.resize(buffer_assignment.object_sizes.size());
      size_t offset = 0;
      for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
        const size_t aligned_size =
            AlignByN(buffer_assignment.object_sizes[i], base_align_bytes);
        ABSL_ASSIGN_OR_RETURN(
            auto sub_buf,
            CreateSubBuffer(*selected_buffer, offset, aligned_size, context));
        sub_buffers_.push_back(std::make_unique<Buffer>(std::move(sub_buf)));
        allocated_buffers[i] = sub_buffers_.back().get();
        offset += aligned_size;
      }
    } else {
      std::vector<const Buffer*> available_buffers =
          GetSortedVector(shared_buffers_);
      allocated_buffers.resize(buffer_assignment.object_sizes.size());
      for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
        size_t size = buffer_assignment.object_sizes[i];
        const Buffer* selected_buffer =
            TakeAvailableBuffer(available_buffers, size);
        if (!selected_buffer) {
          auto buf = std::make_unique<Buffer>();
          Buffer shared_buffer;
          ABSL_RETURN_IF_ERROR(
              CreateReadWriteBuffer(size, context, &shared_buffer));
          *buf = std::move(shared_buffer);
          selected_buffer = buf.get();
          new_buffers.push_back(std::move(buf));
        }
        allocated_buffers[i] = selected_buffer;
      }
    }
  }

  for (auto& buf : new_buffers) {
    shared_buffers_.push_back(std::move(buf));
  }

  for (auto& usage : buffer_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const int tensor_index = value_id_to_shared_buffer_tensors[usage.first];
    const int buffer_index = use_offset_assignment
                                 ? tensor_index
                                 : buffer_assignment.object_ids[tensor_index];
    auto& tensor = value_id_to_buffer_[Key(model_id, usage.first)];
    if (td.GetStorageType() == TensorStorageType::TEXTURE_2D ||
        td.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D) {
      const size_t bytes_per_pixel =
          SizeOf(td.GetDataType()) * td.GetElementSize();
      const size_t width_pixel_alignment =
          GetWidthAlignment(gpu_info, bytes_per_pixel);
      ABSL_RETURN_IF_ERROR(CreateTensorSharedImage2DBuffer(
          *context, allocated_buffers[buffer_index]->GetMemoryPtr(), td,
          width_pixel_alignment, &tensor));
    } else {
      ABSL_RETURN_IF_ERROR(CreateTensorShared(
          *context, allocated_buffers[buffer_index]->GetMemoryPtr(), td,
          &tensor));
    }
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateTextureBasedTensors(
    ModelId model_id, const std::map<ValueId, int2>& texture_usages,
    const GpuModel& gpu_model, const GpuInfo& gpu_info, CLContext* context) {
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
    usage_records.push_back({{gpu_model.tensors.at(usage.first)},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::EQUALITY, &assignment));

  std::vector<const Tensor*> cl_textures(assignment.object_sizes.size(),
                                         nullptr);
  std::vector<std::unique_ptr<Tensor>> new_textures;

  std::vector<const Tensor*> available_textures =
      GetSortedVector(shared_texture_tensors_);

  for (int i = 0; i < assignment.object_sizes.size(); ++i) {
    const auto& td = assignment.object_sizes[i].tensor_desc;
    const Tensor* selected_tensor =
        TakeAvailableTexture(available_textures, td);
    if (selected_tensor) {
      cl_textures[i] = selected_tensor;
    } else {
      new_textures.push_back(std::make_unique<Tensor>());
      ABSL_RETURN_IF_ERROR(
          CreateTensor(*context, td, new_textures.back().get()));
      cl_textures[i] = new_textures.back().get();
    }
  }

  for (auto& tex : new_textures) {
    shared_texture_tensors_.push_back(std::move(tex));
  }

  for (auto& usage : texture_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const auto id = assignment.object_ids[remap_from_value_ids[usage.first]];
    ABSL_RETURN_IF_ERROR(
        CreateTensorShared(*context, cl_textures[id]->GetMemoryPtr(), td,
                           &value_id_to_texture_[Key(model_id, usage.first)]));
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::SetExternalTensor(const Key& key,
                                              Tensor* tensor_ptr) {
  auto it = external_mutable_tensors_.find(key);
  if (it == external_mutable_tensors_.end()) {
    return absl::InvalidArgumentError("No external tensor with this id.");
  }
  external_mutable_tensors_[key] = tensor_ptr;
  return absl::OkStatus();
}

Tensor* MemoryManager::GetTensor(Key key) {
  if (external_immutable_tensors_.find(key) !=
      external_immutable_tensors_.end()) {
    return external_immutable_tensors_[key];
  } else if (external_mutable_tensors_.find(key) !=
             external_mutable_tensors_.end()) {
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

uint64_t MemoryManager::GetSizeOfMemoryAllocatedForIntermediateTensors() const {
  uint64_t total_memory = 0;
  for (const auto& t : shared_texture_tensors_) {
    total_memory += t->GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    // Sub-buffers do not allocate memory. Count the size of the parent buffer
    // object instead.
    if (!b->IsSubBuffer()) {
      total_memory += b->GetMemorySizeInBytes();
    }
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

uint64_t MemoryManager::GetExternalTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& t : external_immutable_tensors_) {
    if (t.second) {
      total_size += t.second->GetMemorySizeInBytes();
    }
  }
  for (const auto& t : external_mutable_tensors_) {
    if (t.second) {
      total_size += t.second->GetMemorySizeInBytes();
    }
  }
  return total_size;
}

absl::Status GetTotalBufferSizeForTensors(
    const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors,
    const GpuInfo& gpu_info, uint64_t* result) {
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  ObjectsAssignment<size_t> buffer_assignment;
  OffsetsAssignment offset_assignment;
  bool use_offset_assignment;
  bool is_sub_buffers_supported;
  std::map<ValueId, int2> buffer_usages;
  std::map<ValueId, int2> texture_usages;
  CollectUsageInformation(gpu_model, gpu_info, external_tensors, &buffer_usages,
                          &texture_usages);
  ABSL_RETURN_IF_ERROR(GetBufferAssignment(
      buffer_usages, gpu_model, gpu_info, &buffer_usage_records, nullptr,
      &buffer_assignment, &offset_assignment, &use_offset_assignment,
      &is_sub_buffers_supported));
  if (use_offset_assignment) {
    *result = offset_assignment.total_size;
    return absl::OkStatus();
  }

  const size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);
  *result = TotalSize(buffer_assignment, base_align_bytes);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace ml_drift
