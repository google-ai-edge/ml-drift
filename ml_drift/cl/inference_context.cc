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

#include "ml_drift/cl/inference_context.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_command_buffer.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/command_buffer.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/memory_manager.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/program_cache.h"
#include "ml_drift/cl/qcom_command_buffer.h"
#include "ml_drift/cl/serialization_generated.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/ir_model_util.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/serialization_base.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {
namespace {
absl::Status ClarifyWithCommandBuffer(ProfilingCommandQueue* queue,
                                      int num_tries, double cb_duration_ms,
                                      const std::vector<CLNode*>& nodes,
                                      std::vector<double>* time_ns) {
  auto get_tasks_count = [&](int node_index) {
    const int tasks_count = cb_duration_ms / ((*time_ns)[node_index] * 1e-6f);
    return std::min(256, std::max(1, tasks_count));
  };

  std::vector<CLCommandBuffer> cbs(nodes.size() * num_tries);
  for (int t = 0; t < num_tries; ++t) {
    for (int node_index = 0; node_index < nodes.size(); ++node_index) {
      const int index = t * nodes.size() + node_index;
      auto& cb = cbs[index];
      ABSL_RETURN_IF_ERROR(cb.Init(queue, /*simultaneous_use=*/false));
      const int num_kernels_in_cb = get_tasks_count(node_index);
      for (int j = 0; j < num_kernels_in_cb; ++j) {
        ABSL_RETURN_IF_ERROR(nodes[node_index]->cl_operation.AddToCommandBuffer(
            cb.GetCommandBuffer()));
      }
      ABSL_RETURN_IF_ERROR(cb.Finalize());
    }
  }
  std::vector<CLEvent> events(nodes.size() * num_tries);
  for (int t = 0; t < num_tries; ++t) {
    for (int node_index = 0; node_index < nodes.size(); ++node_index) {
      const int index = t * nodes.size() + node_index;
      ABSL_RETURN_IF_ERROR(cbs[index].Enqueue(queue, &events[index]));
    }
  }
  clFinish(queue->queue());
  for (int node_index = 0; node_index < nodes.size(); ++node_index) {
    double min_time_ns = std::numeric_limits<double>::max();
    for (int t = 0; t < num_tries; ++t) {
      const int num_kernels_in_cb = get_tasks_count(node_index);
      double time_ns = events[t * nodes.size() + node_index].GetEventTimeNs() /
                       num_kernels_in_cb;
      min_time_ns = std::min(min_time_ns, time_ns);
    }
    (*time_ns)[node_index] = min_time_ns;
  }
  return absl::OkStatus();
}

size_t GetCommandBufferSize(const CLDevice& device) {
  if (IsQcomCommandBufferSupported(device)) {
    return GetQcomCommandBufferMaxSize(device);
  }
  return 32;
}

bool IsClCommandBufferRecommended(const CLDevice& device,
                                  const GpuInfo& gpu_info,
                                  bool allow_cl_khr_command_buffer) {
  if (IsQcomCommandBufferSupported(device)) {
    return true;
  }
  if (!allow_cl_khr_command_buffer) {
    return false;
  }
  if (!gpu_info.SupportsExtension("cl_khr_command_buffer") ||
      clCreateCommandBufferKHR == nullptr) {
    return false;
  }
  auto caps = GetDeviceInfo<cl_device_command_buffer_capabilities_khr>(
      device.id(), CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR);
  if (!(caps & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR)) {
    return false;
  }
  if (!gpu_info.IsPowerVR()) {
    return false;
  }

  if (gpu_info.IsPowerVR() &&
      !gpu_info.powervr_info.driver_version.IsHigherOrEqual(
          PowerVRInfo::DriverVersion{25, 3, 6908880})) {
    return false;
  }
  return true;
}

}  // namespace

CreateInferenceInfo::Options CreateFromModelHints(const ModelHints& hints) {
  CreateInferenceInfo::Options options;
  options.tuning_type = hints.Check(ModelHints::kFastTuning)
                            ? TuningType::kFast
                            : TuningType::kExhaustive;
  options.allow_cl_khr_command_buffer = hints.allow_cl_khr_command_buffer;
  return options;
}

void InferenceContext::ExecutionHints::Init(const GpuInfo& gpu_info) {
  if (gpu_info.IsMali()) {
    need_flush = true;
    need_manual_release =
        gpu_info.mali_info.generation < MaliInfo::Gen::kValhallV1 ? true
                                                                  : false;

    flush_periodically = true;
    flush_period = 24;
  }
  if (gpu_info.IsPowerVR()) {
    need_flush = true;
    flush_periodically = true;
    // Some Ge8xxx devices are slower without frequent periodic flushing.
    flush_period =
        gpu_info.powervr_info.IsBetterThan(PowerVRGpu::kRogueGm9xxx) ? 16 : 4;
  }
  // clvk has inside to know when to flush, do not do it at the application
  // level.
  if (gpu_info.IsApiOpenCl() && gpu_info.opencl_info.IsCLVK()) {
    need_flush = false;
    flush_periodically = false;
  }
}

void InferenceContext::EnableOptionalNodes(absl::flat_hash_set<int> tags) {
  enabled_tags_ = std::move(tags);
  UpdateActiveTiles();
}

absl::Status InferenceContext::InitFromGpuModel(
    const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
    Environment* env, std::vector<uint8_t>* serialized_model,
    Buffer* shared_buffer) {
  CreateInferenceInfo create_inference_info;
  create_inference_info.options = CreateFromModelHints(create_info.hints);
  create_inference_info.external_tensors.immutable_tensors =
      create_info.external_immutable_tensors;
  create_inference_info.external_tensors.mutable_tensors =
      create_info.external_mutable_tensors;
  return InitFromGpuModel(create_inference_info, gpu_model, env,
                          serialized_model, shared_buffer);
}

InferenceContext::InferenceContext(MemoryManager* memory_manager)
    : owned_memory_manager_(memory_manager == nullptr
                                ? std::make_unique<MemoryManager>()
                                : nullptr),
      memory_manager_(memory_manager == nullptr ? owned_memory_manager_.get()
                                                : memory_manager) {}

absl::Status InferenceContext::InitFromGpuModel(
    const CreateInferenceInfo& create_info, GpuModel* gpu_model,
    Environment* env, std::vector<uint8_t>* serialized_model,
    Buffer* shared_buffer) {
  gpu_info_ = env->device().GetInfo();
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = ml_drift::Encode(*gpu_model, &builder);
  }
  std::vector<std::unique_ptr<Tensor>>
      external_mutable_tensors_allocated_temporarily;
  ABSL_ASSIGN_OR_RETURN(
      model_id_,
      memory_manager_->AllocateMemory(
          *gpu_model, gpu_info_, create_info.external_tensors,
          external_mutable_tensors_allocated_temporarily, &env->context()));
  InitFromGpuModelInternal(gpu_model);

  GetMutableNodes(create_info.external_tensors.mutable_tensors);
  execution_hints_.Init(gpu_info_);
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.cl_operation.Compile(
        env->GetDevicePtr(), &env->context(), env->program_cache()));
  }
  ABSL_RETURN_IF_ERROR(BindMemoryToOperations());
  ABSL_RETURN_IF_ERROR(UpdateParams());

  TuningType tuning_type = create_info.options.tuning_type;
  if (gpu_info_.IsMali()) {
    const MaliInfo& info = gpu_info_.mali_info;
    if (info.IsMaliT6xx()) {
      // Mali T628 hangs forever in clFinish when used profiling queue
      // TuningType::FAST does not use profiling queue.
      tuning_type = TuningType::kFast;
    }
  }
  if (gpu_info_.IsPowerVR()) {
    const PowerVRInfo& info = gpu_info_.powervr_info;
    if (info.gpu_version >= PowerVRGpu::kCXT) {
      // PowerVR has weird behavior with kExhaustive tuning. b/397798684
      tuning_type = TuningType::kFast;
    }
  }
  ABSL_RETURN_IF_ERROR(Tune(tuning_type, gpu_info_, env->profiling_queue()));
  ABSL_RETURN_IF_ERROR(
      BuildExecutionPlan(env, create_info.options.allow_cl_khr_command_buffer));

  // Reset external tensors to nullptr as they are allocated temporarily.
  for (auto& external_tensor : create_info.external_tensors.mutable_tensors) {
    ABSL_RETURN_IF_ERROR(memory_manager_->SetExternalTensor(
        GetKey(external_tensor.first), nullptr));
  }

  if (serialized_model) {
    ABSL_ASSIGN_OR_RETURN(auto encoded_fb,
                          Encode(*env->GetDevicePtr(), *env->program_cache(),
                                 gpu_model_fb, &builder));
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AddToCommandBuffer(cl_command_buffer_khr cb) {
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.cl_operation.AddToCommandBuffer(cb));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::RestoreDeserialized(
    const absl::Span<const uint8_t> serialized_model, Environment* env,
    const CreateGpuModelInfo* create_info) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto decoded_fb = data::GetInferenceContext(serialized_model.data());
  std::string platform_version(decoded_fb->driver_version()->c_str(),
                               decoded_fb->driver_version()->size());
  if (env->GetDevicePtr()->GetPlatformVersion() != platform_version) {
    return absl::InvalidArgumentError(
        "OpenCL driver changed, model respresentation invalid, must be "
        "regenerated.");
  }
  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(ml_drift::Decode(decoded_fb->gpu_model(), &gpu_model));
  const auto& create_info_copy =
      create_info ? *create_info : CreateGpuModelInfo();
  ExternalTensorsInfo external_tensors;
  external_tensors.immutable_tensors =
      create_info_copy.external_immutable_tensors;
  external_tensors.mutable_tensors = create_info_copy.external_mutable_tensors;
  std::vector<std::unique_ptr<Tensor>>
      external_mutable_tensors_allocated_temporarily;
  ABSL_ASSIGN_OR_RETURN(
      model_id_,
      memory_manager_->AllocateMemory(
          gpu_model, env->GetDevicePtr()->GetInfo(), external_tensors,
          external_mutable_tensors_allocated_temporarily, &env->context()));
  InitFromGpuModelInternal(&gpu_model);

  // deserializing kernels into program_cache
  for (auto binary_program_fb : *decoded_fb->binary_programs()) {
    ABSL_RETURN_IF_ERROR(env->program_cache()->AddProgramBinary(
        env->context(), *env->GetDevicePtr(), binary_program_fb->fingerprint(),
        absl::MakeSpan(binary_program_fb->binary()->data(),
                       binary_program_fb->binary()->size())));
  }
  GetMutableNodes(create_info_copy.external_mutable_tensors);

  execution_hints_.Init(env->device().GetInfo());

  for (int i = 0; i < nodes_.size(); ++i) {
    uint64_t fingerprint = (*decoded_fb->fingerprints_per_node())[i];
    int3 wg_size;
    wg_size.x = (*decoded_fb->tuned_work_group_sizes_per_node())[i]->x();
    wg_size.y = (*decoded_fb->tuned_work_group_sizes_per_node())[i]->y();
    wg_size.z = (*decoded_fb->tuned_work_group_sizes_per_node())[i]->z();
    ABSL_RETURN_IF_ERROR(nodes_[i].cl_operation.RestoreDeserialized(
        *env->program_cache(), fingerprint, env->GetDevicePtr()->GetInfo(),
        wg_size, &env->context()));
  }
  ABSL_RETURN_IF_ERROR(BindMemoryToOperations());
  ABSL_RETURN_IF_ERROR(UpdateParams());
  ABSL_RETURN_IF_ERROR(BuildExecutionPlan(
      env, create_info_copy.hints.allow_cl_khr_command_buffer));
  for (auto& external_tensor : create_info_copy.external_mutable_tensors) {
    ABSL_RETURN_IF_ERROR(memory_manager_->SetExternalTensor(
        GetKey(external_tensor.first), nullptr));
  }
  return absl::OkStatus();
}

void InferenceContext::InitFromGpuModelInternal(GpuModel* gpu_model) {
  for (const auto& input : gpu_model->input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& output : gpu_model->output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  nodes_.resize(gpu_model->nodes.size());
  for (int i = 0; i < gpu_model->nodes.size(); ++i) {
    nodes_[i].cl_operation.Init(std::move(gpu_model->nodes[i].gpu_operation));
    nodes_[i].inputs = gpu_model->nodes[i].inputs;
    nodes_[i].outputs = gpu_model->nodes[i].outputs;
    nodes_[i].name = gpu_model->nodes[i].name;
    nodes_[i].op_name = gpu_model->nodes[i].op_name;
    nodes_[i].optional_tag = gpu_model->nodes[i].optional_tag;
  }
}

bool InferenceContext::IsMutable(const CLNode& node) const {
  for (ValueId id : node.inputs) {
    if (external_tensor_to_nodes_.contains(id)) {
      return true;
    }
  }
  for (ValueId id : node.outputs) {
    if (external_tensor_to_nodes_.contains(id)) {
      return true;
    }
  }
  return false;
}

void InferenceContext::UpdateActiveTiles() {
  active_tiles_.clear();
  for (int i = 0; i < execution_tiles_.size(); ++i) {
    auto& tile = execution_tiles_[i];
    if (absl::c_any_of(tile.optional_tag,
                       [&](int t) { return !enabled_tags_.contains(t); })) {
      continue;
    }
    active_tiles_.push_back(i);
  }
}

absl::Status InferenceContext::BuildExecutionPlan(
    Environment* env, bool allow_cl_khr_command_buffer) {
  if (!IsClCommandBufferRecommended(env->device(), gpu_info_,
                                    allow_cl_khr_command_buffer)) {
    for (int i = 0; i < nodes_.size(); ++i) {
      execution_tiles_.push_back(
          {ExecutionTile::Type::kKernel, i, nodes_[i].optional_tag});
    }
    UpdateActiveTiles();
    return absl::OkStatus();
  }

  const int max_tasks_per_command_buffer = GetCommandBufferSize(env->device());
  std::unique_ptr<CommandBuffer> current_cb = nullptr;
  absl::flat_hash_set<int> current_cb_tags;
  int current_cb_node_count = 0;

  auto finalize_current_cb = [&]() -> absl::Status {
    if (current_cb) {
      ABSL_RETURN_IF_ERROR(current_cb->Finalize());
      command_buffers_.push_back(std::move(current_cb));
      execution_tiles_.push_back({ExecutionTile::Type::kCommandBuffer,
                                  (int)command_buffers_.size() - 1,
                                  current_cb_tags});
      current_cb = nullptr;
      current_cb_node_count = 0;
      current_cb_tags.clear();
    }
    return absl::OkStatus();
  };

  auto start_new_cb =
      [&](const absl::flat_hash_set<int>& tags) -> absl::Status {
    if (IsQcomCommandBufferSupported(env->device())) {
      ABSL_ASSIGN_OR_RETURN(
          current_cb, CreateQcomCommandBuffer(env->device(), env->context()));
    } else {
      auto cl_cb = std::make_unique<CLCommandBuffer>();
      ABSL_RETURN_IF_ERROR(
          cl_cb->Init(env->queue(), /*simultaneous_use=*/true));
      current_cb = std::move(cl_cb);
    }
    current_cb_tags = tags;
    current_cb_node_count = 0;
    return absl::OkStatus();
  };

  for (int i = 0; i < nodes_.size(); ++i) {
    CLNode& node = nodes_[i];
    if (IsMutable(node)) {
      ABSL_RETURN_IF_ERROR(finalize_current_cb());
      execution_tiles_.push_back(
          {ExecutionTile::Type::kKernel, i, node.optional_tag});
    } else {
      if (!current_cb || node.optional_tag != current_cb_tags ||
          current_cb_node_count >= max_tasks_per_command_buffer) {
        ABSL_RETURN_IF_ERROR(finalize_current_cb());
        ABSL_RETURN_IF_ERROR(start_new_cb(node.optional_tag));
      }
      if (current_cb) {  // always true, compiler doesn't know
        ABSL_RETURN_IF_ERROR(current_cb->AddOp(&node.cl_operation));
      }
      current_cb_node_count++;
    }
  }
  ABSL_RETURN_IF_ERROR(finalize_current_cb());
  UpdateActiveTiles();

  return absl::OkStatus();
}

absl::Status InferenceContext::BindMemoryToOperations() {
  for (auto& node : nodes_) {
    for (int i = 0; i < node.inputs.size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          node.cl_operation.SetSrcTensor(i, GetTensor(node.inputs[i])));
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          node.cl_operation.SetDstTensor(i, GetTensor(node.outputs[i])));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Tune(TuningType tuning_type,
                                    const GpuInfo& gpu_info,
                                    ProfilingCommandQueue* profiling_queue) {
  // Cache tuned CL operations. Multiple CL operations might share the
  // same kernel but use different inputs, which might require different working
  // group setups. Therefore, we store a vector of tuned cl operations for each
  // kernel and match in a second stage based on equal CL arguments.
  typedef std::reference_wrapper<const ClOperation> ClOperationRef;
  absl::flat_hash_map<uint64_t, std::vector<ClOperationRef>> tuned_ops;
  for (auto& node : nodes_) {
    uint64_t fingerprint = node.cl_operation.GetKernelFingerprint();
    auto cl_ops_it = tuned_ops.find(fingerprint);
    bool found_cached_cl_op = false;
    if (cl_ops_it != tuned_ops.end()) {
      for (const auto& cl_op : cl_ops_it->second) {
        if (!node.cl_operation.HasEqualScalarArguments(cl_op)) {
          continue;
        }
        // Fingerprint and CLArguments match, so we reuse the work group size.
        node.cl_operation.SetWorkGroupSize(cl_op.get().GetWorkGroupSize());
        found_cached_cl_op = true;
      }
    }
    if (found_cached_cl_op) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(
        node.cl_operation.Tune(tuning_type, gpu_info, profiling_queue));
    tuned_ops[fingerprint].emplace_back(std::cref(node.cl_operation));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::UpdateParams() {
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.cl_operation.UpdateParams());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::SetTensor(const ValueId& tensor_id,
                                         Tensor* tensor_ptr) {
  ABSL_RETURN_IF_ERROR(
      memory_manager_->SetExternalTensor(GetKey(tensor_id), tensor_ptr));
  for (int node_index : external_tensor_to_nodes_[tensor_id]) {
    auto& node = nodes_[node_index];
    for (int i = 0; i < node.inputs.size(); ++i) {
      if (node.inputs[i] == tensor_id) {
        ABSL_RETURN_IF_ERROR(node.cl_operation.SetSrcTensor(i, tensor_ptr));
      }
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      if (node.outputs[i] == tensor_id) {
        ABSL_RETURN_IF_ERROR(node.cl_operation.SetDstTensor(i, tensor_ptr));
      }
    }
  }
  return absl::OkStatus();
}

void InferenceContext::GetMutableNodes(
    const absl::flat_hash_map<ValueId, TensorDescriptor>&
        external_mutable_tensors) {
  for (auto& external : external_mutable_tensors) {
    for (int i = 0; i < nodes_.size(); ++i) {
      bool has_tensor = false;
      const auto& src_ids = nodes_[i].inputs;
      for (int i = 0; !has_tensor && i < src_ids.size(); ++i) {
        if (src_ids[i] == external.first) {
          has_tensor = true;
        }
      }
      const auto& dst_ids = nodes_[i].outputs;
      for (int i = 0; !has_tensor && i < dst_ids.size(); ++i) {
        if (dst_ids[i] == external.first) {
          has_tensor = true;
        }
      }
      if (has_tensor) {
        external_tensor_to_nodes_[external.first].push_back(i);
      }
    }
  }
}

absl::Status InferenceContext::AddToQueue(const ExecutionTile& tile,
                                          CLCommandQueue* queue,
                                          CLEvent* event) {
  if (tile.type == ExecutionTile::Type::kKernel) {
    return nodes_[tile.index].cl_operation.AddToQueue(queue, event);
  } else {
    return command_buffers_[tile.index]->Enqueue(queue, event);
  }
}

absl::Status InferenceContext::AddToQueue(CLCommandQueue* queue,
                                          CLEvent* event) {
  if (execution_hints_.need_manual_release) {
    if (execution_hints_.prev_enqueue_start_point.is_valid()) {
      execution_hints_.prev_enqueue_start_point.Wait();
    }
    ABSL_RETURN_IF_ERROR(
        queue->EnqueueEvent(&execution_hints_.prev_enqueue_start_point));
  }
  for (int tile_idx = 0; tile_idx < active_tiles_.size(); ++tile_idx) {
    auto& step = execution_tiles_[active_tiles_[tile_idx]];
    CLEvent* current_event =
        (tile_idx == active_tiles_.size() - 1) ? event : nullptr;
    ABSL_RETURN_IF_ERROR(AddToQueue(step, queue, current_event));
    if (execution_hints_.flush_periodically &&
        (tile_idx + 1) % execution_hints_.flush_period == 0) {
      clFlush(queue->queue());
    }
  }
  if (execution_hints_.need_flush) {
    clFlush(queue->queue());
  }
  return absl::OkStatus();
}

absl::StatusOr<int> InferenceContext::AddToQueue(CLCommandQueue* queue,
                                                 int offset, int count,
                                                 CLEvent* event) {
  int i = offset;
  for (; i < offset + count && i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    if (absl::c_any_of(node.optional_tag,
                       [&](int t) { return !enabled_tags_.contains(t); })) {
      continue;
    }
    if (i == nodes_.size() - 1 || i == offset + count - 1) {
      ABSL_RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue, event));
    } else {
      ABSL_RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue));
    }
  }
  if (i == nodes_.size()) {
    return -1;
  } else {
    return i;
  }
}

absl::Status InferenceContext::ClarifyTimeWithCommandBuffer(
    ProfilingCommandQueue* queue, ProfilingInfo* result) {
  const int num_tries = 3;
  const double cb_duration_ms = 10.0;  // looks like enough
  const int node_group_count = 8;  // Current PowerVR drivers have issues with
                                   // big amount of CB or big CB.
  for (int node_index = 0; node_index < nodes_.size();
       node_index += node_group_count) {
    std::vector<CLNode*> nodes_to_clarify;
    std::vector<double> times_ns;
    for (int i = 0; i < node_group_count && node_index + i < nodes_.size();
         ++i) {
      nodes_to_clarify.push_back(&nodes_[node_index + i]);
      times_ns.push_back(absl::ToDoubleNanoseconds(
          result->dispatches[node_index + i].duration));
    }
    ABSL_RETURN_IF_ERROR(ClarifyWithCommandBuffer(
        queue, num_tries, cb_duration_ms, nodes_to_clarify, &times_ns));
    for (int i = 0; i < node_group_count && node_index + i < nodes_.size();
         ++i) {
      result->dispatches[node_index + i].duration =
          absl::Nanoseconds(times_ns[i]);
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::ClarifyTimeMultipleEnqueue(
    double ops_total_duration_ms, int min_ops, int max_ops,
    ProfilingCommandQueue* queue, ProfilingInfo* result) {
  queue->ResetMeasurements();
  for (int i = 0; i < nodes_.size(); ++i) {
    queue->SetEventsLabel(nodes_[i].name);
    const int times =
        ops_total_duration_ms /
        absl::ToDoubleMilliseconds(result->dispatches[i].duration);
    const int n = std::min(max_ops, std::max(min_ops, times));
    ABSL_RETURN_IF_ERROR(
        nodes_[i].cl_operation.AddToQueueForProfiling(queue, n));
  }
  ABSL_RETURN_IF_ERROR(queue->WaitForCompletion());
  *result = queue->GetProfilingInfo();
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfileTime(ProfilingCommandQueue* queue,
                                           ProfilingInfo* result) {
  queue->ResetMeasurements();
  for (auto& node : nodes_) {
    queue->SetEventsLabel(node.name);
    ABSL_RETURN_IF_ERROR(node.cl_operation.AddToQueueForProfiling(queue));
  }
  ABSL_RETURN_IF_ERROR(queue->WaitForCompletion());
  *result = queue->GetProfilingInfo();

  if (!(gpu_info_.IsMali() ||
        (gpu_info_.IsPowerVR() &&
         gpu_info_.powervr_info.gpu_version < PowerVRGpu::kCXT))) {
    return absl::OkStatus();
  }

  if (gpu_info_.IsMali()) {
    // On Mali Valhall Gen3 and later enqueue aborted if enqueue more than ~320k
    // kernels. clFinish doesn't help.
    // Tested on:
    //   Mali-G78 r0p0
    //   OpenCL 3.0 v1.r50p0-00eac0.23a3e850fb3835a51f4357e034c00a3d
    //   Mali-G710 r0p0
    //   OpenCL 3.0 v1.r50p0-00eac0.a6e879c4404a36cfd91acdf7c049b72d
    //   Mali-G715 r0p0
    //   OpenCL 3.0 v1.r50p0-00eac0.9d9ad95a795d1f11bf29f66a99b2792a
    //   Mali-G925-Immortalis MC12 r0p1
    //   OpenCL 3.0 v1.r49p1-03bet0.9a303f2b7f6a1b089a8ec9196cd0a8d4
    const bool is_valhall_gen3_or_later =
        gpu_info_.mali_info.generation >= MaliInfo::Gen::kValhallV3;
    const int max_enqueues_per_op =
        is_valhall_gen3_or_later ? ((320000 / nodes_.size()) / 2) : 256;
    return ClarifyTimeMultipleEnqueue(
        /*ops_total_duration_ms=*/16.0,
        /*min_ops=*/2, /*max_ops=*/max_enqueues_per_op, queue, result);
  }

  if (gpu_info_.IsPowerVR()) {
    if (gpu_info_.SupportsExtension("cl_khr_command_buffer")) {
      ABSL_RETURN_IF_ERROR(ClarifyTimeWithCommandBuffer(queue, result));
      ABSL_RETURN_IF_ERROR(ClarifyTimeWithCommandBuffer(queue, result));
    } else {
      ABSL_RETURN_IF_ERROR(ClarifyTimeMultipleEnqueue(
          /*ops_total_duration_ms=*/32.0,
          /*min_ops=*/4, /*max_ops=*/64, queue, result));
      return ClarifyTimeMultipleEnqueue(/*ops_total_duration_ms=*/128.0,
                                        /*min_ops=*/4, /*max_ops=*/1024, queue,
                                        result);
    }
  }

  return absl::OkStatus();
}

absl::Status InferenceContext::Profile(ProfilingCommandQueue* queue,
                                       ProfilingInfo* result) {
  ABSL_RETURN_IF_ERROR(ProfileTime(queue, result));
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& gpu_op = nodes_[i].cl_operation;
    auto& dispatch_info = result->dispatches[i];

    uint64_t read_size = 0;
    if (gpu_op.GetReadSize() != -1) {
      read_size = gpu_op.GetReadSize();
    } else {
      for (auto& src_id : nodes_[i].inputs) {
        read_size += GetTensor(src_id)->GetMemorySizeInBytes();
      }
      read_size += gpu_op.GetConstArgsSize();
    }

    uint64_t write_size = 0;
    if (gpu_op.GetWriteSize() != -1) {
      write_size = gpu_op.GetWriteSize();
    } else {
      for (auto& dst_id : nodes_[i].outputs) {
        write_size += GetTensor(dst_id)->GetMemorySizeInBytes();
      }
    }

    for (auto& src_id : nodes_[i].inputs) {
      dispatch_info.inshape.push_back(
          GetTensor(src_id)->GetDescriptor().GetBHWCShape().ToShape());
    }
    for (auto& dst_id : nodes_[i].outputs) {
      dispatch_info.outshape.push_back(
          GetTensor(dst_id)->GetDescriptor().GetBHWCShape().ToShape());
    }
    dispatch_info.flops = gpu_op.GetFlopsCount();
    dispatch_info.read_mem_size = read_size;
    dispatch_info.write_mem_size = write_size;
    dispatch_info.work_group_size = gpu_op.GetWorkGroupSize();
  }

  return absl::OkStatus();
}

bool InferenceContext::HasOptionalNode() const {
  return absl::c_any_of(
      nodes_, [](const auto& node) { return !node.optional_tag.empty(); });
}

uint64_t InferenceContext::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return memory_manager_->GetSizeOfMemoryAllocatedForIntermediateTensors();
}

uint64_t InferenceContext::GetConstantTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& node : nodes_) {
    total_size += node.cl_operation.GetConstArgsSize();
  }
  return total_size + memory_manager_->GetConstantTensorsSize();
}

uint64_t InferenceContext::GetExternalTensorsSize() const {
  return memory_manager_->GetExternalTensorsSize();
}

Tensor* InferenceContext::GetTensor(ValueId id) {
  return memory_manager_->GetTensor(GetKey(id));
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorFloat32& tensor,
                                              CLCommandQueue* queue) {
  Tensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  if (descriptor_with_data.GetDataType() != DataType::kFloat32 &&
      descriptor_with_data.GetDataType() != DataType::kFloat16) {
    return absl::InvalidArgumentError(
        absl::StrCat("SetInputTensor attempted to upload TensorFloat32 to the "
                     "tensor of type ",
                     descriptor_with_data.GetDataType(), "."));
  }
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, queue);
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorBool& tensor,
                                              CLCommandQueue* queue) {
  Tensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  if (descriptor_with_data.GetDataType() != DataType::kBool) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SetInputTensor attempted to upload TensorBool to the tensor of type ",
        descriptor_with_data.GetDataType(), "."));
  }
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, queue);
}

absl::Status InferenceContext::SetInputTensor(
    ValueId id, const ml_drift::Tensor<BHWC, DataType::kInt32>& tensor,
    CLCommandQueue* queue) {
  Tensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, queue);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               CLCommandQueue* queue,
                                               TensorFloat32* result) {
  const Tensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());

  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, queue));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               CLCommandQueue* queue,
                                               TensorBool* result) {
  const Tensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());

  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, queue));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               CLCommandQueue* queue,
                                               TensorInt32* result) {
  const Tensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());

  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, queue));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::StatusOr<flatbuffers::Offset<data::InferenceContext>>
InferenceContext::Encode(
    const CLDevice& device, const ProgramCache& program_cache,
    flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<ml_drift::data::Int3>> work_groups_fb;
  for (int i = 0; i < nodes_.size(); ++i) {
    auto work_group_fb =
        ml_drift::Encode(nodes_[i].cl_operation.GetWorkGroupSize(), builder);
    work_groups_fb.push_back(work_group_fb);
  }
  auto work_groups_fb_vec = builder->CreateVector(work_groups_fb);
  std::vector<uint64_t> node_fingerprints(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    node_fingerprints[i] = nodes_[i].cl_operation.GetKernelFingerprint();
  }
  auto node_fingerprints_fb = builder->CreateVector(node_fingerprints);

  std::set<uint64_t> fingerprints;
  for (const auto& node : nodes_) {
    fingerprints.insert(node.cl_operation.GetKernelFingerprint());
  }
  std::vector<flatbuffers::Offset<data::BinaryProgram>> binary_programs_fb;
  for (auto fingerprint : fingerprints) {
    std::vector<uint8_t> program_binary;
    ABSL_RETURN_IF_ERROR(
        program_cache.GetProgramBinary(fingerprint, &program_binary));
    auto binary_fb = builder->CreateVector(program_binary);
    data::BinaryProgramBuilder program_builder(*builder);
    program_builder.add_fingerprint(fingerprint);
    program_builder.add_binary(binary_fb);
    binary_programs_fb.push_back(program_builder.Finish());
  }
  auto binary_programs_fb_vec = builder->CreateVector(binary_programs_fb);
  auto driver_version = builder->CreateString(device.GetPlatformVersion());

  data::InferenceContextBuilder inf_builder(*builder);
  inf_builder.add_gpu_model(gpu_model_fb);
  inf_builder.add_driver_version(driver_version);
  inf_builder.add_binary_programs(binary_programs_fb_vec);
  inf_builder.add_tuned_work_group_sizes_per_node(work_groups_fb_vec);
  inf_builder.add_fingerprints_per_node(node_fingerprints_fb);
  return inf_builder.Finish();
}

absl::Status GetInOutRefs(const absl::Span<const uint8_t> serialized_model,
                          std::vector<int64_t>* in_refs,
                          std::vector<int64_t>* out_refs) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto fb_inference = data::GetInferenceContext(serialized_model.data());
  if (in_refs) {
    in_refs->clear();
    for (auto in_fb : *fb_inference->gpu_model()->input_refs()) {
      in_refs->push_back(in_fb);
    }
  }
  if (out_refs) {
    out_refs->clear();
    for (auto out_fb : *fb_inference->gpu_model()->output_refs()) {
      out_refs->push_back(out_fb);
    }
  }
  return absl::OkStatus();
}

absl::Status GraphToInferenceContext(const GpuInfo& gpu_info,
                                     const GraphFloat32& graph,
                                     Environment* env,
                                     CreateGpuModelInfo& create_info,
                                     InferenceContext* context,
                                     InferenceContext* weights_prep_context) {
  if (weights_prep_context != nullptr &&
      WeightsManager::IsGpuWeightsPreparationSupported(gpu_info)) {
    GpuModel main_gpu_model, gpu_weights_preparation_model;
    absl::flat_hash_map<ValueId, ValueId> weights_mapping;
    std::vector<WeightsManager::UploadWeightsInfo> upload_weights_info;

    ABSL_RETURN_IF_ERROR(GraphToGpuModelWithWeightsConversion(
        graph, create_info, gpu_info, &main_gpu_model,
        &gpu_weights_preparation_model, &weights_mapping,
        &upload_weights_info));
    if (!gpu_weights_preparation_model.nodes.empty()) {
      ABSL_RETURN_IF_ERROR(weights_prep_context->InitFromGpuModel(
          create_info, &gpu_weights_preparation_model, env));
      for (const auto& upload_info : upload_weights_info) {
        auto tensor = weights_prep_context->GetTensor(upload_info.input_id);
        ABSL_RETURN_IF_ERROR(
            tensor->WriteData(static_cast<const uint8_t*>(upload_info.data),
                              env->queue(), /*async=*/true));
      }
      ABSL_RETURN_IF_ERROR(weights_prep_context->AddToQueue(env->queue()));
      for (const auto& [converted_weight_id, main_model_weight_id] :
           weights_mapping) {
        auto tensor = weights_prep_context->GetTensor(converted_weight_id);
        create_info.external_immutable_tensors.insert(
            {main_model_weight_id, tensor});
      }
    }
    ABSL_RETURN_IF_ERROR(
        context->InitFromGpuModel(create_info, &main_gpu_model, env));
  } else {
    GpuModel gpu_model;
    ABSL_RETURN_IF_ERROR(
        GraphToGpuModel(graph, create_info, gpu_info, &gpu_model));
    ABSL_RETURN_IF_ERROR(
        context->InitFromGpuModel(create_info, &gpu_model, env));
  }
  return absl::OkStatus();
}

absl::Status IrModelToInferenceContext(const GpuInfo& gpu_info,
                                       const ir::IrModel& ir_model,
                                       Environment* env,
                                       CreateGpuModelInfo& create_info,
                                       InferenceContext* context,
                                       InferenceContext* weights_prep_context) {
  if (weights_prep_context != nullptr &&
      WeightsManager::IsGpuWeightsPreparationSupported(gpu_info)) {
    GpuModel main_gpu_model, gpu_weights_preparation_model;
    absl::flat_hash_map<ValueId, ValueId> weights_mapping;
    std::vector<WeightsManager::UploadWeightsInfo> upload_weights_info;

    ABSL_RETURN_IF_ERROR(IrModelToGpuModelWithWeightsConversion(
        ir_model, create_info, gpu_info, &main_gpu_model,
        &gpu_weights_preparation_model, &weights_mapping,
        &upload_weights_info));
    if (!gpu_weights_preparation_model.nodes.empty()) {
      ABSL_RETURN_IF_ERROR(weights_prep_context->InitFromGpuModel(
          create_info, &gpu_weights_preparation_model, env));
      for (const auto& upload_info : upload_weights_info) {
        auto tensor = weights_prep_context->GetTensor(upload_info.input_id);
        ABSL_RETURN_IF_ERROR(
            tensor->WriteData(static_cast<const uint8_t*>(upload_info.data),
                              env->queue(), /*async=*/true));
      }
      ABSL_RETURN_IF_ERROR(weights_prep_context->AddToQueue(env->queue()));
      for (const auto& [converted_weight_id, main_model_weight_id] :
           weights_mapping) {
        auto tensor = weights_prep_context->GetTensor(converted_weight_id);
        create_info.external_immutable_tensors.insert(
            {main_model_weight_id, tensor});
      }
    }
    ABSL_RETURN_IF_ERROR(
        context->InitFromGpuModel(create_info, &main_gpu_model, env));
  } else {
    GpuModel gpu_model;
    ABSL_RETURN_IF_ERROR(
        IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
    ABSL_RETURN_IF_ERROR(
        context->InitFromGpuModel(create_info, &gpu_model, env));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace ml_drift
