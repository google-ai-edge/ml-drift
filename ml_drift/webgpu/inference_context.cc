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

#include "ml_drift/webgpu/inference_context.h"

#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/ir_model_util.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/serialization_base.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/memory_manager.h"
#include "ml_drift/webgpu/metrics_collector_internal.h"
#include "ml_drift/webgpu/serialization_generated.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {

absl::StatusOr<std::vector<WebGpuNode>> CopyNodes(
    const Environment& env, std::vector<GpuNode>& gpu_nodes,
    bool from_serialized_model = false) {
  std::vector<WebGpuNode> nodes(gpu_nodes.size());
  for (int i = 0; i < nodes.size(); ++i) {
    if (gpu_nodes[i].gpu_operation) {
      RETURN_IF_ERROR(nodes[i].webgpu_operation.Init(
          env, std::move(gpu_nodes[i].gpu_operation),
          env.GetUniformBufferCreator(), from_serialized_model));
    }
    nodes[i].inputs = gpu_nodes[i].inputs;
    nodes[i].outputs = gpu_nodes[i].outputs;
    nodes[i].name = gpu_nodes[i].name;
    nodes[i].optional_tag = gpu_nodes[i].optional_tag;
    nodes[i].subgraph_id = gpu_nodes[i].subgraph_id;
  }
  return nodes;
}

}  // namespace

InferenceContext::InferenceContext()
    : owned_memory_manager_(std::make_unique<MemoryManager>()),
      memory_manager_(*owned_memory_manager_) {}

InferenceContext::InferenceContext(MemoryManager* memory_manager)
    : memory_manager_(*memory_manager) {}

void InferenceContext::EnableOptionalNodes(absl::flat_hash_set<int> tags) {
  enabled_tags_ = std::move(tags);
}

absl::Status InferenceContext::InitFromGpuModel(const Environment& env,
                                                GpuModel& gpu_model,
                                                bool from_serialized_model) {
  for (const auto& input : gpu_model.input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& output : gpu_model.output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  ASSIGN_OR_RETURN(nodes_,
                   CopyNodes(env, gpu_model.nodes, from_serialized_model));
  return absl::OkStatus();
}

// Encodes the WebGpuProgram to a flatbuffer.
flatbuffers::Offset<data::WebGpuProgram> EncodeProgram(
    const WebGpuNode& node, flatbuffers::FlatBufferBuilder& builder) {
  auto code_fb = builder.CreateString(node.webgpu_operation.GetCode());

  std::vector<uint8_t> const_args;
  data::WebGpuProgramBuilder program_builder(builder);
  program_builder.add_code(code_fb);
  program_builder.add_fingerprint(node.webgpu_operation.GetKernelFingerprint());
  return program_builder.Finish();
}

// Decodes the WebGpuProgram from a flatbuffer.
void DecodeProgram(const data::WebGpuProgram& fb_program, std::string& code,
                   uint64_t& fingerprint) {
  code = std::string(fb_program.code()->c_str(), fb_program.code()->size());
  fingerprint = fb_program.fingerprint();
}

flatbuffers::Offset<data::InferenceContext> InferenceContext::Encode(
    const Environment& env,
    flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder& builder) {
  std::vector<flatbuffers::Offset<ml_drift::data::Int3>> work_groups_fb;
  work_groups_fb.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    auto work_group_fb =
        ml_drift::Encode(node.webgpu_operation.GetWorkGroupSize(), &builder);
    work_groups_fb.push_back(std::move(work_group_fb));
  }
  auto work_groups_fb_vec = builder.CreateVector(work_groups_fb);

  std::vector<uint64_t> fingerprints_per_node;
  absl::flat_hash_map<uint64_t, int> fingerprint_to_first_index;
  fingerprints_per_node.reserve(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    auto fingerprint = node.webgpu_operation.GetKernelFingerprint();
    if (fingerprint_to_first_index.find(fingerprint) ==
        fingerprint_to_first_index.end()) {
      fingerprint_to_first_index[fingerprint] = i;
    }
    fingerprints_per_node.push_back(fingerprint);
  }
  auto fingerprints_per_node_vec = builder.CreateVector(fingerprints_per_node);

  std::vector<flatbuffers::Offset<data::WebGpuProgram>> webgpu_programs_fb;
  webgpu_programs_fb.reserve(fingerprint_to_first_index.size());

  for (auto [_, index] : fingerprint_to_first_index) {
    auto webgpu_program_fb = EncodeProgram(nodes_[index], builder);
    webgpu_programs_fb.push_back(std::move(webgpu_program_fb));
  }
  auto webgpu_programs_fb_vec = builder.CreateVector(webgpu_programs_fb);
  auto platform_description_fb =
      builder.CreateString(env.GetPlatformDescription());
  data::InferenceContextBuilder inference_context_builder(builder);
  inference_context_builder.add_gpu_model(gpu_model_fb);
  inference_context_builder.add_tuned_work_group_sizes_per_node(
      work_groups_fb_vec);
  inference_context_builder.add_webgpu_programs(webgpu_programs_fb_vec);
  inference_context_builder.add_fingerprints_per_node(
      fingerprints_per_node_vec);
  inference_context_builder.add_platform_description(platform_description_fb);
  return inference_context_builder.Finish();
}

absl::Status InferenceContext::RestoreDeserialized(
    absl::Span<const uint8_t> serialized_model, const Environment& env,
    const CreateGpuModelInfo* create_info) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto decoded_fb = data::GetInferenceContext(serialized_model.data());

  std::string platform_description(decoded_fb->platform_description()->c_str(),
                                   decoded_fb->platform_description()->size());
  if (env.GetPlatformDescription() != platform_description) {
    return absl::DataLossError(
        "Platform description does not match the current platform.");
  }
  GpuModel gpu_model;
  RETURN_IF_ERROR(ml_drift::Decode(decoded_fb->gpu_model(), &gpu_model));
  RETURN_IF_ERROR(
      InitFromGpuModel(env, gpu_model, /*from_serialized_model=*/true));
  const auto& create_info_copy =
      create_info != nullptr ? *create_info : CreateGpuModelInfo();
  ASSIGN_OR_RETURN(auto external_mutable_tensors_allocated_temporarily,
                   UpdateMutableObjects(env, create_info_copy, &gpu_model));

  absl::flat_hash_map<std::string, std::vector<WebGpuNode>> subgraphs;
  for (auto& [id, subgraph] : gpu_model.subgraphs) {
    ASSIGN_OR_RETURN(subgraphs[id], CopyNodes(env, subgraph.nodes,
                                              /*from_serialized_model=*/true));
  }
  RETURN_IF_ERROR(InitSubgraphNodes(env, std::move(subgraphs)));

  RETURN_IF_ERROR(BindGpuMemory(env));
  // Compile must be called after BindGpuMemory
  RETURN_IF_ERROR(CompileForSerializedModel(env, *decoded_fb));

  // Reset external tensors to nullptr as they are allocated temporarily.
  for (auto& external_tensor : create_info_copy.external_mutable_tensors) {
    RETURN_IF_ERROR(memory_manager_.SetExternalTensor(
        GetKey(external_tensor.first), nullptr));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<SpatialTensor>>>
InferenceContext::UpdateMutableObjects(const Environment& env,
                                       const CreateGpuModelInfo& create_info,
                                       GpuModel* gpu_model) {
  for (int node_index = 0; node_index < nodes_.size(); ++node_index) {
    const auto& node = nodes_[node_index];
    std::vector<ValueId> ids = node.inputs;
    ids.insert(ids.end(), node.outputs.begin(), node.outputs.end());
    bool incomplete_node = false;
    for (int id_index = 0; id_index < ids.size(); ++id_index) {
      const auto& id = ids[id_index];
      if (create_info.external_mutable_tensors.find(id) !=
          create_info.external_mutable_tensors.end()) {
        incomplete_node = true;
        mutable_objects_[id].push_back(
            {.node_index = node_index, .object_index = id_index});
      }
    }
    if (incomplete_node) {
      mutable_node_indexes_.push_back(node_index);
    }
  }
  ExternalTensorsInfo external_tensors;
  external_tensors.immutable_tensors = create_info.external_immutable_tensors;
  external_tensors.mutable_tensors = create_info.external_mutable_tensors;
  std::vector<std::unique_ptr<SpatialTensor>> external_mutable_tensors;
  ASSIGN_OR_RETURN(model_id_, memory_manager_.AllocateMemory(
                                  env, *gpu_model, external_tensors,
                                  external_mutable_tensors));
  return external_mutable_tensors;
}

absl::Status InferenceContext::InitFromGpuModel(
    const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  return InitFromGpuModel(*env, create_info, gpu_model, serialized_model);
}

absl::Status InferenceContext::InitFromGpuModel(
    const Environment& env, const CreateGpuModelInfo& create_info,
    GpuModel* gpu_model, std::vector<uint8_t>* serialized_model) {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = ml_drift::Encode(*gpu_model, &builder);
  }
  RETURN_IF_ERROR(
      InitFromGpuModel(env, *gpu_model, /*from_serialized_model=*/false));

  absl::flat_hash_map<std::string, std::vector<WebGpuNode>> subgraphs;
  for (auto& [id, subgraph] : gpu_model->subgraphs) {
    ASSIGN_OR_RETURN(subgraphs[id], CopyNodes(env, subgraph.nodes));
  }
  ASSIGN_OR_RETURN(auto external_mutable_tensors_allocated_temporarily,
                   UpdateMutableObjects(env, create_info, gpu_model));

  TuningType tuning_type = TuningType::kExhaustive;
  if (create_info.hints.Check(ModelHints::kFastTuning)) {
    tuning_type = TuningType::kFast;
  }
  RETURN_IF_ERROR(InitSubgraphNodes(env, std::move(subgraphs), &tuning_type));
  RETURN_IF_ERROR(BindGpuMemory(env));
  // Compile must be called after BindGpuMemory
  RETURN_IF_ERROR(Compile(env, tuning_type));

  // Reset external tensors to nullptr as they are allocated temporarily.
  for (auto& external_tensor : create_info.external_mutable_tensors) {
    RETURN_IF_ERROR(memory_manager_.SetExternalTensor(
        GetKey(external_tensor.first), nullptr));
  }

  if (serialized_model != nullptr) {
    auto inference_context_fb = Encode(env, gpu_model_fb, builder);
    data::FinishInferenceContextBuffer(builder, inference_context_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::BindGpuMemory(const Environment& environment) {
  for (int node_index = 0; node_index < nodes_.size(); node_index++) {
    auto& node = nodes_[node_index];
    for (int i = 0; i < node.inputs.size(); ++i) {
      SpatialTensor* tensor =
          memory_manager_.GetSpatialTensor(GetKey(node.inputs[i]));
      RETURN_IF_ERROR(node.webgpu_operation.SetSrcTensor(i, tensor));
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      SpatialTensor* tensor =
          memory_manager_.GetSpatialTensor(GetKey(node.outputs[i]));
      RETURN_IF_ERROR(node.webgpu_operation.SetDstTensor(i, tensor));
    }
    RETURN_IF_ERROR(node.webgpu_operation.Update(environment.device()));
    if (std::find(mutable_node_indexes_.begin(), mutable_node_indexes_.end(),
                  node_index) == mutable_node_indexes_.end()) {
      node.webgpu_operation.UpdateGpuObjectBindings(environment.device());
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Compile(const Environment& env,
                                       TuningType tuning_type) {
  for (auto& node : nodes_) {
    if (!node.subgraph_id.empty()) {
      continue;
    }
    RETURN_IF_ERROR(node.webgpu_operation.TuneAndCompile(
        env, tuning_type, env.GetComputePipelineCache()));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::CompileForSerializedModel(
    const Environment& env, const data::InferenceContext& decoded_fb) {
  absl::flat_hash_map<uint64_t, int> fingerprint_to_webgpu_flatbuffer_index;
  for (int i = 0; i < decoded_fb.webgpu_programs()->size(); ++i) {
    uint64_t fingerprint = (*decoded_fb.webgpu_programs())[i]->fingerprint();
    fingerprint_to_webgpu_flatbuffer_index[fingerprint] = i;
  }

  for (int i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    std::string code;
    uint64_t fingerprint = (*decoded_fb.fingerprints_per_node())[i];
    uint32_t webgpu_flatbuffer_index =
        fingerprint_to_webgpu_flatbuffer_index[fingerprint];

    DecodeProgram(*(*decoded_fb.webgpu_programs())[webgpu_flatbuffer_index],
                  code, fingerprint);
    const int3 wg_size((*decoded_fb.tuned_work_group_sizes_per_node())[i]->x(),
                       (*decoded_fb.tuned_work_group_sizes_per_node())[i]->y(),
                       (*decoded_fb.tuned_work_group_sizes_per_node())[i]->z());
    node.webgpu_operation.SetWorkGroupSize(wg_size);
    RETURN_IF_ERROR(node.webgpu_operation.RestoreDeserialized(
        env, code, fingerprint, env.GetComputePipelineCache()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<wgpu::CommandBuffer>>
InferenceContext::CreateCommandBuffers(
    const Environment& environment, int num_nodes_per_encoder,
    std::vector<CommandBufferInfo>* command_buffer_infos,
    bool submit_command_buffers) {
  RETURN_IF_ERROR(UpdateMutableObjectsBindings(environment));
  std::vector<wgpu::CommandBuffer> command_buffers;
  // Process nodes in batches since executing all the nodes in the same command
  // buffer can cause hangs on some platforms.
  for (int i = 0; i < nodes_.size();) {
    wgpu::CommandEncoder encoder = environment.device().CreateCommandEncoder();
    wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
    CommandBufferInfo command_buffer_info;
    for (int j = 0; j < num_nodes_per_encoder && i < nodes_.size(); i++, j++) {
      if (absl::c_any_of(nodes_[i].optional_tag,
                         [&](int t) { return !enabled_tags_.contains(t); })) {
        continue;
      }
      RETURN_IF_ERROR(nodes_[i].webgpu_operation.Encode(compute_encoder));
      if (command_buffer_infos) {
        command_buffer_info.src_tensors.insert(nodes_[i].inputs.begin(),
                                               nodes_[i].inputs.end());
      }
    }
    compute_encoder.End();
    auto command_buffer = encoder.Finish();
    if (submit_command_buffers) {
      environment.queue().Submit(1, &command_buffer);
    } else {
      command_buffers.push_back(std::move(command_buffer));
    }
    if (command_buffer_infos) {
      command_buffer_infos->push_back(command_buffer_info);
    }
  }
  return command_buffers;
}

absl::Status InferenceContext::Execute(const Environment& environment) {
  return AddToQueue(environment);
}

absl::Status InferenceContext::AddToQueue(const Environment& environment) {
  // At least Adreno 730 and some Intel GPUs fail sometimes with 512(or more)
  // nodes per encoder.  Haven't observed perf drop with 64 nodes per encoder.
  ASSIGN_OR_RETURN(
      auto command_buffers,
      CreateCommandBuffers(environment, /*num_nodes_per_encoder=*/64));
  for (auto& cb : command_buffers) {
    // Submit every buffer separately to avoid hangs, see CreateCommandBuffers.
    environment.queue().Submit(1, &cb);
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfilingAddToQueue(
    const Environment& environment,
    InternalMetricsCollector* metrics_collector) {
  std::vector<std::string> names(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    names[i] = nodes_[i].name;
  }
  RETURN_IF_ERROR(metrics_collector->Init(names));
  RETURN_IF_ERROR(UpdateMutableObjectsBindings(environment));
  wgpu::CommandEncoder encoder = environment.device().CreateCommandEncoder();
  RETURN_IF_ERROR(ProfilingEncode(&encoder, metrics_collector));
  wgpu::CommandBuffer cb = encoder.Finish();
  environment.queue().Submit(1, &cb);
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfileWithTimestamps(
    const Environment& env, ProfilingInfo* result,
    const std::vector<int>& ops_per_encoder, int node_first_index,
    int node_last_index) {
  const int nodes_count = node_last_index - node_first_index + 1;
  const int query_count = 2 * nodes_count;
  wgpu::QuerySetDescriptor query_set_descriptor;
  query_set_descriptor.type = wgpu::QueryType::Timestamp;
  query_set_descriptor.count = query_count;
  wgpu::QuerySet query_set = env.device().CreateQuerySet(&query_set_descriptor);
  std::vector<wgpu::PassTimestampWrites> timestamp_writes(nodes_count);
  for (uint32_t i = 0; i < nodes_count; ++i) {
    timestamp_writes[i] = {.querySet = query_set,
                           .beginningOfPassWriteIndex = i * 2u,
                           .endOfPassWriteIndex = i * 2u + 1u};
  }
  wgpu::Buffer query_buffer;
  {
    wgpu::BufferDescriptor buffer_descriptor;
    buffer_descriptor.size = query_count * sizeof(uint64_t);
    buffer_descriptor.usage =
        wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc |
        wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage;
    query_buffer = env.device().CreateBuffer(&buffer_descriptor);
  }
  wgpu::CommandEncoder encoder = env.device().CreateCommandEncoder();
  for (int i = 0; i < nodes_count; ++i) {
    wgpu::ComputePassDescriptor compute_pass_descriptor;
    compute_pass_descriptor.timestampWrites = &timestamp_writes[i];

    wgpu::ComputePassEncoder compute_encoder =
        encoder.BeginComputePass(&compute_pass_descriptor);
    const int ops_count = i < ops_per_encoder.size() ? ops_per_encoder[i] : 1;
    for (int j = 0; j < ops_count; ++j) {
      RETURN_IF_ERROR(nodes_[i + node_first_index].webgpu_operation.Encode(
          compute_encoder));
    }
    compute_encoder.End();
  }
  encoder.ResolveQuerySet(query_set, /*firstQuery=*/0, query_count,
                          query_buffer,
                          /*destinationOffset=*/0);
  wgpu::CommandBuffer cb = encoder.Finish();
  env.queue().Submit(1, &cb);
  RETURN_IF_ERROR(
      WaitUntilCompleted(env.queue(), env.device(), absl::Seconds(10)));
  std::vector<uint64_t> query_data(query_count);
  RETURN_IF_ERROR(ReadDataFromBuffer(env.device(), env.queue(), query_buffer,
                                     query_buffer.GetSize(),
                                     query_data.data()));
  for (int i = 0; i < nodes_count; ++i) {
    const double duration_ns = query_data[i * 2 + 1] - query_data[i * 2];
    const int ops_count = i < ops_per_encoder.size() ? ops_per_encoder[i] : 1;
    result->dispatches[i + node_first_index].duration =
        absl::Nanoseconds(duration_ns / ops_count);
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Profile(const Environment& env,
                                       ProfilingInfo* result) {
  result->dispatches.resize(nodes_.size());
  if (env.GetInfo().webgpu_info.supports_timestamp_query) {
    // warming up
    {
      const int batch_size = 64;
      for (int i = 0; i < nodes_.size(); i += batch_size) {
        RETURN_IF_ERROR(ProfileWithTimestamps(
            env, result, {}, i,
            std::min(i + batch_size - 1, static_cast<int>(nodes_.size()) - 1)));
      }
      for (int i = 0; i < nodes_.size(); i += batch_size) {
        RETURN_IF_ERROR(ProfileWithTimestamps(
            env, result, {}, i,
            std::min(i + batch_size - 1, static_cast<int>(nodes_.size()) - 1)));
      }
    }
    // clarification profiling, can report smaller time than actual. Runs the
    // same op multiple time, op objects can stay in cache and execute faster
    // than normal. Multiple ops in single ComputePassEncoder also don't have
    // any barriers/synchronizations.
    const int batch_size = 16;
    for (int i = 0; i < nodes_.size(); i += batch_size) {
      std::vector<int> ops_per_encoder(batch_size);
      for (int j = 0; j < batch_size; ++j) {
        if (i + j >= nodes_.size()) {
          break;
        }
        // Values below can be tuned per device.
        const double max_per_op_time_ms = 32.0;
        const int min_ops_per_encoder = 1;
        const int max_ops_per_encoder = 256;
        const double duration_ms =
            absl::ToDoubleMilliseconds(result->dispatches[i + j].duration);
        ops_per_encoder[j] = max_per_op_time_ms / duration_ms;
        ops_per_encoder[j] = std::max(ops_per_encoder[j], min_ops_per_encoder);
        ops_per_encoder[j] = std::min(ops_per_encoder[j], max_ops_per_encoder);
      }
      RETURN_IF_ERROR(ProfileWithTimestamps(
          env, result, ops_per_encoder, i,
          std::min(i + batch_size - 1, static_cast<int>(nodes_.size()) - 1)));
    }
  } else {
    for (int k = 0; k < nodes_.size(); ++k) {
      result->dispatches[k].duration =
          nodes_[k].webgpu_operation.GetOperationTime(env);
    }
  }
  for (int k = 0; k < nodes_.size(); ++k) {
    auto& gpu_op = nodes_[k].webgpu_operation;
    auto& dispatch_info = result->dispatches[k];

    dispatch_info.label = nodes_[k].name;

    uint64_t read_size = 0;
    if (gpu_op.GetReadSize() != -1) {
      read_size = gpu_op.GetReadSize();
    } else {
      for (auto& src_id : nodes_[k].inputs) {
        read_size += memory_manager_.GetSpatialTensor(GetKey(src_id))
                         ->GetMemorySizeInBytes();
      }
      read_size += gpu_op.GetConstArgsSize();
    }

    uint64_t write_size = 0;
    if (gpu_op.GetWriteSize() != -1) {
      write_size = gpu_op.GetWriteSize();
    } else {
      for (auto& dst_id : nodes_[k].outputs) {
        write_size += memory_manager_.GetSpatialTensor(GetKey(dst_id))
                          ->GetMemorySizeInBytes();
      }
    }

    for (auto& src_id : nodes_[k].inputs) {
      dispatch_info.inshape.push_back(
            memory_manager_.GetSpatialTensor(GetKey(src_id))
                ->GetDescriptor()
                .GetBHWCShape()
                .ToShape());
    }
    for (auto& dst_id : nodes_[k].outputs) {
      dispatch_info.outshape.push_back(
            memory_manager_.GetSpatialTensor(GetKey(dst_id))
                ->GetDescriptor()
                .GetBHWCShape()
                .ToShape());
    }
    dispatch_info.flops = gpu_op.GetFlopsCount();
    dispatch_info.read_mem_size = read_size;
    dispatch_info.write_mem_size = write_size;
    dispatch_info.work_group_size = gpu_op.GetWorkGroupSize();
  }
  return absl::OkStatus();
}

// for profiling and memory statistics
uint64_t InferenceContext::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return memory_manager_.GetSizeOfMemoryAllocatedForIntermediateTensors();
}

uint64_t InferenceContext::GetSizeOfMemoryAllocatedForConstTensors() const {
  uint64_t total_size =
      memory_manager_.GetSizeOfMemoryAllocatedForConstTensors();
  for (const auto& node : nodes_) {
    total_size += node.webgpu_operation.GetGpuOperation().const_args_size_;
  }
  return total_size;
}

absl::Status InferenceContext::SetTensor(const ValueId& tensor_id,
                                         SpatialTensor* tensor) {
  updated_mutable_objects_.push_back(tensor_id);
  return memory_manager_.SetExternalTensor(GetKey(tensor_id), tensor);
}

absl::Status InferenceContext::SetInputTensor(const Environment& environment,
                                              ValueId id,
                                              const TensorFloat32& tensor) {
  SpatialTensor* gpu_tensor = memory_manager_.GetSpatialTensor(GetKey(id));
  if (!gpu_tensor) {
    return absl::InternalError(absl::StrCat("Can not find tensor.", id));
  }
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(environment.queue(),
                                          descriptor_with_data);
}

absl::Status InferenceContext::SetInputTensor(
    const Environment& environment, ValueId id,
    const ml_drift::Tensor<BHWC, DataType::INT32>& tensor) {
  SpatialTensor* gpu_tensor = memory_manager_.GetSpatialTensor(GetKey(id));
  if (!gpu_tensor) {
    return absl::InternalError(absl::StrCat("Can not find tensor.", id));
  }
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(environment.queue(),
                                          descriptor_with_data);
}

absl::Status InferenceContext::GetOutputTensor(const Environment& environment,
                                               ValueId id,
                                               TensorFloat32* result) {
  SpatialTensor* gpu_tensor = memory_manager_.GetSpatialTensor(GetKey(id));
  if (!gpu_tensor) {
    return absl::InternalError(absl::StrCat("Can not find tensor.", id));
  }
  TensorDescriptor desc;
  RETURN_IF_ERROR(gpu_tensor->ToDescriptor(environment.device(), &desc));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::Status InferenceContext::GetOutputTensor(const Environment& environment,
                                               ValueId id,
                                               TensorInt32* result) {
  SpatialTensor* gpu_tensor = memory_manager_.GetSpatialTensor(GetKey(id));
  if (!gpu_tensor) {
    return absl::InternalError(absl::StrCat("Can not find tensor.", id));
  }
  TensorDescriptor desc;
  RETURN_IF_ERROR(gpu_tensor->ToDescriptor(environment.device(), &desc));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfilingEncode(
    wgpu::CommandEncoder* encoder,
    InternalMetricsCollector* metrics_collector) {
  if (metrics_collector->IsSingleCommandBuffer()) {
    RETURN_IF_ERROR(
        ProfilingEncodeSingleCommandBuffer(encoder, metrics_collector));
  } else {
    RETURN_IF_ERROR(
        ProfilingEncodePerOpCommandBuffer(encoder, metrics_collector));
  }
  metrics_collector->ResolveQueries(encoder);
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfilingEncodeSingleCommandBuffer(
    wgpu::CommandEncoder* encoder,
    InternalMetricsCollector* metrics_collector) {
  wgpu::ComputePassDescriptor descriptor =
      metrics_collector->GetComputePassDescriptor(0);
  wgpu::ComputePassEncoder compute_encoder =
      encoder->BeginComputePass(&descriptor);
  for (const auto& node : nodes_) {
    RETURN_IF_ERROR(node.webgpu_operation.Encode(compute_encoder));
  }
  compute_encoder.End();
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfilingEncodePerOpCommandBuffer(
    wgpu::CommandEncoder* encoder,
    InternalMetricsCollector* metrics_collector) {
  for (int i = 0; i < nodes_.size(); ++i) {
    wgpu::ComputePassDescriptor descriptor =
        metrics_collector->GetComputePassDescriptor(i);
    wgpu::ComputePassEncoder compute_encoder =
        encoder->BeginComputePass(&descriptor);
    RETURN_IF_ERROR(nodes_[i].webgpu_operation.Encode(compute_encoder));
    compute_encoder.End();
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::UpdateMutableObjectsBindings(
    const Environment& environment) {
  for (const auto& id : updated_mutable_objects_) {
    auto& obj = mutable_objects_[id];
    SpatialTensor* tensor = memory_manager_.GetSpatialTensor(GetKey(id));
    for (const auto& indexes : obj) {
      auto& node = nodes_[indexes.node_index];
      if (indexes.object_index < node.inputs.size()) {
        int src_index = indexes.object_index;
        RETURN_IF_ERROR(node.webgpu_operation.SetSrcTensor(src_index, tensor));
      } else {
        int dst_index = indexes.object_index - node.inputs.size();
        RETURN_IF_ERROR(node.webgpu_operation.SetDstTensor(dst_index, tensor));
      }
    }
  }
  updated_mutable_objects_.clear();
  for (auto node_index : mutable_node_indexes_) {
    auto& node = nodes_[node_index];
    // optional nodes can be 'invalid' and disabled by user. User must guarantee
    // validness of usage in this scenario(bind all necessary objects before
    // usage).
    bool is_node_valid = true;
    for (auto id : node.inputs) {
      SpatialTensor* tensor = memory_manager_.GetSpatialTensor(GetKey(id));
      if (!tensor || !tensor->IsValidMemory()) {
        is_node_valid = false;
        break;
      }
    }
    for (auto id : node.outputs) {
      SpatialTensor* tensor = memory_manager_.GetSpatialTensor(GetKey(id));
      if (!tensor || !tensor->IsValidMemory()) {
        is_node_valid = false;
        break;
      }
    }
    if (is_node_valid) {
      node.webgpu_operation.UpdateGpuObjectBindings(environment.device());
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::InitSubgraphNodes(
    const Environment& env,
    absl::flat_hash_map<std::string, std::vector<WebGpuNode>> subgraphs,
    TuningType* tuning_type) {
  absl::flat_hash_set<std::string> compiled_subgraphs;
  for (int i = 0; i < nodes_.size();) {
    if (nodes_[i].subgraph_id.empty()) {
      ++i;
    } else {
      auto& subgraph_nodes = subgraphs[nodes_[i].subgraph_id];
      RETURN_IF_ERROR(InitFromSubgraph(env, i, subgraph_nodes,
                                       compiled_subgraphs, tuning_type));
      i += subgraph_nodes.size();
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::InitFromSubgraph(
    const Environment& env, int start_index,
    std::vector<WebGpuNode>& subgraph_nodes,
    absl::flat_hash_set<std::string>& compiled_subgraphs,
    TuningType* tuning_type) {
  bool needs_compile =
      compiled_subgraphs.insert(nodes_[start_index].subgraph_id).second;
  for (int j = 0; j < subgraph_nodes.size(); ++j) {
    auto& sub_node = subgraph_nodes[j];
    auto& node = nodes_[start_index + j];
    // The first time this subgraph is used, the nodes need to be compiled.
    if (needs_compile) {
      // Set all src/dst tensors since this is needed for compile.
      for (int i = 0; i < node.inputs.size(); ++i) {
        RETURN_IF_ERROR(sub_node.webgpu_operation.SetSrcTensor(
            i, GetTensor(node.inputs[i])));
      }
      for (int i = 0; i < node.outputs.size(); ++i) {
        RETURN_IF_ERROR(sub_node.webgpu_operation.SetDstTensor(
            i, GetTensor(node.outputs[i])));
      }
      // Update and compile the subgraph op.
      RETURN_IF_ERROR(sub_node.webgpu_operation.Update(env.device()));
      if (tuning_type != nullptr) {
        RETURN_IF_ERROR(sub_node.webgpu_operation.TuneAndCompile(
            env, *tuning_type, env.GetComputePipelineCache()));
      }
    }
    node.webgpu_operation.CopyFrom(sub_node.webgpu_operation);
  }
  return absl::OkStatus();
}

absl::Status IrModelToInferenceContext(
    Environment* env, const ::ml_drift::ir::IrModel& ir_model,
    ::ml_drift::CreateGpuModelInfo& create_info, InferenceContext* result) {
  GpuModel gpu_model;
  RETURN_IF_ERROR(::ml_drift::ir::IrModelToGpuModel(
      ir_model, create_info, env->GetInfo(), &gpu_model));
  RETURN_IF_ERROR(result->InitFromGpuModel(*env, create_info, &gpu_model));
  return absl::OkStatus();
}

}  // namespace webgpu
}  // namespace ml_drift
