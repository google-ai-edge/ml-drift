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

#include "ml_drift/metal/inference_context.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/string.h"
#include "flatbuffers/verifier.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/ir_model_util.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/serialization_base.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/metal/common.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/inference_context_generated.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {
namespace {
flatbuffers::Offset<data::MetalProgram> EncodeProgram(
    const std::string& code, const std::map<std::string, std::string>& defines,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> names_fb;
  std::vector<flatbuffers::Offset<flatbuffers::String>> expressions_fb;
  for (auto& define : defines) {
    names_fb.push_back(builder->CreateString(define.first));
    expressions_fb.push_back(builder->CreateString(define.second));
  }
  auto names_fb_vec = builder->CreateVector(names_fb);
  auto expressions_fb_vec = builder->CreateVector(expressions_fb);
  auto code_fb = builder->CreateString(code);
  data::MetalProgramBuilder program_builder(*builder);
  program_builder.add_define_names(names_fb_vec);
  program_builder.add_define_expressions(expressions_fb_vec);
  program_builder.add_code(code_fb);
  return program_builder.Finish();
}

void DecodeProgram(const data::MetalProgram* metal_program, std::string* code,
                   std::map<std::string, std::string>* defines) {
  *code = std::string(metal_program->code()->c_str(),
                      metal_program->code()->size());
  for (int i = 0; i < metal_program->define_names()->size(); ++i) {
    std::string key((*metal_program->define_names())[i]->c_str(),
                    (*metal_program->define_names())[i]->size());
    std::string value((*metal_program->define_expressions())[i]->c_str(),
                      (*metal_program->define_expressions())[i]->size());
    (*defines)[key] = value;
  }
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

void InferenceContext::CopyFromGpuModel(GpuModel* gpu_model,
                                        bool use_arguments_buffer) {
  for (const auto& input : gpu_model->input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& output : gpu_model->output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  nodes_.resize(gpu_model->nodes.size());
  for (int i = 0; i < gpu_model->nodes.size(); ++i) {
    nodes_[i].task.Init(std::move(gpu_model->nodes[i].gpu_operation),
                        use_arguments_buffer);
    nodes_[i].inputs = gpu_model->nodes[i].inputs;
    nodes_[i].outputs = gpu_model->nodes[i].outputs;
    nodes_[i].name = gpu_model->nodes[i].name;
    nodes_[i].optional_tag = gpu_model->nodes[i].optional_tag;
  }
}

absl::Status InferenceContext::InitFromGpuModel(
    const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
    id<MTLDevice> device_id, std::vector<uint8_t>* serialized_model) {
  Environment environment(device_id);
  return InitFromGpuModel(create_info, gpu_model, &environment,
                          serialized_model);
}

absl::Status InferenceContext::InitFromGpuModel(
    const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
    Environment* environment, std::vector<uint8_t>* serialized_model) {
  device_ = environment->device();
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = ml_drift::Encode(*gpu_model, &builder);
  }
  CopyFromGpuModel(gpu_model, create_info.hints.use_metal_argument_buffers);

  ExternalTensorsInfo external_tensors;
  external_tensors.immutable_tensors = create_info.external_immutable_tensors;
  external_tensors.mutable_tensors = create_info.external_mutable_tensors;
  std::vector<std::unique_ptr<MetalSpatialTensor>> external_mutable_tensors_allocated_temporarily;
  ABSL_ASSIGN_OR_RETURN(
      model_id_, memory_manager_.AllocateMemory(*environment, *gpu_model, external_tensors,
                                                external_mutable_tensors_allocated_temporarily));

  GetMutableNodes(create_info.external_mutable_tensors);
  ABSL_RETURN_IF_ERROR(CompileOperations(environment));
  ABSL_RETURN_IF_ERROR(BindTensorsToOperations());
  ABSL_RETURN_IF_ERROR(UpdateParams(environment->GetInfo()));
  ABSL_RETURN_IF_ERROR(Tune(TuningType::kFast, environment));

  for (auto& external_tensor : create_info.external_mutable_tensors) {
    ABSL_RETURN_IF_ERROR(memory_manager_.SetExternalTensor(GetKey(external_tensor.first), nullptr));
  }

  if (serialized_model) {
    auto encoded_fb = Encode(environment, gpu_model_fb, &builder);
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }

  bool add_icb_support = false &&
                         create_info.external_mutable_tensors.empty() &&
                         !HasOptionalNode();
  if (add_icb_support) {
    if (@available(macOS 11.00, iOS 13.0, tvOS 13.0, *)) {
      @autoreleasepool {
        MTLIndirectCommandBufferDescriptor* icb_desc =
            [[MTLIndirectCommandBufferDescriptor alloc] init];
        icb_desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icb_desc.inheritBuffers = NO;
        icb_desc.inheritPipelineState = NO;
        icb_desc.maxKernelBufferBindCount = 1;

        icb_ = [device_ newIndirectCommandBufferWithDescriptor:icb_desc
                                               maxCommandCount:nodes_.size()
                                                       options:0];

        for (int i = 0; i < nodes_.size(); ++i) {
          id<MTLIndirectComputeCommand> icb_command =
              [icb_ indirectComputeCommandAtIndex:i];
          auto& node = nodes_[i];
          node.task.EncodeToICB(icb_command);
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::RestoreDeserialized(
    const absl::Span<const uint8_t> serialized_model, id<MTLDevice> device_id,
    const CreateGpuModelInfo* create_info) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto decoded_fb = data::GetInferenceContext(serialized_model.data());
  device_ = device_id;
  Environment environment(device_id);
  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(ml_drift::Decode(decoded_fb->gpu_model(), &gpu_model));
  const CreateGpuModelInfo& create_info_ref =
      create_info ? *create_info : CreateGpuModelInfo();
  CopyFromGpuModel(&gpu_model,
                   create_info_ref.hints.use_metal_argument_buffers);
  ExternalTensorsInfo external_tensors;
  external_tensors.immutable_tensors =
      create_info_ref.external_immutable_tensors;
  external_tensors.mutable_tensors = create_info_ref.external_mutable_tensors;
  std::vector<std::unique_ptr<MetalSpatialTensor>> external_mutable_tensors_allocated_temporarily;
  ABSL_ASSIGN_OR_RETURN(
      model_id_, memory_manager_.AllocateMemory(environment, gpu_model, external_tensors,
                                                external_mutable_tensors_allocated_temporarily));

  ABSL_RETURN_IF_ERROR(DecodeTasks(&environment, decoded_fb));

  GetMutableNodes(create_info_ref.external_mutable_tensors);

  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.task.RestoreDeserialized(&environment));
  }

  ABSL_RETURN_IF_ERROR(BindTensorsToOperations());

  ABSL_RETURN_IF_ERROR(UpdateParams(environment.GetInfo()));
  for (auto& external_tensor : create_info_ref.external_mutable_tensors) {
    ABSL_RETURN_IF_ERROR(memory_manager_.SetExternalTensor(GetKey(external_tensor.first), nullptr));
  }
  return absl::OkStatus();
}

flatbuffers::Offset<data::InferenceContext> InferenceContext::Encode(
    Environment* env,
    flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<ml_drift::data::Int3>> work_groups_fb;
  for (int i = 0; i < nodes_.size(); ++i) {
    auto work_group_fb =
        ml_drift::Encode(nodes_[i].task.GetWorkGroupSize(), builder);
    work_groups_fb.push_back(work_group_fb);
  }
  auto work_groups_fb_vec = builder->CreateVector(work_groups_fb);

  std::vector<flatbuffers::Offset<data::MetalProgram>> programs_fb;
  for (int i = 0; i < nodes_.size(); ++i) {
    auto program_fb = EncodeProgram(nodes_[i].task.GetCode(),
                                    nodes_[i].task.GetDefines(), builder);
    programs_fb.push_back(program_fb);
  }
  auto programs_fb_vec = builder->CreateVector(programs_fb);

  data::InferenceContextBuilder inf_builder(*builder);
  inf_builder.add_gpu_model(gpu_model_fb);
  inf_builder.add_tuned_work_group_sizes_per_node(work_groups_fb_vec);
  inf_builder.add_metal_programs(programs_fb_vec);
  return inf_builder.Finish();
}

absl::Status InferenceContext::DecodeTasks(
    Environment* env, const data::InferenceContext* fb_inference) {
  for (int i = 0; i < nodes_.size(); ++i) {
    std::string code;
    std::map<std::string, std::string> defines;
    DecodeProgram((*fb_inference->metal_programs())[i], &code, &defines);
    ABSL_RETURN_IF_ERROR(nodes_[i].task.Init(env, code, defines));

    int3 wg_size;
    wg_size.x = (*fb_inference->tuned_work_group_sizes_per_node())[i]->x();
    wg_size.y = (*fb_inference->tuned_work_group_sizes_per_node())[i]->y();
    wg_size.z = (*fb_inference->tuned_work_group_sizes_per_node())[i]->z();
    nodes_[i].task.SetWorkGroupSize(wg_size);
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::CompileOperations(Environment* env) {
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.task.Compile(env));
  }
  return absl::OkStatus();
}

MetalSpatialTensor* InferenceContext::GetTensor(ValueId tensor_id) {
  return memory_manager_.GetSpatialTensor(GetKey(tensor_id));
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorFloat32& tensor) {
  MetalSpatialTensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, device_);
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorInt32& tensor) {
  MetalSpatialTensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, device_);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               TensorFloat32* result) {
  const MetalSpatialTensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());

  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device_));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               TensorInt32* result) {
  const MetalSpatialTensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());

  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device_));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::Status InferenceContext::BindTensorsToOperations() {
  for (auto& node : nodes_) {
    const auto& src_ids = node.inputs;
    for (int i = 0; i < src_ids.size(); ++i) {
      ABSL_RETURN_IF_ERROR(node.task.SetSrcTensor(GetTensor(src_ids[i]), i));
    }
    const auto& dst_ids = node.outputs;
    for (int i = 0; i < dst_ids.size(); ++i) {
      ABSL_RETURN_IF_ERROR(node.task.SetDstTensor(GetTensor(dst_ids[i]), i));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::UpdateParams(const GpuInfo& gpu_info) {
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.task.UpdateParams());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Tune(TuningType tuning_type, Environment* env) {
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.task.Tune(tuning_type, env));
  }
  return absl::OkStatus();
}

void InferenceContext::EncodeWithEncoder(
    id<MTLComputeCommandEncoder> command_encoder) {
  for (int i = 0; i < nodes_.size(); ++i) {
    if (absl::c_any_of(nodes_[i].optional_tag,
                       [&](int t) { return !enabled_tags_.contains(t); })) {
      continue;
    }
    auto& task = nodes_[i].task;
    task.Encode(command_encoder);
  }
}

API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
void InferenceContext::AddResources(
    id<MTLComputeCommandEncoder> command_encoder) {
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& task = nodes_[i].task;
    task.AddResourcesToEncoder(command_encoder);
  }
}

API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
void InferenceContext::EncodeWithICB(
    id<MTLComputeCommandEncoder> command_encoder) {
  [command_encoder executeCommandsInBuffer:icb_
                                 withRange:NSMakeRange(0, nodes_.size())];
}

API_AVAILABLE(macos(11.00), ios(14.0))
void InferenceContext::Profile(id<MTLDevice> device, ProfilingInfo* result) {
  result->dispatches.resize(nodes_.size());
  for (int k = 0; k < nodes_.size(); ++k) {
    auto& gpu_op = nodes_[k].task;
    auto& dispatch_info = result->dispatches[k];

    dispatch_info.label = nodes_[k].name;

    uint64_t read_size = 0;
    if (gpu_op.GetReadSize() != -1) {
      read_size = gpu_op.GetReadSize();
    } else {
      for (auto& src_id : nodes_[k].inputs) {
        read_size += GetTensor(src_id)->GetMemorySizeInBytes();
      }
      read_size += gpu_op.GetConstArgsSize();
    }

    uint64_t write_size = 0;
    if (gpu_op.GetWriteSize() != -1) {
      write_size = gpu_op.GetWriteSize();
    } else {
      for (auto& dst_id : nodes_[k].outputs) {
        write_size += GetTensor(dst_id)->GetMemorySizeInBytes();
      }
    }

    for (auto& src_id : nodes_[k].inputs) {
      dispatch_info.inshape.push_back(GetTensor(src_id)->GetDescriptor().GetBHWCShape().ToShape());
    }
    for (auto& dst_id : nodes_[k].outputs) {
      dispatch_info.outshape.push_back(GetTensor(dst_id)->GetDescriptor().GetBHWCShape().ToShape());
    }
    dispatch_info.flops = gpu_op.GetFlopsCount();
    dispatch_info.read_mem_size = read_size;
    dispatch_info.write_mem_size = write_size;
    dispatch_info.work_group_size = gpu_op.GetWorkGroupSize();
  }
  ProfileTime(device, result);
}

void InferenceContext::ProfileTime(id<MTLDevice> device, ProfilingInfo* result) {
  @autoreleasepool {
    if (@available(macOS 11.00, iOS 14.0, *)) {
      auto status = ProfileTimeWithTimestamps(device, result);
      if (status.ok()) {
        return;
      }
    }

    // Measuring time for every kernel individually. Much slower than timestamps
    // but doesn't require any special hardware support. Can have "better" results than timestamps,
    // because data can fit in cache during individual repetitive runs.
    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    for (int k = 0; k < nodes_.size(); k++) {
      result->dispatches[k].duration = nodes_[k].task.GetTaskTime(command_queue);
    }
  }
}

absl::Status InferenceContext::ProfileTimeWithTimestamps(id<MTLDevice> device,
                                                         ProfilingInfo* result) {
  id<MTLCounterSet> timestamp_set = nil;
  for (id<MTLCounterSet> set in [device counterSets]) {
    if ([[set name] isEqualToString:MTLCommonCounterSetTimestamp]) {
      timestamp_set = set;
      break;
    }
  }

  if (!timestamp_set) {
    return absl::InternalError("Timestamp counter set not found.");
  }

  // Maximum counter sample buffer length is 32KB.
  const size_t kMaxSampleBufferSize = 32 * 1024;
  const size_t kMaxSamplesPerBuffer = kMaxSampleBufferSize / sizeof(uint64_t);
  const size_t kMaxTimestampsPerBuffer = kMaxSamplesPerBuffer / 2;

  const size_t timestamps_per_buffer = std::min(nodes_.size(), kMaxTimestampsPerBuffer);

  MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
  desc.counterSet = timestamp_set;
  desc.storageMode = MTLStorageModeShared;
  desc.sampleCount = timestamps_per_buffer * 2;
  const int buffers_count = DivideRoundUp(nodes_.size(), timestamps_per_buffer);
  id<MTLBuffer> resolve_buffer =
      [device newBufferWithLength:(sizeof(uint64_t) * timestamps_per_buffer * buffers_count * 2)
                          options:MTLResourceStorageModeShared];

  NSError* error = nil;
  id<MTLCounterSampleBuffer> sample_buffer = [device newCounterSampleBufferWithDescriptor:desc
                                                                                    error:&error];
  if (!sample_buffer) {
    return absl::InternalError("Failed to create sample buffer.");
  }

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  for (int buffer_index = 0; buffer_index < buffers_count; ++buffer_index) {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    for (int i = 0; i < timestamps_per_buffer; ++i) {
      int node_index = buffer_index * timestamps_per_buffer + i;
      if (node_index >= nodes_.size()) {
        break;
      }
      MTLComputePassDescriptor* pass_desc = [MTLComputePassDescriptor computePassDescriptor];
      pass_desc.sampleBufferAttachments[0].sampleBuffer = sample_buffer;
      pass_desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = i * 2;
      pass_desc.sampleBufferAttachments[0].endOfEncoderSampleIndex = i * 2 + 1;
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoderWithDescriptor:pass_desc];
      nodes_[node_index].task.Encode(encoder);
      [encoder endEncoding];
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];  // Doesn't work properly without this wait, returns zeroes
                                          // for last kernels.

    command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    [blit resolveCounters:sample_buffer
                  inRange:NSMakeRange(0, timestamps_per_buffer * 2)
        destinationBuffer:resolve_buffer
        destinationOffset:sizeof(uint64_t) * timestamps_per_buffer * buffer_index * 2];
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  uint64_t* timestamps = (uint64_t*)[resolve_buffer contents];

  for (int k = 0; k < nodes_.size(); k++) {
    const uint64_t time_ns = timestamps[k * 2 + 1] - timestamps[k * 2];
    result->dispatches[k].duration = absl::Nanoseconds(time_ns);
  }
  return absl::OkStatus();
}

bool InferenceContext::HasOptionalNode() const {
  return absl::c_any_of(
      nodes_, [](const auto& node) { return !node.optional_tag.empty(); });
}

uint64_t InferenceContext::GetIntermediateTensorsSize() const {
  return memory_manager_.GetIntermediateTensorsSize();
}

uint64_t InferenceContext::GetConstantTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& node : nodes_) {
    total_size += node.task.GetConstArgsSize();
  }
  return total_size + memory_manager_.GetConstantTensorsSize();
}

API_AVAILABLE(ios(18.0), macos(15.0))
void InferenceContext::AddConstantsToResidencySet(id<MTLResidencySet> residency_set) const {
  memory_manager_.AddConstantsToResidencySet(residency_set);
}

API_AVAILABLE(ios(18.0), macos(15.0))
void InferenceContext::AddIntermediatesToResidencySet(id<MTLResidencySet> residency_set) const {
  memory_manager_.AddIntermediatesToResidencySet(residency_set);
}

API_AVAILABLE(ios(18.0), macos(15.0))
void InferenceContext::AddExternalImmutableToResidencySet(id<MTLResidencySet> residency_set) const {
  memory_manager_.AddExternalImmutableToResidencySet(residency_set);
}

API_AVAILABLE(ios(18.0), macos(15.0))
void InferenceContext::AddExternalMutableToResidencySet(id<MTLResidencySet> residency_set) const {
  memory_manager_.AddExternalMutableToResidencySet(residency_set);
}

void InferenceContext::EncodeWithCommandBuffer(
    id<MTLCommandBuffer> command_buffer, int tasks_per_encoder) {
  const int encoders_count = DivideRoundUp(nodes_.size(), tasks_per_encoder);
  int node_index = 0;
  for (int i = 0; i < encoders_count; ++i) {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      int num_tasks = 0;
      while (num_tasks < tasks_per_encoder && node_index < nodes_.size()) {
        auto& node = nodes_[node_index++];
        if (absl::c_any_of(node.optional_tag,
                           [&](int t) { return !enabled_tags_.contains(t); })) {
          continue;
        }
        auto& task = node.task;
        task.Encode(encoder);
        num_tasks++;
      }
      [encoder endEncoding];
    }
  }
}

void InferenceContext::EncodeWithCommandQueue(id<MTLCommandQueue> command_queue,
                                              int flush_period) {
  const int cb_count = DivideRoundUp(nodes_.size(), flush_period);
  int node_index = 0;
  for (int i = 0; i < cb_count; ++i) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      int num_tasks = 0;
      while (num_tasks < flush_period && node_index < nodes_.size()) {
        auto& node = nodes_[node_index++];
        if (absl::c_any_of(node.optional_tag,
                           [&](int t) { return !enabled_tags_.contains(t); })) {
          continue;
        }
        auto& task = node.task;
        task.Encode(encoder);
        num_tasks++;
      }
      [encoder endEncoding];
      [command_buffer commit];
    }
  }
}

absl::Status InferenceContext::SetTensor(const ValueId& tensor_id,
                                         MetalSpatialTensor* tensor_ptr) {
  ABSL_RETURN_IF_ERROR(memory_manager_.SetExternalTensor(GetKey(tensor_id), tensor_ptr));
  for (int node_index : external_tensor_to_nodes_[tensor_id]) {
    auto& node = nodes_[node_index];
    for (int i = 0; i < node.inputs.size(); ++i) {
      if (node.inputs[i] == tensor_id) {
        ABSL_RETURN_IF_ERROR(node.task.SetSrcTensor(tensor_ptr, i));
      }
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      if (node.outputs[i] == tensor_id) {
        ABSL_RETURN_IF_ERROR(node.task.SetDstTensor(tensor_ptr, i));
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

absl::Status IrModelToInferenceContext(const ::ml_drift::ir::IrModel& ir_model,
                                       ::ml_drift::CreateGpuModelInfo& create_info,
                                       const ::ml_drift::GpuInfo& gpu_info, id<MTLDevice> device,
                                       InferenceContext* result) {
  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(
      ::ml_drift::ir::IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
  ABSL_RETURN_IF_ERROR(result->InitFromGpuModel(create_info, &gpu_model, device, nullptr));
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace ml_drift
