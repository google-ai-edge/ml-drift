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

#include "ml_drift/gl/gl_inference_context.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/verifier.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/gpu_model_util.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/ir_model_util.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/serialization_base.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gl_inference_context_generated.h"
#include "ml_drift/gl/gl_spatial_tensor.h"
#include "ml_drift/gl/portable_gl31.h"
#include "ml_drift/gl/request_gpu_info.h"

namespace ml_drift {
namespace gl {

void GlInferenceContext::CopyFromGpuModel(GpuModel* gpu_model) {
  for (const auto& input : gpu_model->input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& output : gpu_model->output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  nodes_.resize(gpu_model->nodes.size());
  for (int i = 0; i < gpu_model->nodes.size(); ++i) {
    nodes_[i].gl_operation.Init(std::move(gpu_model->nodes[i].gpu_operation));
    nodes_[i].inputs = gpu_model->nodes[i].inputs;
    nodes_[i].outputs = gpu_model->nodes[i].outputs;
    nodes_[i].name = gpu_model->nodes[i].name;
  }
}

absl::Status GlInferenceContext::InitFromGpuModel(
    const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
    std::vector<uint8_t>* serialized_model) {
  GpuInfo gpu_info;
  ABSL_RETURN_IF_ERROR(gl::RequestGpuInfo(&gpu_info));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = ml_drift::Encode(*gpu_model, &builder);
  }
  CopyFromGpuModel(gpu_model);

  if (!create_info.external_mutable_tensors.empty()) {
    return absl::InvalidArgumentError(
        "Mutable external tensors are not supported on OpenGL backend.");
  }

  ExternalTensorsInfo external_tensors;
  external_tensors.immutable_tensors =
      create_info.external_immutable_tensors;

  ABSL_RETURN_IF_ERROR(
      memory_manager_.AllocateMemory(*gpu_model, external_tensors));
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.gl_operation.InitArgs(gpu_info));
  }

  if (gpu_info.IsMali()) {
    need_flush_ = true;
    flush_periodically_ = true;
    flush_period_ = 24;
  }
  if (gpu_info.IsPowerVR()) {
    need_flush_ = true;
  }
  if (gpu_info.IsAdreno() &&
      gpu_info.adreno_info.generation <= AdrenoInfo::Generation::kGen6) {
    // Adreno620 or lower devices have smaller GPU buffer.
    // (Adreno506 could crash with OOM in streaming use cases without periodic
    // flush.)
    need_flush_ = true;
    flush_periodically_ = true;
    if (gpu_info.adreno_info.generation <= AdrenoInfo::Generation::kGen5) {
      // Fails without `flush_period_ = 1` on Adreno 506, Compiler E031.37.12.06
      // (Others, like Adreno 505, 509, 512, could be affected too.)
      flush_period_ = 1;
    } else {
      flush_period_ = 16;
    }
  }

  if (gpu_info.IsMali()) {
    // maybe clarify driver versions,
    // definitely fails on this:
    //   1) Mali-G72
    //   v1.r38p1-01bet0-mbs2v41_0.49c562ebe5faf1a47d2b8e7043cf01b0
    //   2) Mali-G76
    //   v1.r32p1-01bet2-mbs2v39_0.131801e953429f661ecce1d5e1d2b3ef
    //   3) Mali-G52
    //   v1.r32p1-01eac0.fc49d6bd800ecb1492a1f77284a4d178
    add_texture_fetch_barrier_for_buffer_ = true;
  }

  ABSL_RETURN_IF_ERROR(BindMemoryToOperations());
  ABSL_RETURN_IF_ERROR(Compile(gpu_info));

  InitBarriers();

  if (serialized_model) {
    flatbuffers::Offset<data::InferenceContext> encoded_fb;
    ABSL_RETURN_IF_ERROR(Encode(gpu_model_fb, &builder, &encoded_fb));
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }

  return absl::OkStatus();
}

absl::Status GlInferenceContext::RestoreDeserialized(
    const absl::Span<const uint8_t> serialized_model,
    CreateGpuModelInfo* create_info) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  GpuInfo gpu_info;
  ABSL_RETURN_IF_ERROR(gl::RequestGpuInfo(&gpu_info));

  auto decoded_fb = data::GetInferenceContext(serialized_model.data());
  ABSL_RETURN_IF_ERROR(Decode(gpu_info, decoded_fb));

  if (gpu_info.IsMali()) {
    need_flush_ = true;
    flush_periodically_ = true;
    flush_period_ = 24;
  }
  if (gpu_info.IsPowerVR()) {
    need_flush_ = true;
  }

  ABSL_RETURN_IF_ERROR(BindMemoryToOperations());

  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.gl_operation.RestoreDeserialized(gpu_info));
  }

  InitBarriers();

  return absl::OkStatus();
}

absl::Status GlInferenceContext::Encode(
    flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder* builder,
    flatbuffers::Offset<data::InferenceContext>* inf_context_fb) {
  std::vector<flatbuffers::Offset<ml_drift::data::Int3>> work_groups_fb;
  for (int i = 0; i < nodes_.size(); ++i) {
    auto work_group_fb =
        ml_drift::Encode(nodes_[i].gl_operation.GetWorkGroupSize(), builder);
    work_groups_fb.push_back(work_group_fb);
  }
  auto work_groups_fb_vec = builder->CreateVector(work_groups_fb);

  struct BinaryProgram {
    uint32_t binary_format;
    std::vector<uint8_t> data;
  };
  std::map<uint64_t, BinaryProgram> programs_cache;
  std::vector<uint64_t> fingerprints;
  fingerprints.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    uint64_t fingerprint = node.gl_operation.GetFingerprint();
    fingerprints.push_back(fingerprint);
    if (programs_cache.find(fingerprint) == programs_cache.end()) {
      ABSL_RETURN_IF_ERROR(node.gl_operation.GetProgramBinary(
          &programs_cache[fingerprint].data,
          &programs_cache[fingerprint].binary_format));
    }
  }
  auto fingerprints_fb = builder->CreateVector(fingerprints);
  std::vector<flatbuffers::Offset<data::GlProgram>> programs_fb;
  for (const auto& program : programs_cache) {
    auto binary_fb = builder->CreateVector(program.second.data);
    data::GlProgramBuilder program_builder(*builder);
    program_builder.add_fingerprint(program.first);
    program_builder.add_binary_format(program.second.binary_format);
    program_builder.add_binary(binary_fb);
    programs_fb.push_back(program_builder.Finish());
  }
  auto programs_fb_vec = builder->CreateVector(programs_fb);

  data::InferenceContextBuilder inf_builder(*builder);
  inf_builder.add_gpu_model(gpu_model_fb);
  inf_builder.add_tuned_work_group_sizes_per_node(work_groups_fb_vec);
  inf_builder.add_gl_programs(programs_fb_vec);
  inf_builder.add_fingerprints_per_node(fingerprints_fb);
  *inf_context_fb = inf_builder.Finish();
  return absl::OkStatus();
}

absl::Status GlInferenceContext::Decode(
    const GpuInfo& gpu_info, const data::InferenceContext* fb_inference) {
  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(ml_drift::Decode(fb_inference->gpu_model(), &gpu_model));
  CopyFromGpuModel(&gpu_model);
  ABSL_RETURN_IF_ERROR(memory_manager_.AllocateMemory(gpu_model));
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.gl_operation.InitArgsDeserialized(gpu_info));
  }

  struct BinaryProgram {
    uint32_t binary_format;
    std::vector<uint8_t> data;
  };
  std::map<uint64_t, BinaryProgram> programs_cache;

  for (auto program_fb : *fb_inference->gl_programs()) {
    programs_cache[program_fb->fingerprint()].data = std::vector<uint8_t>(
        program_fb->binary()->data(),
        program_fb->binary()->data() + program_fb->binary()->size());
    programs_cache[program_fb->fingerprint()].binary_format =
        program_fb->binary_format();
  }

  for (int i = 0; i < nodes_.size(); ++i) {
    const uint64_t fingerprint = (*fb_inference->fingerprints_per_node())[i];
    ABSL_RETURN_IF_ERROR(
        nodes_[i].gl_operation.Init(programs_cache[fingerprint].data,
                                    programs_cache[fingerprint].binary_format,
                                    fingerprint, &program_cache_));

    int3 wg_size;
    wg_size.x = (*fb_inference->tuned_work_group_sizes_per_node())[i]->x();
    wg_size.y = (*fb_inference->tuned_work_group_sizes_per_node())[i]->y();
    wg_size.z = (*fb_inference->tuned_work_group_sizes_per_node())[i]->z();
    nodes_[i].gl_operation.SetWorkGroupSize(wg_size);
  }
  return absl::OkStatus();
}

absl::Status GlInferenceContext::BindMemoryToOperations() {
  for (auto& node : nodes_) {
    for (int i = 0; i < node.inputs.size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          node.gl_operation.SetSrcTensor(GetTensor(node.inputs[i]), i));
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          node.gl_operation.SetDstTensor(GetTensor(node.outputs[i]), i));
    }
  }
  return absl::OkStatus();
}

absl::Status GlInferenceContext::Compile(const GpuInfo& gpu_info) {
  for (auto& node : nodes_) {
    ABSL_RETURN_IF_ERROR(node.gl_operation.Assemble(gpu_info, &program_cache_));
  }
  return absl::OkStatus();
}

void GlInferenceContext::InitBarriers() {
  in_barrier_ = 0;
  for (auto id : input_ids_) {
    const auto storage_type = GetTensor(id)->GetDescriptor().GetStorageType();
    if (storage_type == TensorStorageType::kBuffer) {
      in_barrier_ = in_barrier_ | GL_SHADER_STORAGE_BARRIER_BIT;
      if (add_texture_fetch_barrier_for_buffer_) {
        in_barrier_ = in_barrier_ | GL_TEXTURE_FETCH_BARRIER_BIT;
      }
    } else if (storage_type == TensorStorageType::kTexture2D ||
               storage_type == TensorStorageType::kTexture3D ||
               storage_type == TensorStorageType::kTextureArray) {
      in_barrier_ = in_barrier_ | GL_TEXTURE_FETCH_BARRIER_BIT;
      in_barrier_ = in_barrier_ | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
    } else if (storage_type == TensorStorageType::kImageBuffer) {
      in_barrier_ = in_barrier_ | GL_SHADER_STORAGE_BARRIER_BIT;
      in_barrier_ = in_barrier_ | GL_TEXTURE_FETCH_BARRIER_BIT;
      in_barrier_ = in_barrier_ | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
    }
  }

  barriers_.resize(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    barriers_[i] = 0;
    for (auto id : nodes_[i].outputs) {
      const auto storage_type = GetTensor(id)->GetDescriptor().GetStorageType();
      if (storage_type == TensorStorageType::kBuffer) {
        barriers_[i] = barriers_[i] | GL_SHADER_STORAGE_BARRIER_BIT;
        if (add_texture_fetch_barrier_for_buffer_) {
          barriers_[i] = barriers_[i] | GL_TEXTURE_FETCH_BARRIER_BIT;
        }
      } else if (storage_type == TensorStorageType::kTexture2D ||
                 storage_type == TensorStorageType::kTexture3D ||
                 storage_type == TensorStorageType::kTextureArray) {
        barriers_[i] = barriers_[i] | GL_TEXTURE_FETCH_BARRIER_BIT;
        barriers_[i] = barriers_[i] | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
      } else if (storage_type == TensorStorageType::kImageBuffer) {
        barriers_[i] = barriers_[i] | GL_SHADER_STORAGE_BARRIER_BIT;
        barriers_[i] = barriers_[i] | GL_TEXTURE_FETCH_BARRIER_BIT;
        barriers_[i] = barriers_[i] | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
      }
    }
  }
}

absl::Status GlInferenceContext::AddToQueue() {
  glMemoryBarrier(in_barrier_);

  for (int i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    ABSL_RETURN_IF_ERROR(node.gl_operation.AddToQueue());
    glMemoryBarrier(barriers_[i]);
    if (flush_periodically_ && i % flush_period_ == 0) {
      glFlush();
    }
  }
  if (need_flush_) {
    glFlush();
  }

  return absl::OkStatus();
}

absl::Status GlInferenceContext::Profile(ProfilingInfo* result) {
  result->dispatches.resize(nodes_.size());

  GpuInfo gpu_info;
  ABSL_RETURN_IF_ERROR(gl::RequestGpuInfo(&gpu_info));
  bool use_queries = false;
  if (gpu_info.SupportsExtension("GL_EXT_disjoint_timer_query")) {
    PFNGLGETQUERYOBJECTUI64VEXTPROC glGetQueryObjectui64vEXT =
        reinterpret_cast<PFNGLGETQUERYOBJECTUI64VEXTPROC>(
            eglGetProcAddress("glGetQueryObjectui64vEXT"));
    use_queries = glGetQueryObjectui64vEXT != nullptr;
    use_queries = false;  // disabled, has too big overhead and pointless. Can
                          // be enabled in future for selected devices.
    if (use_queries) {
      std::vector<GLuint> time_queries(nodes_.size());
      glGenQueries(nodes_.size(), time_queries.data());
      for (int k = 0; k < nodes_.size(); ++k) {
        glBeginQuery(GL_TIME_ELAPSED_EXT, time_queries[k]);
        ABSL_RETURN_IF_ERROR(nodes_[k].gl_operation.AddToQueue());
        glEndQuery(GL_TIME_ELAPSED_EXT);
      }
      glFinish();
      for (int k = 0; k < nodes_.size(); ++k) {
        GLuint64 timeElapsed = 0;
        glGetQueryObjectui64vEXT(time_queries[k], GL_QUERY_RESULT,
                                 &timeElapsed);
        result->dispatches[k].duration = absl::Nanoseconds(timeElapsed);
      }
    }
  }
  if (!use_queries) {
    for (int k = 0; k < nodes_.size(); ++k) {
      ABSL_ASSIGN_OR_RETURN(result->dispatches[k].duration,
                            nodes_[k].gl_operation.GetOperationTime());
    }
  }

  for (int k = 0; k < nodes_.size(); ++k) {
    auto& gpu_op = nodes_[k].gl_operation;
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
      dispatch_info.inshape.push_back(
          GetTensor(src_id)->GetDescriptor().GetBHWCShape().ToShape());
    }
    for (auto& dst_id : nodes_[k].outputs) {
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

uint64_t GlInferenceContext::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return memory_manager_.GetSizeOfMemoryAllocatedForIntermediateTensors();
}

GlSpatialTensor* GlInferenceContext::GetTensor(ValueId id) {
  return memory_manager_.GetTensor(id);
}

absl::Status GlInferenceContext::SetInputTensor(ValueId id,
                                                const TensorFloat32& tensor) {
  GlSpatialTensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data);
}

absl::Status GlInferenceContext::SetInputTensor(
    ValueId id, const Tensor<BHWC, DataType::kInt32>& tensor) {
  GlSpatialTensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data);
}

absl::Status GlInferenceContext::GetOutputTensor(ValueId id,
                                                 TensorFloat32* result) {
  const GlSpatialTensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());
  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc));
  desc.DownloadData(result);
  return absl::OkStatus();
}

absl::Status GlInferenceContext::GetOutputTensor(ValueId id,
                                                 TensorInt32* result) {
  const GlSpatialTensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());
  TensorDescriptor desc;
  ABSL_RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc));
  desc.DownloadData(result);
  return absl::OkStatus();
}

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info) {
  if (gpu_info.IsMali()) {
    return TensorStorageType::kBuffer;
  } else {
    return TensorStorageType::kTexture2D;
  }
}

absl::Status GraphToInferenceContext(const GpuInfo& gpu_info,
                                     const GraphFloat32& graph,
                                     CreateGpuModelInfo& create_info,
                                     GlInferenceContext* context,
                                     GlInferenceContext* weights_prep_context) {
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
          create_info, &gpu_weights_preparation_model));
      for (const auto& upload_info : upload_weights_info) {
        auto tensor = weights_prep_context->GetTensor(upload_info.input_id);
        ABSL_RETURN_IF_ERROR(
            tensor->WriteData(static_cast<const uint8_t*>(upload_info.data)));
      }
      ABSL_RETURN_IF_ERROR(weights_prep_context->AddToQueue());
      for (const auto& [converted_weight_id, main_model_weight_id] :
           weights_mapping) {
        auto tensor = weights_prep_context->GetTensor(converted_weight_id);
        create_info.external_immutable_tensors.insert(
            {main_model_weight_id, tensor});
      }
    }
    ABSL_RETURN_IF_ERROR(
        context->InitFromGpuModel(create_info, &main_gpu_model));
  } else {
    GpuModel gpu_model;
    ABSL_RETURN_IF_ERROR(
        GraphToGpuModel(graph, create_info, gpu_info, &gpu_model));
    ABSL_RETURN_IF_ERROR(context->InitFromGpuModel(create_info, &gpu_model));
  }
  return absl::OkStatus();
}

absl::Status IrModelToInferenceContext(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::ir::IrModel& ir_model,
    ::ml_drift::CreateGpuModelInfo& create_info, GlInferenceContext* context,
    GlInferenceContext* weights_prep_context) {
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
          create_info, &gpu_weights_preparation_model));
      for (const auto& upload_info : upload_weights_info) {
        auto tensor = weights_prep_context->GetTensor(upload_info.input_id);
        ABSL_RETURN_IF_ERROR(
            tensor->WriteData(static_cast<const uint8_t*>(upload_info.data)));
      }
      ABSL_RETURN_IF_ERROR(weights_prep_context->AddToQueue());
      for (const auto& [converted_weight_id, main_model_weight_id] :
           weights_mapping) {
        auto tensor = weights_prep_context->GetTensor(converted_weight_id);
        create_info.external_immutable_tensors.insert(
            {main_model_weight_id, tensor});
      }
    }
    ABSL_RETURN_IF_ERROR(
        context->InitFromGpuModel(create_info, &main_gpu_model));
  } else {
    GpuModel gpu_model;
    ABSL_RETURN_IF_ERROR(
        IrModelToGpuModel(ir_model, create_info, gpu_info, &gpu_model));
    ABSL_RETURN_IF_ERROR(context->InitFromGpuModel(create_info, &gpu_model));
  }
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace ml_drift
