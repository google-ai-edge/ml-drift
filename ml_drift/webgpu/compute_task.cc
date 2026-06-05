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

#include "ml_drift/webgpu/compute_task.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"
#include "ml_drift/webgpu/arguments.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/compute_pipeline_cache.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/preprocessor.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"  // IWYU pragma: keep
#include <farmhash.h>

namespace ml_drift {
namespace webgpu {
namespace {

// Number of op dispatches for tuning as many small ops might be close to 0ns.
constexpr int kNumOpDispatchesForTuning = 10;

void AppendArgument(const std::string& arg, std::string* args) {
  if (!args->empty()) {
    absl::StrAppend(args, ",\n");
  }
  absl::StrAppend(args, arg);
}

void AddSystemArguments(const CodeInfo& code_info, std::string* code) {
  std::string system_arguments;
  if (code_info.uses_global_id) {
    AppendArgument("@builtin(global_invocation_id) reserved_gid : vec3<u32>",
                   &system_arguments);
  }
  if (code_info.uses_local_id) {
    AppendArgument("@builtin(local_invocation_id) reserved_lid : vec3<u32>",
                   &system_arguments);
  }
  if (code_info.uses_group_id) {
    AppendArgument("@builtin(workgroup_id) reserved_group_id : vec3<u32>",
                   &system_arguments);
  }
  if (code_info.uses_sub_group_local_id) {
    AppendArgument(
        "@builtin(subgroup_invocation_id) reserved_subgroup_local_id : u32",
        &system_arguments);
  }
  if (code_info.uses_sub_group_id) {
    AppendArgument("@builtin(subgroup_id) reserved_subgroup_id : u32",
                   &system_arguments);
  }
  if (code_info.uses_sub_group_size) {
    AppendArgument("@builtin(subgroup_size) reserved_subgroup_size : u32",
                   &system_arguments);
  }
  *code = absl::Substitute(*code, system_arguments);
}
void ResolveSystemDefines(std::string* code) {
  const std::string main_func =
      "@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, "
      "WORKGROUP_SIZE_Z)\nfn main";

  absl::StrReplaceAll({{"MAIN_FUNCTION", main_func},
                       {"reserved_group_size.x", "WORKGROUP_SIZE_X"},
                       {"reserved_group_size.y", "WORKGROUP_SIZE_Y"},
                       {"reserved_group_size.z", "WORKGROUP_SIZE_Z"}},
                      code);
}

absl::StatusOr<int> GetBestDispatchIndex(
    const Environment& env, const std::string& code,
    const WebGpuArguments& args,
    const std::vector<GPUOperation::DispatchInfo>& possible_dispatches) {
  std::vector<std::unique_ptr<ComputePipelineHolder>> pipelines(
      possible_dispatches.size());

  for (int i = 0; i < possible_dispatches.size(); ++i) {
    const auto& wg_size = possible_dispatches[i].work_group_size;
    std::vector<wgpu::ConstantEntry> constants = {
        {.key = "0", .value = static_cast<double>(wg_size.x)},
        {.key = "1", .value = static_cast<double>(wg_size.y)},
        {.key = "2", .value = static_cast<double>(wg_size.z)},
    };
    ASSIGN_OR_RETURN(pipelines[i],
                     CreateComputePipeline(env.device(), code, constants,
                                           "main", args.GetPipelineLayout(),
                                           /*executor=*/nullptr,
                                           /*async_create_call=*/false));
  }

  const int query_count = 2 * possible_dispatches.size();
  wgpu::QuerySetDescriptor query_set_descriptor;
  query_set_descriptor.type = wgpu::QueryType::Timestamp;
  query_set_descriptor.count = query_count;
  wgpu::QuerySet query_set = env.device().CreateQuerySet(&query_set_descriptor);
  std::vector<wgpu::PassTimestampWrites> timestamp_writes(
      possible_dispatches.size());
  for (uint32_t i = 0; i < possible_dispatches.size(); ++i) {
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
  for (int i = 0; i < possible_dispatches.size(); ++i) {
    wgpu::ComputePassDescriptor compute_pass_descriptor;
    compute_pass_descriptor.timestampWrites = &timestamp_writes[i];

    wgpu::ComputePassEncoder compute_encoder =
        encoder.BeginComputePass(&compute_pass_descriptor);
    {
      const auto& wg_count = possible_dispatches[i].work_groups_count;
      ASSIGN_OR_RETURN(auto pipeline, pipelines[i]->Get());
      compute_encoder.SetPipeline(pipeline);
      args.Bind(compute_encoder);
      for (int loop = 0; loop < kNumOpDispatchesForTuning; ++loop) {
        compute_encoder.DispatchWorkgroups(wg_count.x, wg_count.y, wg_count.z);
      }
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
  uint64_t best_time_ns = query_data[1] - query_data[0];
  int best_index = 0;
  for (int i = 1; i < possible_dispatches.size(); ++i) {
    const uint64_t duration_ns = query_data[i * 2 + 1] - query_data[i * 2];
    if (duration_ns < best_time_ns) {
      best_time_ns = duration_ns;
      best_index = i;
    } else if (duration_ns == best_time_ns) {
      int64_t current_size = possible_dispatches[i].work_group_size.x *
                             possible_dispatches[i].work_group_size.y *
                             possible_dispatches[i].work_group_size.z;
      int64_t best_size = possible_dispatches[best_index].work_group_size.x *
                          possible_dispatches[best_index].work_group_size.y *
                          possible_dispatches[best_index].work_group_size.z;
      // Prefer bigger work group size.
      if (current_size > best_size) {
        best_index = i;
      }
    }
  }

  return best_index;
}
}  // namespace

absl::Status ComputeTask::SetSrcTensor(int index, SpatialTensor* tensor) {
  operation_->SetSrc(tensor, index);
  return webgpu_args_.SetObjectRef(operation_->GetSrcTensorsNames()[index],
                                   *tensor);
}

absl::Status ComputeTask::SetDstTensor(int index, SpatialTensor* tensor) {
  operation_->SetDst(tensor, index);
  return webgpu_args_.SetObjectRef(operation_->GetDstTensorsNames()[index],
                                   *tensor);
}

absl::Status ComputeTask::SetSrcBuffer(int index, Buffer* buffer) {
  return webgpu_args_.SetObjectRef(operation_->GetSrcTensorsNames()[index],
                                   *buffer);
}

absl::Status ComputeTask::SetDstBuffer(int index, Buffer* buffer) {
  return webgpu_args_.SetObjectRef(operation_->GetDstTensorsNames()[index],
                                   *buffer);
}

absl::Status ComputeTask::Tune(const Environment& env, TuningType tuning_type) {
  if (!env.GetInfo().webgpu_info.supports_timestamp_query) {
    tuning_type = TuningType::kFast;
  }
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = env.GetInfo().GetMaxWorkGroupTotalSize();
  kernel_info.private_memory_size = 0;
  std::vector<GPUOperation::DispatchInfo> possible_dispatches;
  operation_->GetPossibleDispatches(tuning_type, env.GetInfo(), kernel_info,
                                    &possible_dispatches);
  if (possible_dispatches.empty()) {
    return absl::NotFoundError("No dispatch parameters to launch kernel");
  }
  if (possible_dispatches.size() == 1 ||
      !env.GetInfo().webgpu_info.supports_timestamp_query) {
    operation_->work_group_size_ = possible_dispatches[0].work_group_size;
    operation_->RecalculateWorkGroupsCount();
    return absl::OkStatus();
  }
  std::vector<ml_drift::int3> work_group_sizes(possible_dispatches.size());
  std::vector<ml_drift::int3> work_groups_counts(possible_dispatches.size());
  for (int i = 0; i < possible_dispatches.size(); ++i) {
    work_group_sizes[i] = possible_dispatches[i].work_group_size;
    work_groups_counts[i] = possible_dispatches[i].work_groups_count;
  }

  UpdateGpuObjectBindings(env.device());
  ASSIGN_OR_RETURN(int best_dispatch_index,
                   GetBestDispatchIndex(env, operation_->code_, webgpu_args_,
                                        possible_dispatches));
  operation_->work_group_size_ =
      possible_dispatches[best_dispatch_index].work_group_size;
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

void ComputeTask::AddGlobalDeclarations(const WebGpuInfo& webgpu_info,
                                        const ExtensionsInfo& extensions_info) {
  std::string declarations;
  if (webgpu_info.supports_fp16) {
    declarations += "enable f16;\n";
  }
  for (const auto& enable_extension : extensions_info.enable_extensions) {
    declarations += "enable " + enable_extension + ";\n";
    if (enable_extension == "subgroups") {
      declarations += "diagnostic(off,subgroup_uniformity);\n";
    }
    if (enable_extension == "chromium_experimental_subgroup_matrix") {
      declarations +=
          "diagnostic (off, chromium.subgroup_matrix_uniformity);\n";
    }
  }
  for (const auto& language_extension : extensions_info.language_extensions) {
    declarations += "requires " + language_extension + ";\n";
  }
  declarations += "@id(0) override WORKGROUP_SIZE_X: i32;\n";
  declarations += "@id(1) override WORKGROUP_SIZE_Y: i32;\n";
  declarations += "@id(2) override WORKGROUP_SIZE_Z: i32;\n";
  if (webgpu_args_.HasFloat16Buffers()) {
    declarations += R"(
fn Pack4x16float(v : vec4<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}

fn Unpack4x16float(v : vec2<u32>) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y));
}
)";
  }
  if (!declarations.empty()) {
    operation_->code_ = declarations + operation_->code_;
  }
}

absl::Status ComputeTask::TuneAndCompile(
    const Environment& env, TuningType tuning_type,
    ComputePipelineCache* compute_pipeline_cache) {
  RETURN_IF_ERROR(CompileToWGSL(env, compute_pipeline_cache));
  code_fingerprint_ = ::util::Fingerprint64(operation_->code_);
  RETURN_IF_ERROR(Tune(env, tuning_type));
  return CompileWGSLToPipeline(env, compute_pipeline_cache);
}

absl::Status ComputeTask::Compile(
    const Environment& env, ComputePipelineCache* compute_pipeline_cache) {
  RETURN_IF_ERROR(CompileToWGSL(env, compute_pipeline_cache));
  code_fingerprint_ = ::util::Fingerprint64(operation_->code_);
  return CompileWGSLToPipeline(env, compute_pipeline_cache);
}

absl::Status ComputeTask::RestoreDeserialized(
    const Environment& env, std::string& code, const uint64_t fingerprint,
    ComputePipelineCache* compute_pipeline_cache) {
  code_fingerprint_ = fingerprint;
  operation_->code_ = code;
  return CompileWGSLToPipeline(env, compute_pipeline_cache);
}

absl::Status ComputeTask::CompileToWGSL(
    const Environment& env, ComputePipelineCache* compute_pipeline_cache) {
  ExtensionsInfo extensions_info;
  if (compute_pipeline_cache) {
    RETURN_IF_ERROR(compute_pipeline_cache->ConvertToWGSL(
        env.GetInfo().webgpu_info, &operation_->code_, &extensions_info));
  } else {
    RETURN_IF_ERROR(ConvertToWGSL(env.GetInfo().webgpu_info, &operation_->code_,
                                  &extensions_info));
  }
  AddSystemArguments(operation_->code_info_, &operation_->code_);
  ResolveSystemDefines(&operation_->code_);
  AddGlobalDeclarations(env.GetInfo().webgpu_info, extensions_info);
  return absl::OkStatus();
}

absl::Status ComputeTask::CompileWGSLToPipeline(
    const Environment& env, ComputePipelineCache* compute_pipeline_cache) {
  const auto& wg_size = operation_->work_group_size_;
  std::vector<wgpu::ConstantEntry> constants = {{
      {.key = "0", .value = static_cast<double>(wg_size.x)},
      {.key = "1", .value = static_cast<double>(wg_size.y)},
      {.key = "2", .value = static_cast<double>(wg_size.z)},
  }};
  if (compute_pipeline_cache != nullptr) {
    ASSIGN_OR_RETURN(compute_pipeline_,
                     compute_pipeline_cache->GetOrCreatePipeline(
                         env.device(), operation_->code_, constants, "main",
                         webgpu_args_.GetPipelineLayout(),
                         env.use_async_create_calls(), GetFullFingerprint()));
  } else {
    // No caching in this case.
    ASSIGN_OR_RETURN(owned_compute_pipeline_,
                     CreateComputePipeline(
                         env.device(), operation_->code_, constants, "main",
                         webgpu_args_.GetPipelineLayout(),
                         /*executor=*/nullptr, env.use_async_create_calls()));
    compute_pipeline_ = owned_compute_pipeline_.get();
  }
  return absl::OkStatus();
}

void ComputeTask::UpdateGpuObjectBindings(const wgpu::Device& device) {
  webgpu_args_.UpdateBindingsOnGPU(device);
}

absl::Status ComputeTask::Update(const wgpu::Device& device) {
  RETURN_IF_ERROR(operation_->BindArguments(&webgpu_args_));
  RETURN_IF_ERROR(webgpu_args_.UpdateScalars(device));
  operation_->RecalculateGridSize();
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

absl::Status ComputeTask::Encode(
    const wgpu::ComputePassEncoder& compute_encoder) const {
  if (!compute_pipeline_) {
    return absl::InternalError("Compute pipeline has not been created.");
  }
  ASSIGN_OR_RETURN(auto pipeline, compute_pipeline_->Get());
  compute_encoder.SetPipeline(pipeline);
  webgpu_args_.Bind(compute_encoder);
  const int3 work_groups_count = operation_->GetWorkGroupsCount();
  compute_encoder.DispatchWorkgroups(work_groups_count.x, work_groups_count.y,
                                     work_groups_count.z);
  return absl::OkStatus();
}

absl::Status ComputeTask::Execute(const Environment& env) {
  UpdateGpuObjectBindings(env.device());
  wgpu::CommandEncoder encoder = env.device().CreateCommandEncoder();
  wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
  RETURN_IF_ERROR(Encode(compute_encoder));
  compute_encoder.End();
  wgpu::CommandBuffer cb = encoder.Finish();
  env.queue().Submit(1, &cb);
  return absl::OkStatus();
}

absl::Duration ComputeTask::GetOperationTime(const Environment& env) {
  const int kMaxRuns = 5000;
  const int kMaxIterations = 5;
  const double kConvergeTolerance = 10.0;  // in percents
  const double kTestRunMs = 100.0;
  absl::Duration duration = absl::ZeroDuration();
  double prev_time_ms = 10000.0f;  // 10sec
  for (int i = 0; i < kMaxIterations; ++i) {
    int num_runs = static_cast<int>(kTestRunMs / prev_time_ms);
    num_runs = std::min(std::max(num_runs, 4), kMaxRuns);
    wgpu::CommandEncoder encoder = env.device().CreateCommandEncoder();
    wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
    for (int j = 0; j < num_runs; ++j) {
      Encode(compute_encoder).IgnoreError();
    }
    compute_encoder.End();
    wgpu::CommandBuffer cb = encoder.Finish();
    auto start = absl::Now();
    env.queue().Submit(1, &cb);
    WaitUntilCompleted(env.queue(), env.device(), absl::Seconds(10))
        .IgnoreError();
    auto end = absl::Now();

    duration = (end - start) / static_cast<float>(num_runs);

    if (num_runs == kMaxRuns) {
      break;
    }

    const double current_time_ms = absl::ToDoubleMilliseconds(duration);
    const double diff_percent =
        std::abs(1.0 - current_time_ms / prev_time_ms) * 100.0;
    prev_time_ms = current_time_ms;
    if (diff_percent <= kConvergeTolerance) {
      break;
    }
  }
  return duration;
}

void ComputeTask::CopyFrom(const ComputeTask& other) {
  operation_ = other.operation_;
  compute_pipeline_ = other.compute_pipeline_;
  webgpu_args_.CopyFrom(other.webgpu_args_);
  code_fingerprint_ = other.code_fingerprint_;
}

void ComputeTask::SetWorkGroupSize(const int3& work_group_size) {
  operation_->work_group_size_ = work_group_size;
  operation_->RecalculateWorkGroupsCount();
}

}  // namespace webgpu
}  // namespace ml_drift
