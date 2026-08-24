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

#include "ml_drift/metal/compute_task.h"

#include <Availability.h>

#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/status.h"           // IWYU pragma: keep
#include "absl/status/status_macros.h"    // IWYU pragma: keep
#include "absl/strings/match.h"           // IWYU pragma: keep
#include "absl/strings/substitute.h"      // IWYU pragma: keep
#include "ml_drift/common/kernel_info.h"  // IWYU pragma: keep
#include "ml_drift/common/shape.h"        // IWYU pragma: keep
#include "ml_drift/common/types.h"        // IWYU pragma: keep
#include "ml_drift/common/util.h"         // IWYU pragma: keep
#include "ml_drift/metal/common.h"        // IWYU pragma: keep

namespace ml_drift {
namespace metal {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

int ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                    std::string* str) {
  int count = 0;
  size_t position = str->find(old_word);
  while (position != std::string::npos) {
    const char prev = position == 0 ? ' ' : (*str)[position - 1];
    const char next = position + old_word.size() < str->size()
                          ? (*str)[position + old_word.size()]
                          : ' ';
    if (IsWordSymbol(prev) || IsWordSymbol(next)) {
      position = str->find(old_word, position + 1);
      continue;
    }
    str->replace(position, old_word.size(), new_word);
    count++;
    position = str->find(old_word, position + new_word.size());
  }
  return count;
}

std::map<std::string, std::string> GetMetalDefines(Environment* device,
                                                   const std::string& function_name) {
  return {{"MAIN_FUNCTION", "kernel void " + function_name},
          {"__local", "threadgroup"},
          {"__global", "device"},
          {"__constant", "constant"}};
}

std::string GetCustomTypeStructDefinition(const std::string& type_name,
                                          int type_size) {
  std::string struct_definition;
  struct_definition +=
      absl::StrCat("struct ", type_name, type_size, "_custom {\n");
  for (int i = 0; i < type_size; i += 4) {
    struct_definition += absl::StrCat("  ", type_name, "4 v", i / 4, ";\n");
  }
  struct_definition += "};\n";
  return struct_definition;
}
}  // namespace

ComputeTask::ComputeTask(ComputeTask&& task)
    : operation_(std::move(task.operation_)),
      program_(task.program_),
      metal_args_(std::move(task.metal_args_)),
      use_arguments_buffer_(task.use_arguments_buffer_),
      need_icb_support_(task.need_icb_support_),
      arguments_encoder_(task.arguments_encoder_),
      arg_buffer_(task.arg_buffer_) {
  task.program_ = nullptr;
  task.arguments_encoder_ = nullptr;
  task.arg_buffer_ = nullptr;
}

ComputeTask& ComputeTask::operator=(ComputeTask&& task) {
  if (this != &task) {
    Release();
    operation_ = std::move(task.operation_);
    std::swap(program_, task.program_);
    metal_args_ = std::move(task.metal_args_);
    std::swap(use_arguments_buffer_, task.use_arguments_buffer_);
    std::swap(need_icb_support_, task.need_icb_support_);
    std::swap(arguments_encoder_, task.arguments_encoder_);
    std::swap(arg_buffer_, task.arg_buffer_);
  }
  return *this;
}

ComputeTask::~ComputeTask() { Release(); }

void ComputeTask::Release() {
  if (program_) {
    program_ = nullptr;
  }
  if (arguments_encoder_) {
    arguments_encoder_ = nullptr;
  }
  if (arg_buffer_) {
    arg_buffer_ = nullptr;
  }
}

void ComputeTask::Init(std::unique_ptr<GPUOperation>&& operation,
                       bool use_argument_buffer) {
  operation_ = std::move(operation);
  use_arguments_buffer_ = use_argument_buffer;
}

absl::Status ComputeTask::InitArgs(Environment* env) {
  ABSL_RETURN_IF_ERROR(metal_args_.Init(operation_->code_info_, use_arguments_buffer_, env,
                                        &operation_->args_, &operation_->code_));

  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status ComputeTask::InitArgsDeserialized(Environment* env) {
  ABSL_RETURN_IF_ERROR(metal_args_.Init(use_arguments_buffer_, env, &operation_->args_));

  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status ComputeTask::Compile(Environment* env, const std::string& name) {
  ABSL_RETURN_IF_ERROR(InitArgs(env));

  // manually resolving Metal reserved types(float16, half8, etc)
  std::string struct_definitions;
  for (const std::string& type_name :
       {"float", "half", "uint", "int", "ushort", "short", "uchar", "char"}) {
    for (const int type_size : {16, 8}) {
      const std::string struct_name =
          type_name + std::to_string(type_size) + "_custom";
      if (ReplaceAllWords(type_name + std::to_string(type_size), struct_name,
                          &operation_->code_)) {
        struct_definitions +=
            GetCustomTypeStructDefinition(type_name, type_size);
      }
    }
  }
  std::string function_name = name;
  if (function_name.empty()) {
    function_name = "ComputeFunction";
  } else {
    for (char& c : function_name) {
      if (!absl::ascii_isalnum(c)) {
        c = '_';
      }
    }
  }
  operation_->code_ = struct_definitions + operation_->code_;
  defines_ = GetMetalDefines(env, function_name);
  return CompileProgram(env, operation_->code_, defines_, function_name);
}

absl::Status ComputeTask::CompileProgram(
    Environment* env, const std::string& code,
    const std::map<std::string, std::string>& defines,
    const std::string& function_name) {
  @autoreleasepool {
    id<MTLComputePipelineState> program;
    if (use_arguments_buffer_) {
      id<MTLArgumentEncoder> arguments_encoder;
      if (need_icb_support_) {
        ABSL_RETURN_IF_ERROR(CreateComputeProgramWithICBSupport(
            env->device(), code, function_name, defines, &program, &arguments_encoder));
      } else {
        ABSL_RETURN_IF_ERROR(CreateComputeProgramWithArgumentBuffer(
            env->device(), code, function_name, defines, &program, &arguments_encoder));
      }
      arguments_encoder_ = arguments_encoder;
      arg_buffer_ =
          [env->device() newBufferWithLength:arguments_encoder_.encodedLength
                                     options:0];
      if (!arg_buffer_) {
        return absl::InternalError("Failed to create MTLBuffer.");
      }
    } else {
      ABSL_RETURN_IF_ERROR(
          CreateComputeProgram(env->device(), code, function_name, defines, &program));
    }
    program_ = program;
  }
  return absl::OkStatus();
}

absl::Status ComputeTask::Init(
    Environment* env, const std::string& code,
    const std::map<std::string, std::string>& defines,
    const std::string& function_name) {
  return CompileProgram(env, code, defines, function_name);
}

absl::Status ComputeTask::RestoreDeserialized(Environment* env) {
  return InitArgsDeserialized(env);
}

absl::Status ComputeTask::UpdateParams() {
  ABSL_RETURN_IF_ERROR(operation_->BindArguments(&metal_args_));
  operation_->RecalculateGridSize();
  operation_->RecalculateWorkGroupsCount();
  UpdateArgumentBuffer();
  return absl::OkStatus();
}

API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
void ComputeTask::EncodeToICB(id<MTLIndirectComputeCommand> icb_command) {
  MTLSize groupsCount, groupsSize;
  groupsCount.width = operation_->GetWorkGroupsCount().x;
  groupsCount.height = operation_->GetWorkGroupsCount().y;
  groupsCount.depth = operation_->GetWorkGroupsCount().z;
  groupsSize.width = operation_->work_group_size_.x;
  groupsSize.height = operation_->work_group_size_.y;
  groupsSize.depth = operation_->work_group_size_.z;
  [icb_command setComputePipelineState:program_];
  [icb_command setKernelBuffer:arg_buffer_ offset:0 atIndex:0];
  [icb_command concurrentDispatchThreadgroups:groupsCount
                        threadsPerThreadgroup:groupsSize];
  [icb_command setBarrier];
}

void ComputeTask::AddResourcesToEncoder(
    id<MTLComputeCommandEncoder> encoder) const {
  metal_args_.AddResourcesToEncoder(encoder);
}

void ComputeTask::UpdateArgumentBuffer() {
  if (use_arguments_buffer_) {
    [arguments_encoder_ setArgumentBuffer:arg_buffer_ offset:0];
    metal_args_.EncodeArguments(arguments_encoder_);
  }
}

void ComputeTask::Encode(id<MTLComputeCommandEncoder> encoder) {
  [encoder setComputePipelineState:program_];
  if (use_arguments_buffer_) {
    metal_args_.AddResourcesToEncoder(encoder);
    [encoder setBuffer:arg_buffer_ offset:0 atIndex:0];
  } else {
    metal_args_.Encode(encoder, 0);
  }
  MTLSize groupsCount, groupsSize;
  groupsCount.width = operation_->GetWorkGroupsCount().x;
  groupsCount.height = operation_->GetWorkGroupsCount().y;
  groupsCount.depth = operation_->GetWorkGroupsCount().z;
  groupsSize.width = operation_->work_group_size_.x;
  groupsSize.height = operation_->work_group_size_.y;
  groupsSize.depth = operation_->work_group_size_.z;
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

void ComputeTask::SetSrcTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetSrc(tensor, index);
  auto status = metal_args_.SetObjectRef(
      operation_->GetSrcTensorsNames()[index], *tensor);
  UpdateArgumentBuffer();
}

void ComputeTask::SetDstTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetDst(tensor, index);
  auto status = metal_args_.SetObjectRef(
      operation_->GetDstTensorsNames()[index], *tensor);
  UpdateArgumentBuffer();
}

void ComputeTask::SetSrcBuffer(Buffer* buffer, int index) {
  auto status = metal_args_.SetObjectRef(
      operation_->GetSrcTensorsNames()[index], *buffer);
  UpdateArgumentBuffer();
}

void ComputeTask::SetDstBuffer(Buffer* buffer, int index) {
  auto status = metal_args_.SetObjectRef(
      operation_->GetDstTensorsNames()[index], *buffer);
  UpdateArgumentBuffer();
}

absl::Status ComputeTask::Tune(TuningType tuning_type, Environment* env) {
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = [program_ maxTotalThreadsPerThreadgroup];
  kernel_info.private_memory_size = 0;
  std::vector<GPUOperation::DispatchInfo> possible_dispatches;
  operation_->GetPossibleDispatches(tuning_type, env->GetInfo(), kernel_info,
                                    &possible_dispatches);
  if (possible_dispatches.empty()) {
    return absl::NotFoundError("No dispatch parameters to launch kernel");
  }
  operation_->work_group_size_ = possible_dispatches[0].work_group_size;
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

void ComputeTask::SetWorkGroupSize(const int3& work_group_size) {
  operation_->work_group_size_ = work_group_size;
  operation_->RecalculateWorkGroupsCount();
}

absl::Duration ComputeTask::GetTaskTime(id<MTLCommandQueue> command_queue) {
  const int kMaxRuns = 5000;
  const int kMaxIterations = 5;
  const double kConvergeTolerance = 10.0;  // in percents
  const double kTestRunMs = 100.0;
  absl::Duration duration = absl::ZeroDuration();
  double prev_time_ms = 10000.0f;  // 10sec
  for (int i = 0; i < kMaxIterations; ++i) {
    int num_runs = static_cast<int>(kTestRunMs / prev_time_ms);
    num_runs = std::min(std::max(num_runs, 4), kMaxRuns);
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      for (int i = 0; i < num_runs; ++i) {
        Encode(encoder);
      }
      [encoder endEncoding];
      auto start = absl::Now();
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      auto end = absl::Now();

      duration = (end - start) / static_cast<float>(num_runs);
    }

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

}  // namespace metal
}  // namespace ml_drift
