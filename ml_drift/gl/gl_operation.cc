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

#include "ml_drift/gl/gl_operation.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_shader.h"
#include "ml_drift/gl/gl_spatial_tensor.h"
#include "ml_drift/gl/portable_gl31.h"
#include <farmhash.h>

namespace ml_drift {
namespace gl {
namespace {

std::vector<std::string> ExtractExtensions(std::string* text) {
  std::vector<std::string> extensions;
  size_t start = 0;
  while ((start = text->find("#extension", start)) != std::string::npos) {
    const size_t end = text->find('\n', start);
    extensions.push_back(text->substr(start, end - start));
    text->erase(start, end - start + 1);  // +1 to remove the newline as well
  }
  return extensions;
}

void MoveExtensionsToBeginning(std::string* text) {
  const auto extensions = ExtractExtensions(text);
  for (const auto& extension : extensions) {
    text->insert(0, extension + "\n");
  }
}

std::string GetCommonOpenGLDefines() {
  std::string result;
  result += "precision highp int;\n";
  result += "#define MAIN_FUNCTION $1 void main\n";
  result += "#define fabs abs\n";
  result += "#define rsqrt inversesqrt\n";
  result += "#define int2 ivec2\n";
  result += "#define int4 ivec4\n";
  result += "#define uint2 uvec2\n";
  result += "#define uint4 uvec4\n";
  result += "#define short mediump int\n";
  result += "#define short2 mediump ivec2\n";
  result += "#define short4 mediump ivec4\n";
  result += "#define ushort mediump uint\n";
  result += "#define ushort2 mediump uvec2\n";
  result += "#define ushort4 mediump uvec4\n";
  result += "#define char lowp int\n";
  result += "#define char2 lowp ivec2\n";
  result += "#define char4 lowp ivec4\n";
  result += "#define uchar lowp uint\n";
  result += "#define uchar2 lowp uvec2\n";
  result += "#define uchar4 lowp uvec4\n";
  result += "#define half mediump float\n";
  result += "#define half2 mediump vec2\n";
  result += "#define half3 mediump vec3\n";
  result += "#define half4 mediump vec4\n";
  result += "#define atan2 atan\n";
  result += "#define float2 highp vec2\n";
  result += "#define float3 highp vec3\n";
  result += "#define float4 highp vec4\n";
  result += "#define float16 mat4x4\n";
  result += "#define half16 mediump mat4x4\n";
  result += "#define bool4 bvec4\n";
  result += "#define select(a, b, c) ((c) ? (b) : (a))\n";
  result += "#define mad24(a, b, c) ((a) * (b) + (c))\n";
  result += "struct uvec8 { uvec4 a; uvec4 b;};\n";
  return result;
}

std::string GetGlslVersion(const GpuInfo& gpu_info) {
  std::string glsl_version;
  if (gpu_info.opengl_info.IsApiOpenGl32OrAbove()) {
    return "#version 320 es";
  } else {
    return "#version 310 es";
  }
}

absl::Status CheckProgramLinked(GLuint program_id) {
  GLint linked;
  glGetProgramiv(program_id, GL_LINK_STATUS, &linked);
  if (linked == GL_TRUE) {
    return absl::OkStatus();
  }
  GLint info_size;
  glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_size);
  std::string errors;
  errors.resize(info_size + 1 /* plus \0 */);
  glGetProgramInfoLog(program_id, info_size + 1, nullptr, &errors[0]);
  return absl::UnavailableError("Program is not properly linked: " + errors);
}

absl::Status ProgramFromCode(const std::string& code, GLuint* program) {
  *program = glCreateProgram();
  if (!(*program)) {
    return absl::UnavailableError("Can't create program.");
  }
  gl::GlShader compute_shader;
  ABSL_RETURN_IF_ERROR(
      gl::GlShader::CompileShader(GL_COMPUTE_SHADER, code, &compute_shader));
  glAttachShader(*program, compute_shader.id());
  glLinkProgram(*program);
  return CheckProgramLinked(*program);
}

absl::Status ProgramFromBinary(const std::vector<uint8_t>& program_binary,
                               uint32_t binary_format, GLuint* program) {
  *program = glCreateProgram();
  if (!(*program)) {
    return absl::UnavailableError("Can't create program.");
  }

  glProgramBinary(*program, binary_format, program_binary.data(),
                  program_binary.size());
  return CheckProgramLinked(*program);
}

}  // namespace

absl::Status ProgramCache::GetProgram(const std::string& code,
                                      uint64_t fingerprint, GLuint* result) {
  auto it = programs_cache_.find(fingerprint);
  if (it != programs_cache_.end()) {
    *result = it->second;
    return absl::OkStatus();
  } else {
    ABSL_RETURN_IF_ERROR(ProgramFromCode(code, result));
    programs_cache_[fingerprint] = *result;
    return absl::OkStatus();
  }
}

absl::Status ProgramCache::GetProgram(
    const std::vector<uint8_t>& program_binary, uint32_t binary_format,
    uint64_t fingerprint, GLuint* result) {
  auto it = programs_cache_.find(fingerprint);
  if (it != programs_cache_.end()) {
    *result = it->second;
    return absl::OkStatus();
  } else {
    ABSL_RETURN_IF_ERROR(
        ProgramFromBinary(program_binary, binary_format, result));
    programs_cache_[fingerprint] = *result;
    return absl::OkStatus();
  }
}

GlOperation::GlOperation(GlOperation&& operation)
    : operation_(std::move(operation.operation_)),
      program_(std::move(operation.program_)),
      owner_(operation.owner_),
      gl_args_(std::move(operation.gl_args_)),
      fingerprint_(operation.fingerprint_) {
  operation.program_ = -1;
}

GlOperation& GlOperation::operator=(GlOperation&& operation) {
  if (this != &operation) {
    Release();
    operation_ = std::move(operation.operation_);
    std::swap(program_, operation.program_);
    owner_ = operation.owner_;
    gl_args_ = std::move(operation.gl_args_);
    fingerprint_ = operation.fingerprint_;
  }
  return *this;
}

void GlOperation::Release() {
  if (owner_ && program_ != -1) {
    glDeleteProgram(program_);
    program_ = -1;
  }
}

absl::Status GlOperation::InitArgs(const GpuInfo& gpu_info) {
  ABSL_RETURN_IF_ERROR(
      gl_args_.Init(gpu_info, &operation_->args_, &operation_->code_));
  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status GlOperation::InitArgsDeserialized(const GpuInfo& gpu_info) {
  ABSL_RETURN_IF_ERROR(gl_args_.Init(gpu_info, &operation_->args_));
  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status GlOperation::UpdateParams() {
  ABSL_RETURN_IF_ERROR(operation_->BindArguments(&gl_args_));
  operation_->RecalculateGridSize();
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

absl::Status GlOperation::Tune(const GpuInfo& gpu_info) {
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = gpu_info.GetMaxWorkGroupTotalSize();
  kernel_info.private_memory_size = 0;
  std::vector<GPUOperation::DispatchInfo> possible_dispatches;
  operation_->GetPossibleDispatches(TuningType::kFast, gpu_info, kernel_info,
                                    &possible_dispatches);
  if (possible_dispatches.empty()) {
    return absl::NotFoundError("No dispatch parameters to launch program");
  }
  operation_->work_group_size_ = possible_dispatches[0].work_group_size;
  operation_->RecalculateWorkGroupsCount();

  return absl::OkStatus();
}

absl::Status GlOperation::Assemble(const GpuInfo& gpu_info,
                                   ProgramCache* program_cache,
                                   const int3* work_group_size) {
  operation_->code_ = GetCommonOpenGLDefines() + operation_->code_;
  MoveExtensionsToBeginning(&operation_->code_);
  if (work_group_size) {
    operation_->work_group_size_ = *work_group_size;
  } else {
    ABSL_RETURN_IF_ERROR(UpdateParams());
    ABSL_RETURN_IF_ERROR(Tune(gpu_info));
  }
  const std::string shader_wg_size = absl::Substitute(
      "layout(local_size_x = $0, local_size_y = $1, local_size_z = $2) in;",
      operation_->work_group_size_.x, operation_->work_group_size_.y,
      operation_->work_group_size_.z);
  operation_->code_ = absl::Substitute(operation_->code_, "", shader_wg_size);
  operation_->code_ = GetGlslVersion(gpu_info) + "\n" + operation_->code_;
  ABSL_RETURN_IF_ERROR(Compile(program_cache));
  glUseProgram(program_);
  gl_args_.GetLocations(program_);
  return absl::OkStatus();
}

absl::Status GlOperation::Compile(ProgramCache* program_cache) {
  fingerprint_ = ::util::Fingerprint64(operation_->code_);

  if (program_cache) {
    owner_ = false;
    return program_cache->GetProgram(operation_->code_, fingerprint_,
                                     &program_);
  } else {
    return ProgramFromCode(operation_->code_, &program_);
  }
}

absl::Status GlOperation::Init(const std::vector<uint8_t>& program_binary,
                               uint32_t binary_format, uint64_t fingerprint,
                               ProgramCache* program_cache) {
  fingerprint_ = fingerprint;
  if (program_cache) {
    owner_ = false;
    return program_cache->GetProgram(program_binary, binary_format,
                                     fingerprint_, &program_);
  } else {
    return ProgramFromBinary(program_binary, binary_format, &program_);
  }
}

absl::Status GlOperation::GetProgramBinary(std::vector<uint8_t>* data,
                                           uint32_t* binary_format) const {
  GLint size = 0;
  glGetProgramiv(program_, GL_PROGRAM_BINARY_LENGTH, &size);
  if (!size) {
    return absl::InternalError("Getting binary size failed.");
  }
  data->resize(size);
  GLsizei returned_size;
  GLenum format;
  glGetProgramBinary(program_, size, &returned_size, &format, data->data());
  if (size != returned_size) {
    return absl::InternalError("Getting binary is failed.");
  }
  *binary_format = format;
  return absl::OkStatus();
}

absl::Status GlOperation::RestoreDeserialized(const GpuInfo& gpu_info) {
  ABSL_RETURN_IF_ERROR(UpdateParams());
  operation_->RecalculateWorkGroupsCount();
  glUseProgram(program_);
  gl_args_.GetLocations(program_);
  return absl::OkStatus();
}

void GlOperation::SetWorkGroupSize(const int3& work_group_size) {
  operation_->work_group_size_ = work_group_size;
  operation_->RecalculateWorkGroupsCount();
}

absl::Status GlOperation::AddToQueue() {
  glUseProgram(program_);
  gl_args_.Bind();
  glDispatchCompute(operation_->GetWorkGroupsCount().x,
                    operation_->GetWorkGroupsCount().y,
                    operation_->GetWorkGroupsCount().z);
  return absl::OkStatus();
}

absl::Status GlOperation::SetSrcTensor(GlSpatialTensor* tensor, int index) {
  operation_->SetSrc(tensor, index);
  return gl_args_.SetObjectRef(operation_->GetSrcTensorsNames()[index], tensor);
}

absl::Status GlOperation::SetDstTensor(GlSpatialTensor* tensor, int index) {
  operation_->SetDst(tensor, index);
  return gl_args_.SetObjectRef(operation_->GetDstTensorsNames()[index], tensor);
}

absl::Status GlOperation::SetSrcBuffer(GlBuffer* buffer, int index) {
  return gl_args_.SetObjectRef(operation_->GetSrcTensorsNames()[index], buffer);
}

absl::Status GlOperation::SetDstBuffer(GlBuffer* buffer, int index) {
  return gl_args_.SetObjectRef(operation_->GetDstTensorsNames()[index], buffer);
}

absl::StatusOr<absl::Duration> GlOperation::GetOperationTime() {
  const int kMaxRuns = 5000;
  const int kMaxIterations = 5;
  const double kConvergeTolerance = 10.0;  // in percents
  const double kTestRunMs = 100.0;
  absl::Duration duration = absl::ZeroDuration();
  double prev_time_ms = 10000.0f;  // 10sec
  for (int i = 0; i < kMaxIterations; ++i) {
    int num_runs = static_cast<int>(kTestRunMs / prev_time_ms);
    num_runs = std::min(std::max(num_runs, 4), kMaxRuns);
    const int flush_runs = std::max(4, num_runs / 10);
    auto start = absl::Now();
    for (int j = 0; j < num_runs; ++j) {
      ABSL_RETURN_IF_ERROR(AddToQueue());
      if (i % flush_runs == flush_runs - 1) {
        glFlush();
      }
    }
    glFlush();
    glFinish();
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

}  // namespace gl
}  // namespace ml_drift
