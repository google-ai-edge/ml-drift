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

#ifndef ML_DRIFT_GL_GL_OPERATION_H_
#define ML_DRIFT_GL_GL_OPERATION_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gl_arguments.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_spatial_tensor.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {

class ProgramCache {
 public:
  ProgramCache() = default;
  ~ProgramCache() {
    for (const auto& item : programs_cache_) {
      glDeleteProgram(item.second);
    }
  }
  // Move only
  ProgramCache(ProgramCache&&) = default;
  ProgramCache& operator=(ProgramCache&&) = default;
  ProgramCache(const ProgramCache&) = delete;
  ProgramCache& operator=(const ProgramCache&) = delete;

  absl::Status GetProgram(const std::string& code, uint64_t fingerprint,
                          GLuint* result);
  absl::Status GetProgram(const std::vector<uint8_t>& program_binary,
                          uint32_t binary_format, uint64_t fingerprint,
                          GLuint* result);

 private:
  std::map<uint64_t, GLuint> programs_cache_;
};

class GlOperation {
 public:
  GlOperation() = default;
  virtual ~GlOperation() { Release(); }
  // Move only
  GlOperation(GlOperation&& operation);
  GlOperation& operator=(GlOperation&& operation);
  GlOperation(const GlOperation&) = delete;
  GlOperation& operator=(const GlOperation&) = delete;

  void Release();

  void Init(std::unique_ptr<GPUOperation>&& gpu_operation) {
    operation_ = std::move(gpu_operation);
  }

  absl::Status InitArgs(const GpuInfo& gpu_info);
  absl::Status InitArgsDeserialized(const GpuInfo& gpu_info);

  absl::Status AddOperation(GlOperation* operation);

  absl::Status Tune(const GpuInfo& gpu_info);

  // should be called after changes of inputs/outputs.
  absl::Status UpdateParams();

  absl::Status SetInt(const std::string& name, int value) {
    return gl_args_.SetInt(name, value);
  }
  absl::Status SetUint(const std::string& name, uint value) {
    return gl_args_.SetUint(name, value);
  }
  absl::Status SetFloat(const std::string& name, float value) {
    return gl_args_.SetFloat(name, value);
  }
  absl::Status SetHalf(const std::string& name, float value) {
    return gl_args_.SetHalf(name, half(value));
  }

  absl::Status SetSrcTensor(GlSpatialTensor* tensor, int index);
  absl::Status SetDstTensor(GlSpatialTensor* tensor, int index);
  absl::Status SetSrcBuffer(GlBuffer* buffer, int index);
  absl::Status SetDstBuffer(GlBuffer* buffer, int index);

  absl::Status AddToQueue();

  // if work_group_size not specified, src/dst objects must be binded before
  // this call to determine optimal work group size automatically
  absl::Status Assemble(const GpuInfo& gpu_info,
                        ProgramCache* program_cache = nullptr,
                        const int3* work_group_size = nullptr);
  absl::Status Compile(ProgramCache* program_cache = nullptr);

  absl::Status GetProgramBinary(std::vector<uint8_t>* data,
                                uint32_t* binary_format) const;
  uint64_t GetFingerprint() const { return fingerprint_; }
  absl::Status Init(const std::vector<uint8_t>& program_binary,
                    uint32_t binary_format, uint64_t fingerprint,
                    ProgramCache* program_cache = nullptr);
  absl::Status RestoreDeserialized(const GpuInfo& gpu_info);
  int3 GetWorkGroupSize() const { return operation_->work_group_size_; }
  void SetWorkGroupSize(const int3& work_group_size);

  uint64_t GetConstArgsSize() const { return operation_->const_args_size_; }
  uint64_t GetFlopsCount() const { return operation_->flops_; }
  uint64_t GetReadSize() const { return operation_->read_size_; }
  uint64_t GetWriteSize() const { return operation_->write_size_; }

  absl::StatusOr<absl::Duration> GetOperationTime();

 private:
  std::unique_ptr<GPUOperation> operation_;
  GLuint program_ = -1;
  bool owner_ = true;
  GlArguments gl_args_;
  uint64_t fingerprint_;
};

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_OPERATION_H_
