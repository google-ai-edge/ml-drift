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

#ifndef ML_DRIFT_WEBGPU_COMPUTE_TASK_H_
#define ML_DRIFT_WEBGPU_COMPUTE_TASK_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "ml_drift/common/gpu_info.h"
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
#include "ml_drift/webgpu/webgpu_headers.h"
#include <farmhash.h>

namespace ml_drift {
namespace webgpu {

class ComputeTask {
 public:
  ComputeTask() = default;
  ~ComputeTask() { Release(); }
  // Move only
  ComputeTask(ComputeTask&& task) = default;
  ComputeTask& operator=(ComputeTask&& task) = default;
  ComputeTask(const ComputeTask&) = delete;
  ComputeTask& operator=(const ComputeTask&) = delete;

  absl::Status Init(const Environment& env,
                    std::unique_ptr<GPUOperation>&& gpu_operation,
                    UniformBufferCreator* uniform_buffer_creator = nullptr,
                    bool from_serialized_model = false) {
    operation_ = std::move(gpu_operation);
    // TODO: The ComputePipelineCache should be able to be used
    // in all cases, but there seems to be some problem with this (see linked
    // bug). Switch this to always use the cache once the bug is resolved.

    if (from_serialized_model) {
      return webgpu_args_.InitWithoutCodeGeneration(env.device(), env.GetInfo(),
                                                    &operation_->args_,
                                                    uniform_buffer_creator);
    }
    return webgpu_args_.Init(env.device(), env.GetInfo(), &operation_->args_,
                             &operation_->code_, uniform_buffer_creator);
  }

  GPUOperation& GetGpuOperation() { return *operation_; }
  const GPUOperation& GetGpuOperation() const { return *operation_; }

  uint64_t GetConstArgsSize() const { return operation_->const_args_size_; }
  uint64_t GetFlopsCount() const { return operation_->flops_; }
  uint64_t GetReadSize() const { return operation_->read_size_; }
  uint64_t GetWriteSize() const { return operation_->write_size_; }

  // Must be called after binding.
  absl::Status TuneAndCompile(
      const Environment& env, TuningType tuning_type,
      ComputePipelineCache* compute_pipeline_cache = nullptr);
  // Can be called without binding. But work_group_size must be valid for this
  // op.
  absl::Status Compile(const Environment& env,
                       ComputePipelineCache* compute_pipeline_cache = nullptr);

  // Restores the deserialized pipeline. If `compute_pipeline_cache` is
  // provided, the pipeline will be restored from the cache if possible.
  // Otherwise, the pipeline will be created using the code.
  absl::Status RestoreDeserialized(
      const Environment& env, std::string& code, uint64_t fingerprint,
      ComputePipelineCache* compute_pipeline_cache = nullptr);

  absl::Status SetSrcTensor(int index, SpatialTensor* tensor);
  absl::Status SetDstTensor(int index, SpatialTensor* tensor);
  absl::Status SetSrcBuffer(int index, Buffer* buffer);
  absl::Status SetDstBuffer(int index, Buffer* buffer);

  absl::Status SetInt(const std::string& name, int value) {
    return webgpu_args_.SetInt(name, value);
  }
  absl::Status SetUint(const std::string& name, unsigned int value) {
    return webgpu_args_.SetUint(name, value);
  }
  absl::Status SetFloat(const std::string& name, float value) {
    return webgpu_args_.SetFloat(name, value);
  }
  absl::Status SetHalf(const std::string& name, float value) {
    return webgpu_args_.SetHalf(name, half(value));
  }

  // Must be called after all objects set, objects can be partially
  // initialized (have nullptr for wgpu::Buffer/wgpu::Texture)
  // Update() must be called before Compile()
  absl::Status Update(const wgpu::Device& device);
  // Must be called after all objects fully initialized, gpu
  // handles(wgpu::Buffer/wgpu::Texture) are set
  // UpdateGpuObjectBindings() must be called after Update()
  // UpdateGpuObjectBindings() must be called before Encode() at least once
  // UpdateGpuObjectBindings() can be called multiple times(if gpu handles
  // updated) after Compile()
  //         SetSrcTensor()/SetDstTensor()(once)
  //                      ||
  //                      \/
  //                   Update()(once)
  //                      ||
  //                      \/
  //        UpdateGpuObjectBindings()(optional)
  //                      ||                   |
  //                      \/                    |
  //                   Compile()(once)         must be called at least once
  //                      ||                    /     before Encode
  //                      \/                   /
  //   UpdateGpuObjectBindings()(optional, multiple)
  //                      ||
  //                      \/
  //                   Encode()(multiple)
  void UpdateGpuObjectBindings(const wgpu::Device& device);
  absl::Status Encode(const wgpu::ComputePassEncoder& compute_encoder) const;
  absl::Status Execute(const Environment& env);

  absl::StatusOr<absl::Duration> GetOperationTime(const Environment& env);

  void CopyFrom(const ComputeTask& other);

  // Returns the work group size of the GPU operation.
  const int3& GetWorkGroupSize() const { return operation_->work_group_size_; }
  // Sets the work group size of the GPU operation.
  void SetWorkGroupSize(const int3& work_group_size);
  // Returns the code of the GPU operation.
  const std::string& GetCode() const { return operation_->code_; }
  // Returns the fingerprint of the GPU operation.
  const uint64_t& GetKernelFingerprint() const { return code_fingerprint_; }

 private:
  uint64_t GetFullFingerprint() {
    const std::string wg_size_str =
        absl::StrCat("wg_size<", operation_->work_group_size_.x, "x",
                     operation_->work_group_size_.y, "x",
                     operation_->work_group_size_.z, ">");
    const uint64_t wg_size_fingerprint = ::util::Fingerprint64(wg_size_str);
    return ::util::Fingerprint(
        ::util::uint128_t(code_fingerprint_, wg_size_fingerprint));
  }
  absl::Status Tune(const Environment& env, TuningType tuning_type);
  absl::Status CompileToWGSL(
      const Environment& env,
      ComputePipelineCache* compute_pipeline_cache = nullptr);
  absl::Status CompileWGSLToPipeline(
      const Environment& env,
      ComputePipelineCache* compute_pipeline_cache = nullptr);
  void AddGlobalDeclarations(const WebGpuInfo& webgpu_info,
                             const ExtensionsInfo& extensions_info);
  void Release() {}
  // Subgraph ops may share the underlying GPUOperation.
  std::shared_ptr<GPUOperation> operation_;
  WebGpuArguments webgpu_args_;

  // This will either be owned by |owned_compute_pipeline_| or
  // ComputePipelineCache, which will outlive this object.
  ComputePipelineHolder* compute_pipeline_ = nullptr;
  std::unique_ptr<ComputePipelineHolder> owned_compute_pipeline_;
  uint64_t code_fingerprint_;  // code fingerprint
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_COMPUTE_TASK_H_
