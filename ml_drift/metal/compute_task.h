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

#ifndef ML_DRIFT_METAL_COMPUTE_TASK_H_
#define ML_DRIFT_METAL_COMPUTE_TASK_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"
#include "ml_drift/metal/buffer.h"
#include "ml_drift/metal/common.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_arguments.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {

class ComputeTask {
 public:
  ComputeTask() = default;
  ~ComputeTask();

  // Move only
  ComputeTask(ComputeTask&& task);
  ComputeTask& operator=(ComputeTask&& task);
  ComputeTask(const ComputeTask&) = delete;
  ComputeTask& operator=(const ComputeTask&) = delete;

  void Init(std::unique_ptr<GPUOperation>&& operation,
            bool use_argument_buffer = false);

  absl::Status InitArgs(Environment* env);
  absl::Status InitArgsDeserialized(Environment* env);

  absl::Status Compile(Environment* env);

  // should be called after changes of inputs/outputs.
  absl::Status UpdateParams();

  absl::Status SetInt(const std::string& name, int value) {
    return metal_args_.SetInt(name, value);
  }
  absl::Status SetUint(const std::string& name, uint value) {
    return metal_args_.SetUint(name, value);
  }
  absl::Status SetFloat(const std::string& name, float value) {
    return metal_args_.SetFloat(name, value);
  }
  absl::Status SetHalf(const std::string& name, float value) {
    return metal_args_.SetHalf(name, half(value));
  }

  void Encode(id<MTLComputeCommandEncoder> encoder);

  API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
  void EncodeToICB(id<MTLIndirectComputeCommand> icb_command);
  void AddResourcesToEncoder(id<MTLComputeCommandEncoder> encoder) const;

  void SetSrcTensor(MetalSpatialTensor* tensor, int index);
  void SetDstTensor(MetalSpatialTensor* tensor, int index);
  void SetSrcBuffer(Buffer* buffer, int index);
  void SetDstBuffer(Buffer* buffer, int index);

  absl::Status Tune(TuningType tuning_type, Environment* env);

  absl::Duration GetTaskTime(id<MTLCommandQueue> command_queue);

  int3 GetWorkGroupSize() const { return operation_->work_group_size_; }
  void SetWorkGroupSize(const int3& work_group_size);

  uint64_t GetConstArgsSize() const { return operation_->const_args_size_; }
  uint64_t GetFlopsCount() const { return operation_->flops_; }
  uint64_t GetReadSize() const { return operation_->read_size_; }
  uint64_t GetWriteSize() const { return operation_->write_size_; }

  const std::string& GetCode() const { return operation_->code_; }
  const std::map<std::string, std::string>& GetDefines() const {
    return defines_;
  }

  absl::Status Init(Environment* env, const std::string& code,
                    const std::map<std::string, std::string>& defines);
  absl::Status RestoreDeserialized(Environment* env);

 private:
  void UpdateArgumentBuffer();

  absl::Status CompileProgram(
      Environment* env, const std::string& code,
      const std::map<std::string, std::string>& defines);
  void Release();

  std::unique_ptr<GPUOperation> operation_;
  id<MTLComputePipelineState> program_ = nullptr;
  MetalArguments metal_args_;

  bool use_arguments_buffer_ = false;  // optional
  bool need_icb_support_ = false;      // optional
  id<MTLArgumentEncoder> arguments_encoder_ = nullptr;
  id<MTLBuffer> arg_buffer_ = nullptr;

  // for serialization
  std::map<std::string, std::string> defines_;
};

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_COMPUTE_TASK_H_
