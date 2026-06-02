// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_WEBGPU_WEBGPU_API_UTIL_H_
#define ML_DRIFT_WEBGPU_WEBGPU_API_UTIL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/executor.h"
#include "ml_drift/common/status.h"
#include "ml_drift/webgpu/webgpu_headers.h"

#ifndef __EMSCRIPTEN__
#include "absl/synchronization/mutex.h"
#endif

namespace ml_drift {
namespace webgpu {

// ReadDataFromBuffer/ReadDataFromTexture/WaitUntilCompleted will not work
// by default with emscripten.
// To make it work in emscripten ASYNCIFY must be used.

// buffer usage must include MapRead or CopySrc
absl::Status ReadDataFromBuffer(const wgpu::Device& device,
                                const wgpu::Queue& queue,
                                const wgpu::Buffer& buffer, uint64_t data_size,
                                void* data_ptr);

absl::Status ReadDataFromTexture(const wgpu::Device& device,
                                 const wgpu::Queue& queue,
                                 const wgpu::Texture& texture,
                                 int bytes_per_pixel, int width, int height,
                                 int depth, void* data_ptr);

// Non blocking
void WriteDataToBuffer(const wgpu::Queue& queue, const wgpu::Buffer& buffer,
                       uint64_t data_size, const void* data_ptr);

// Non blocking
void WriteDataToTexture(const wgpu::Queue& queue, const wgpu::Texture& texture,
                        int bytes_per_pixel, int width, int height, int depth,
                        const void* data_ptr);

// wgpu::Device parameter needed for Dawn implementation
absl::Status WaitUntilCompleted(const wgpu::Queue& queue,
                                const wgpu::Device& device,
                                absl::Duration timeout);

wgpu::ShaderModule CreateComputeShaderModule(const wgpu::Device& device,
                                             const std::string& src);

// Encapsulates the logic to create a pipeline asynchronously and wait for it to
// be created.
class ComputePipelineHolder {
 public:
  ComputePipelineHolder(
      const wgpu::Device& device, const std::string& code,
      const std::vector<wgpu::ConstantEntry>& pipeline_constants,
      const std::string& entry_point_name, const wgpu::PipelineLayout& layout,
      Executor* executor, bool async_create_call = false);
  ~ComputePipelineHolder();

  // Not copyable or movable.
  ComputePipelineHolder(const ComputePipelineHolder&) = delete;
  ComputePipelineHolder& operator=(const ComputePipelineHolder&) = delete;

  // Waits for the pipeline to be created and returns it when ready.
  absl::StatusOr<wgpu::ComputePipeline> Get() const;

 protected:
  struct ThreadData {
    // TODO: b/418039926 - Remove these after debugging the crash.
    std::string code;
    absl::Time start;
    std::string debug_str;

    absl::StatusOr<wgpu::ComputePipeline> pipeline;
    std::function<void()> fn;
#ifdef __EMSCRIPTEN__
    wgpu::Future pipeline_future = {};
#else
    absl::Mutex mutex;
    enum class State {
      kUnstarted,
      kRunning,
      kDone,
    } state ABSL_GUARDED_BY(mutex) = State::kUnstarted;
#endif  // __EMSCRIPTEN__
  };
  std::shared_ptr<ThreadData> thread_data_ = std::make_shared<ThreadData>();
  bool async_create_call_;
};

absl::StatusOr<std::unique_ptr<ComputePipelineHolder>> CreateComputePipeline(
    const wgpu::Device& device, const std::string& code,
    const std::vector<wgpu::ConstantEntry>& pipeline_constants,
    const std::string& entry_point_name, const wgpu::PipelineLayout& layout,
    Executor* executor, bool async_create_call = false);

wgpu::TextureFormat DataTypeToTextureFormat(DataType data_type,
                                            int channels_count = 4);

absl::Status VerifyQueueWorkDoneStatus(wgpu::QueueWorkDoneStatus status);

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_WEBGPU_API_UTIL_H_
