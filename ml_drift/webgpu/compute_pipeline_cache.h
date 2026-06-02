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

#ifndef ML_DRIFT_WEBGPU_COMPUTE_PIPELINE_CACHE_H_
#define ML_DRIFT_WEBGPU_COMPUTE_PIPELINE_CACHE_H_

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "ml_drift/common/executor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/webgpu/preprocessor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"
#include <farmhash.h>

namespace ml_drift {
namespace webgpu {

class ComputePipelineCache {
 public:
  // Returns the pipeline from the cache if it exists. Otherwise, creates a new
  // pipeline and adds it to the cache.
  absl::StatusOr<ComputePipelineHolder*> GetOrCreatePipeline(
      const wgpu::Device& device, const std::string& code,
      const std::vector<wgpu::ConstantEntry>& pipeline_constants,
      const std::string& entry_point_name, const wgpu::PipelineLayout& layout,
      bool use_async_create_call, const uint64_t fingerprint) {
    auto it = cached_data_.find(fingerprint);
    if (it != cached_data_.end()) {
      return it->second.get();
    } else {
      ASSIGN_OR_RETURN(auto pipeline,
                       CreateComputePipeline(
                           device, code, pipeline_constants, entry_point_name,
                           layout, executor_.get(), use_async_create_call));
      return cached_data_.insert({fingerprint, std::move(pipeline)})
          .first->second.get();
    }
  }

  absl::Status ConvertToWGSL(const WebGpuInfo& webgpu_info, std::string* code,
                             ExtensionsInfo* extensions_info) {
    const uint64_t hash = ::util::Fingerprint64(*code);
    auto it = cached_code_.find(hash);
    if (it != cached_code_.end()) {
      *code = it->second.code;
      *extensions_info = it->second.extensions_info;
    } else {
      RETURN_IF_ERROR(
          ml_drift::webgpu::ConvertToWGSL(webgpu_info, code, extensions_info));
      cached_code_[hash].code = *code;
      cached_code_[hash].extensions_info = *extensions_info;
    }
    return absl::OkStatus();
  }

  void set_executor(std::unique_ptr<Executor> executor) {
    executor_ = std::move(executor);
  }

 private:
  absl::flat_hash_map<uint64_t, std::unique_ptr<ComputePipelineHolder>>
      cached_data_;
  struct CodeDesc {
    std::string code;
    ExtensionsInfo extensions_info;
  };
  absl::flat_hash_map<uint64_t, CodeDesc> cached_code_;
  std::unique_ptr<Executor> executor_;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_COMPUTE_PIPELINE_CACHE_H_
