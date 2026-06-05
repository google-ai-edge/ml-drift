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

#ifndef ML_DRIFT_WEBGPU_METRICS_COLLECTOR_INTERNAL_H_
#define ML_DRIFT_WEBGPU_METRICS_COLLECTOR_INTERNAL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ml_drift/common/status.h"
#include "ml_drift/webgpu/metrics_collector.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

// Represents metrics collection interface for the internal use of the WebGPU
// inference framework.
class InternalMetricsCollector : public MetricsCollector {
 public:
  // Returns WebGPU extension required for working with Queries.
  virtual std::string GetExtension() = 0;
  // Initializes WebGPU objects required for metrics collection.
  virtual absl::Status Init(const std::vector<std::string>& op_names) = 0;
  // Resolves query set and submits data read back command.
  virtual void ResolveQueries(const wgpu::CommandEncoder* encoder) = 0;
  // Active waiting for query buffers readiness.
  virtual absl::Status WaitQueriesReadBack() = 0;
  // Returns the descriptor used to initialize compute pass with timestamps info
  // The operation_position is respected in per op profiling only.
  virtual wgpu::ComputePassDescriptor GetComputePassDescriptor(
      std::optional<uint32_t> operation_position) = 0;
  // Notifies inference context of the amount of command buffers to allocate.
  virtual bool IsSingleCommandBuffer() = 0;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_METRICS_COLLECTOR_INTERNAL_H_
