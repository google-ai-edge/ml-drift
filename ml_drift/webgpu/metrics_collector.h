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

#ifndef ML_DRIFT_WEBGPU_METRICS_COLLECTOR_H_
#define ML_DRIFT_WEBGPU_METRICS_COLLECTOR_H_

#include <memory>
#include <vector>

#include "absl/time/time.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

// This structure allows to specify which metrics collection to initiate. Only
// a single series of metrics can be chosen at a time.
struct RequestedMetrics {
  // Time between starting first compute pass and after finishing the last one.
  // Shaders are grouped into a single command buffer and executed as a batch.
  bool total_inference_time = false;

  // Time, which every shader takes independently. Shaders are split into
  // separate command buffers, which are executed sequentially.
  bool per_operation_profiling = false;
};

// MetricsCollector object collects various metrics, analyzes them, and provides
// accessors to them. Currently it can measure the total inference time and give
// the per operation profiling.
class MetricsCollector {
 public:
  virtual ~MetricsCollector() = default;
  // Returns the inference execution time.
  virtual absl::Duration GetTotalInferenceTime() = 0;
  // Returns durations for every consequent operation.
  virtual std::vector<absl::Duration> GetPerOperationProfiling() = 0;
  virtual void SetEnvironment(const Environment* environment) = 0;
};

std::unique_ptr<MetricsCollector> NewMetricsCollector(
    const RequestedMetrics& requested_metrics);

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_METRICS_COLLECTOR_H_
