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

#ifndef ML_DRIFT_COMMON_TASK_PROFILING_INFO_H_
#define ML_DRIFT_COMMON_TASK_PROFILING_INFO_H_

#include <stdbool.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

struct ProfilingInfo {
  struct DispatchInfo {
    std::string label;
    absl::Duration duration;
    uint64_t read_mem_size = 0;
    uint64_t write_mem_size = 0;
    uint64_t flops = 0;
    std::vector<Shape> inshape;
    std::vector<Shape> outshape;
  };

  struct DetailedReportOptions {
    bool add_shapes_info = false;
  };

  std::vector<DispatchInfo> dispatches;

  absl::Duration GetTotalTime() const;

  // Returns report (string of lines delimited by \n)
  // This method uses GPU counters and measure GPU time only.
  // Report has next structure:
  // Per kernel timing(K kernels):
  //   conv2d 3.2ms
  //   ...
  // --------------------
  // Accumulated time per operation type:
  //   conv2d - 14.5ms
  //   ....
  // --------------------
  // Ideal total time: 23.4ms // Total time for all kernels
  // add_shapes_info option: if True adds columns in the report for
  // input and output tensor shapes for each node. By default this is
  // set to false.
  std::string GetDetailedReport(
      const DetailedReportOptions& options = DetailedReportOptions{
          /*add_shapes_info=*/false}) const;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_PROFILING_INFO_H_
