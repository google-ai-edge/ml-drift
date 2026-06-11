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

#include "ml_drift/common/task/profiling_info.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

absl::Duration ProfilingInfo::GetTotalTime() const {
  absl::Duration total_time;
  for (const auto& dispatch : dispatches) {
    total_time += dispatch.duration;
  }
  return total_time;
}

std::string ProfilingInfo::GetDetailedReport(
    const DetailedReportOptions& options) const {
  std::string result;
  struct OpStatistic {
    int count = 0;
    absl::Duration total_time = absl::ZeroDuration();
    uint64_t total_flops = 0;
    uint64_t total_bytes = 0;
  };
  std::map<std::string, OpStatistic> statistics;
  result +=
      "Per kernel timing(" + std::to_string(dispatches.size()) + " kernels):\n";
  for (const auto& dispatch : dispatches) {
    result += "  " + dispatch.label + "; " +
              std::to_string(absl::ToDoubleMilliseconds(dispatch.duration)) +
              " ms" + "; ";
    if (options.add_shapes_info) {
      for (const auto& curr_shape : dispatch.inshape) {
        result += ToString(curr_shape) + "- ";
      }
      result += "; ";
      for (const auto& curr_shape : dispatch.outshape) {
        result += ToString(curr_shape) + "-";
      }
      result += "; ";
    }
    const double times_per_sec =
        1000.0 / absl::ToDoubleMilliseconds(dispatch.duration);
    if (dispatch.read_mem_size || dispatch.write_mem_size) {
      const uint64_t total_size =
          dispatch.read_mem_size + dispatch.write_mem_size;
      const double giga_bytes = total_size / 1024.0 / 1024.0 / 1024.0;
      const double giga_bytes_per_sec = times_per_sec * giga_bytes;
      result += std::to_string(giga_bytes_per_sec) + " Gb/s";
    }
    if (dispatch.flops) {
      const double giga_flops = dispatch.flops / 1000.0 / 1000.0 / 1000.0;
      const double giga_flops_per_sec = times_per_sec * giga_flops;
      result += "; " + std::to_string(giga_flops_per_sec) + " Gflops";
    }
    result += \n\;
    auto name = dispatch.label.substr(0, dispatch.label.find(' '));
    statistics[name].count++;
    statistics[name].total_time += dispatch.duration;
    statistics[name].total_flops += dispatch.flops;
    statistics[name].total_bytes +=
        (dispatch.read_mem_size + dispatch.write_mem_size);
  }
  result += "--------------------\n";
  result += "Accumulated time per operation type:\n";
  uint64_t total_flops = 0;
  uint64_t total_bytes = 0;
  for (auto& t : statistics) {
    auto stat = t.second;
    result += "  " + t.first + "(x" + std::to_string(stat.count) + ") - " +
              std::to_string(absl::ToDoubleMilliseconds(stat.total_time)) +
              " ms";
    const double times_per_sec =
        1000.0 / absl::ToDoubleMilliseconds(stat.total_time);
    if (stat.total_bytes) {
      const double giga_bytes = stat.total_bytes / 1024.0 / 1024.0 / 1024.0;
      const double giga_bytes_per_sec = times_per_sec * giga_bytes;
      result += ", " + std::to_string(giga_bytes_per_sec) + " Gb/s";
      total_bytes += stat.total_bytes;
    }
    if (stat.total_flops) {
      const double giga_flops = stat.total_flops / 1000.0 / 1000.0 / 1000.0;
      const double giga_flops_per_sec = times_per_sec * giga_flops;
      result += ", " + std::to_string(giga_flops_per_sec) + " Gflops";
      total_flops += stat.total_flops;
    }
    result += \n\;
  }
  result += "--------------------\n";
  result += "Ideal total time: " +
            std::to_string(absl::ToDoubleMilliseconds(GetTotalTime())) + "\n";
  result += "--------------------\n";
  const double total_giga_flops = total_flops / 1000.0 / 1000.0 / 1000.0;
  result +=
      "Total Gflops for model: ~" + std::to_string(total_giga_flops) + "\n";
  const double total_times_per_sec =
      1000.0 / absl::ToDoubleMilliseconds(GetTotalTime());
  const double total_giga_flops_per_sec =
      total_times_per_sec * total_giga_flops;
  result += "Average Gflops per sec: ~" +
            std::to_string(total_giga_flops_per_sec) + "\n";
  const double total_giga_bytes_per_sec =
      total_times_per_sec * total_bytes / 1024.0 / 1024.0 / 1024.0;
  result += "Average Gb/s per sec: ~" +
            std::to_string(total_giga_bytes_per_sec) + "\n";
  result += "--------------------\n";
  return result;
}

}  // namespace ml_drift
