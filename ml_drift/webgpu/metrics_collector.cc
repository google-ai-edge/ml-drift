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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "ml_drift/common/status.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/metrics_collector_internal.h"
#include "ml_drift/webgpu/webgpu_api_util.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

namespace {

// Expose the information which is used in JavaScript later to display the
// WebGPU metrics in real-time.
EM_JS(void, ExposeMetricsInfo,
      (WGPUBuffer wgpu_buffer_ptr, bool total_inference_time_enabled,
       int detailed_query_count, const char* in_names),
      {
        const gpuReadBuffer = WebGPU.getJsObject(wgpu_buffer_ptr);
        Module.MP_WEBGPU_METRICS = Module.MP_WEBGPU_METRICS || {};
        Module.MP_WEBGPU_METRICS.queryBuffer = gpuReadBuffer;
        Module.MP_WEBGPU_METRICS.totalInferenceTimeEnabled =
            total_inference_time_enabled;
        Module.MP_WEBGPU_METRICS.detailedQueryCount = detailed_query_count;
        Module.MP_WEBGPU_METRICS.operation_names =
            UTF8ToString(in_names).split("\n");
      });

}  // namespace

#endif

namespace ml_drift {
namespace webgpu {

namespace {

static constexpr int32_t kUninitializedId = -1;
static constexpr uint32_t kZeroOffset = 0;
static constexpr uint32_t kQuerySize = sizeof(uint64_t);

inline int64_t TickToNanosecond(const int64_t tick) {
  // Currently the timestamp value is a number of GPU ticks, not nanoseconds.
  // The value of the tick may be requested via VulkanInfo.timestampPeriod.
  // For NVIDIA Quadro K1200 and Vulkan Instance Version 1.2.154
  // timestampPeriod equals to 1, which means every tick equals to 1
  // nanosecond. So, we skip requesting here this timestampPeriod value,
  // because it is likely to be equal to 1. This issue will get fixed when
  // Dawn gets compilable with Tint compiler in google3 and we will be able to
  // retrieve the tick value in runtime.
  return tick;
}

// The main purpose of the QueryProcessor interface is to make MetricsCollector
// more readable by moving the query processing logic into different classes,
// which implement this interface.
class QueryProcessor {
 public:
  virtual ~QueryProcessor() = default;
  virtual void Init(int ops_amount) = 0;
  virtual void InitTimestampWrites(const wgpu::QuerySet& query_set,
                                   uint32_t amount_of_compute_passes) {}
  virtual void ReadQueries(const int64_t* data, int offset) = 0;
  virtual void ResetResults() = 0;
  virtual wgpu::ComputePassDescriptor GetComputePassDescriptor(
      std::optional<uint32_t> operation_position) = 0;
};

// This processor measures total inference time. It initializes two
// timestamps: before the very first and after the very last compute pass.
// Difference between those values is considered to be the total inference
// time and can be retrieved with GetTotalInferenceTime() method of
// MetricsCollector object.
class TotalInferenceTimeCollector : public QueryProcessor {
 public:
  TotalInferenceTimeCollector(int* query_count,
                              absl::Duration* total_inference_duration)
      : query_count_(query_count),
        total_inference_duration_(total_inference_duration) {}

  wgpu::ComputePassDescriptor GetComputePassDescriptor(
      std::optional<uint32_t>) final {
    return {
        .timestampWrites = &timestamp_writes_,
    };
  }

  void InitTimestampWrites(const wgpu::QuerySet& query_set,
                           uint32_t amount_of_compute_passes) final {
    timestamp_writes_ = {.querySet = query_set,
                         .beginningOfPassWriteIndex = 0,
                         .endOfPassWriteIndex = 1};
  }

  void Init(int ops_amount) final { *query_count_ = kAmountOfQueries; }

  void ReadQueries(const int64_t* data, int offset) final {
    *total_inference_duration_ = absl::Nanoseconds(TickToNanosecond(
        data[offset + kInferenceEndId] - data[offset + kInferenceStartId]));
  }
  void ResetResults() final {
    *total_inference_duration_ = absl::ZeroDuration();
  }

 private:
  static constexpr uint32_t kAmountOfQueries = 2;
  static constexpr uint32_t kInferenceStartId = 0;
  static constexpr uint32_t kInferenceEndId = 1;

  int* query_count_;
  absl::Duration* total_inference_duration_;
  wgpu::PassTimestampWrites timestamp_writes_;
};

// This processor measures per operation computation time. For every operation
// it initializes the pair of queries: first query right before and the second
// right after the compute shader execution. Difference between that pair of
// values is considered to be the operation computation time. In total the
// amount of queries equals to doubled amount of operations. The resulting
// statistics can be retrieved with GetPerOperationProfiling() method of
// MetricsCollector object.
class PerOperationTimeProfiler : public QueryProcessor {
 public:
  PerOperationTimeProfiler(int* query_count,
                           std::vector<absl::Duration>* per_op_durations)
      : query_count_(query_count), per_op_durations_(per_op_durations) {}

  void Init(int ops_amount) final {
    *query_count_ = ops_amount * kPairSize;
    ResetResults();
  }

  wgpu::ComputePassDescriptor GetComputePassDescriptor(
      std::optional<uint32_t> operation_position) final {
    return {
        .timestampWrites = &(timestamp_writes_[operation_position.value()]),
    };
  }

  void InitTimestampWrites(const wgpu::QuerySet& query_set,
                           uint32_t amount_of_compute_passes) final {
    timestamp_writes_.reserve(amount_of_compute_passes);
    for (uint32_t pos = 0; pos < amount_of_compute_passes; pos++) {
      const uint32_t pos_begin = pos * kPairSize + kFirstInPair;
      const uint32_t pos_end = pos * kPairSize + kSecondInPair;
      timestamp_writes_.push_back({.querySet = query_set,
                                   .beginningOfPassWriteIndex = pos_begin,
                                   .endOfPassWriteIndex = pos_end});
    }
  }

  void ReadQueries(const int64_t* data, int offset) final {
    for (int i = 0; i < *query_count_ / kPairSize; i++) {
      per_op_durations_->at(i) = absl::Nanoseconds(
          TickToNanosecond(data[offset + i * kPairSize + kSecondInPair] -
                           data[offset + i * kPairSize + kFirstInPair]));
    }
  }

  void ResetResults() final {
    if (per_op_durations_->empty()) {
      per_op_durations_->resize(*query_count_ / kPairSize);
    }
    for (absl::Duration& duration : *per_op_durations_) {
      duration = absl::ZeroDuration();
    }
  }

 private:
  static constexpr uint32_t kPairSize = 2;
  static constexpr uint32_t kFirstInPair = 0;
  static constexpr uint32_t kSecondInPair = 1;

  int* query_count_;
  std::vector<absl::Duration>* per_op_durations_;
  std::vector<wgpu::PassTimestampWrites> timestamp_writes_;
};

class MetricsCollectorImpl : public InternalMetricsCollector {
 public:
  explicit MetricsCollectorImpl(const RequestedMetrics& requested_metrics) {
    // It is important to reserve the memory here for the max available amount
    // of query processors. Otherwise, query_counts_ memory can re reallocated
    // after the push_back(), which makes previously saved pointers invalid.
    // Currently we reserve memory for two elements because only two query
    // processors are available.
    query_counts_.reserve(2);
    if (requested_metrics.total_inference_time) {
      query_counts_.push_back(kUninitializedId);
      query_processors_.push_back(std::make_unique<TotalInferenceTimeCollector>(
          &query_counts_.back(), &total_inference_duration_));
      total_inference_time_collector_id_ = query_processors_.size() - 1;
    }
    if (requested_metrics.per_operation_profiling) {
      query_counts_.push_back(kUninitializedId);
      query_processors_.push_back(std::make_unique<PerOperationTimeProfiler>(
          &query_counts_.back(), &per_op_durations_));
      per_operation_time_profiler_id_ = query_processors_.size() - 1;
    }
  }

  std::string GetExtension() final { return "timestamp_query"; }

  absl::Status Init(const std::vector<std::string>& op_names) final {
    if (!op_names_.empty()) {
      return absl::OkStatus();
    }
    for (const auto& query_processor : query_processors_) {
      query_processor->Init(op_names.size());
      query_processor->ResetResults();
    }
    for (const auto& count : query_counts_) {
      total_query_count_ += count;
    }
    if (!environment_) {
      return absl::InternalError("Environment is not set.");
    }
    wgpu::QuerySetDescriptor query_set_descriptor = {
        .type = wgpu::QueryType::Timestamp,
        .count = static_cast<uint32_t>(total_query_count_),
    };
    query_set_ = environment_->device().CreateQuerySet(&query_set_descriptor);
    query_buffer_ = CreateBuffer(
        environment_->device(),
        wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::QueryResolve,
        total_query_count_ * kQuerySize);
    dst_buffer_ =
        CreateBuffer(environment_->device(),
                     wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
                     total_query_count_ * kQuerySize);
    for (const auto& query_processor : query_processors_) {
      query_processor->InitTimestampWrites(query_set_, op_names.size());
    }
    op_names_ = op_names;
    return absl::OkStatus();
  }

  wgpu::ComputePassDescriptor GetComputePassDescriptor(
      std::optional<uint32_t> operation_position) override {
    return IsTotalTimeMetricsEnabled()
               ? query_processors_[total_inference_time_collector_id_]
                     ->GetComputePassDescriptor(operation_position.value())
               : query_processors_[per_operation_time_profiler_id_]
                     ->GetComputePassDescriptor(operation_position.value());
  }

  bool IsSingleCommandBuffer() final { return IsTotalTimeMetricsEnabled(); }

  // This method returning false means per operation profiling is enabled.
  bool IsTotalTimeMetricsEnabled() {
    return total_inference_time_collector_id_ != kUninitializedId;
  }

  void ResolveQueries(const wgpu::CommandEncoder* encoder) final {
    encoder->ResolveQuerySet(query_set_, /*firstQuery=*/0, total_query_count_,
                             query_buffer_.GetMemoryHandle(),
                             /*destinationOffset=*/0);
    encoder->CopyBufferToBuffer(query_buffer_.GetMemoryHandle(), kZeroOffset,
                                dst_buffer_.GetMemoryHandle(), kZeroOffset,
                                query_buffer_.GetMemorySizeInBytes());
  }

  void SetEnvironment(const Environment* environment) override {
    environment_ = environment;
  }

  absl::Status WaitQueriesReadBack() final {
#ifdef __EMSCRIPTEN__
    int32_t detailed_query_count =
        per_operation_time_profiler_id_ == kUninitializedId
            ? -1
            : query_counts_[per_operation_time_profiler_id_];
    std::string op_names;
    for (const auto& op_name : op_names_) {
      absl::StrAppend(&op_names, absl::StrCat(op_name, "\n"));
    }
    ExposeMetricsInfo(dst_buffer_.GetMemoryHandle().Get(),
                      IsTotalTimeMetricsEnabled(), detailed_query_count,
                      op_names.c_str());
#else
    std::vector<int64_t> querry_buffer_data(
        query_buffer_.GetMemorySizeInBytes() / 8);
    RETURN_IF_ERROR(ReadDataFromBuffer(
        environment_->device(), environment_->queue(),
        dst_buffer_.GetMemoryHandle(), query_buffer_.GetMemorySizeInBytes(),
        querry_buffer_data.data()));

    int offset = 0, i = 0;
    for (const auto& query_processor : query_processors_) {
      query_processor->ReadQueries(querry_buffer_data.data(), offset);
      offset += query_counts_[i++];
    }
#endif  // __EMSCRIPTEN__
    return absl::OkStatus();
  }

  absl::Duration GetTotalInferenceTime() final {
    absl::Duration to_return = total_inference_duration_;
    total_inference_duration_ = absl::ZeroDuration();
    if (total_inference_time_collector_id_ > kUninitializedId) {
      query_processors_[total_inference_time_collector_id_]->ResetResults();
    }
    return to_return;
  }

  std::vector<absl::Duration> GetPerOperationProfiling() final {
    std::vector<absl::Duration> to_return = per_op_durations_;
    if (per_operation_time_profiler_id_ > kUninitializedId) {
      query_processors_[per_operation_time_profiler_id_]->ResetResults();
    }
    return to_return;
  }

 private:
  absl::Duration total_inference_duration_ = absl::ZeroDuration();
  std::vector<absl::Duration> per_op_durations_;

  std::vector<int> query_counts_;
  int total_query_count_ = 0;
  wgpu::QuerySet query_set_;
  Buffer query_buffer_;
  Buffer dst_buffer_;
  std::vector<std::unique_ptr<QueryProcessor>> query_processors_;
  std::vector<std::string> op_names_;

  int total_inference_time_collector_id_ = kUninitializedId;
  int per_operation_time_profiler_id_ = kUninitializedId;

  const Environment* environment_;
};

}  // namespace

std::unique_ptr<MetricsCollector> NewMetricsCollector(
    const RequestedMetrics& requested_metrics) {
  return requested_metrics.per_operation_profiling ^
                 requested_metrics.total_inference_time
             ? std::make_unique<MetricsCollectorImpl>(requested_metrics)
             : nullptr;
}

}  // namespace webgpu
}  // namespace ml_drift
