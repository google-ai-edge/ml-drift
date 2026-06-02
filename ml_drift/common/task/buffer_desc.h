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

#ifndef ML_DRIFT_COMMON_TASK_BUFFER_DESC_H_
#define ML_DRIFT_COMMON_TASK_BUFFER_DESC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"

namespace ml_drift {

// TODO(linchan): b/427808277 - Change struct BufferDescriptor to class.
struct BufferDescriptor : public GPUObjectDescriptor {
  bool IsBufferDescriptor() const override { return true; }
  DataType element_type;
  int element_size;
  MemoryType memory_type = MemoryType::GLOBAL;
  std::vector<std::string> attributes;

  // optional
  int size = 0;
  std::vector<uint8_t> data;

  BufferDescriptor() = default;
  BufferDescriptor(const BufferDescriptor&) = default;
  BufferDescriptor& operator=(const BufferDescriptor&) = default;
  BufferDescriptor(BufferDescriptor&& desc) = default;
  BufferDescriptor& operator=(BufferDescriptor&& desc) = default;

  absl::Status PerformConstExpr(const GpuInfo& gpu_info,
                                absl::string_view const_expr,
                                std::string* result) const override;

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               absl::string_view selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(const GpuInfo& gpu_info) const override;
  absl::Status PerformReadSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;
  absl::Status PerformReadVec16AsVec4x4Selector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;
  absl::Status PerformReadAsU16Selector(const GpuInfo& gpu_info,
                                        const std::vector<std::string>& args,
                                        std::string* result) const;
  // Generates a selector expression to read a 16-bit integer from a buffer of
  // 32-bit integers. This function assumes that two 16-bit integers are packed
  // into each 32-bit element of the buffer.
  //
  // For example, if the buffer contains [A, B, C, D], where each is an int32,
  // this function generates code to read the 0th i16 from A, the 1st i16 from
  // A, the 2nd i16 from B, the 3rd i16 from B, and so on.
  //
  // @param gpu_info GPU-specific information, used by sub-calls.
  // @param args A vector containing a single string which is the index of the
  //   16-bit integer to read.
  // @param result A pointer to a string where the generated code expression
  //   will be stored.
  // @return absl::OkStatus() on success, or an error if the buffer type is
  //   unsuitable or the wrong number of arguments is provided.
  absl::Status PerformReadAsI16Selector(const GpuInfo& gpu_info,
                                        const std::vector<std::string>& args,
                                        std::string* result) const;
  absl::Status PerformReadAsU8Selector(const GpuInfo& gpu_info,
                                       const std::vector<std::string>& args,
                                       std::string* result) const;
  absl::Status PerformReadAsI8Selector(const GpuInfo& gpu_info,
                                       const std::vector<std::string>& args,
                                       std::string* result) const;
  absl::Status PerformWriteSelector(const GpuInfo& gpu_info,
                                    const std::vector<std::string>& args,
                                    std::string* result) const;
  absl::Status PerformGetPtrSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  uint64_t GetSizeInBytes() const override { return data.size(); }
};

// For BufferDescriptor
inline BufferDescriptor* AsBufferDescriptor(GPUObjectDescriptor* desc) {
  return (desc && desc->IsBufferDescriptor())
             ? static_cast<BufferDescriptor*>(desc)
             : nullptr;
}

inline const BufferDescriptor* AsBufferDescriptor(
    const GPUObjectDescriptor* desc) {
  return (desc && desc->IsBufferDescriptor())
             ? static_cast<const BufferDescriptor*>(desc)
             : nullptr;
}

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_BUFFER_DESC_H_
