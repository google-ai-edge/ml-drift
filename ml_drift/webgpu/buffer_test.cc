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

#include "ml_drift/webgpu/buffer.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/memory_manager.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {

using ::testing::HasSubstr;

constexpr uint64_t kGiB = uint64_t{1} << 30;

// Regression test: previously `Buffer(const wgpu::Buffer&, int size)` truncated
// 2 GiB (0x80000000) to -2147483648 and sign-extended `size_` to
// 0xFFFFFFFF80000000 (17179869182 GiB).
TEST(WebGpuBufferTest, PreservesExact64BitSizeAt2GiBBoundaryAndAbove) {
  constexpr uint64_t kTwoGiB = 2 * kGiB;  // 0x80000000
  Buffer two_gib_buffer(wgpu::Buffer{}, kTwoGiB);
  EXPECT_EQ(two_gib_buffer.GetMemorySizeInBytes(), kTwoGiB);
  EXPECT_EQ(two_gib_buffer.offset(), 0);

  BufferDescriptor desc;
  GpuResourcesWithValue resources;
  ABSL_ASSERT_OK(two_gib_buffer.GetGPUResources(&desc, &resources));
  ASSERT_EQ(resources.buffers.size(), 1);
  EXPECT_EQ(resources.buffers[0].second.size, kTwoGiB);
  EXPECT_EQ(resources.buffers[0].second.offset, 0);

  constexpr uint64_t kFiveGiB = 5 * kGiB;
  constexpr uint64_t kThreeGiBOffset = 3 * kGiB;
  Buffer sub_buffer(wgpu::Buffer{}, kFiveGiB, kThreeGiBOffset);
  EXPECT_EQ(sub_buffer.GetMemorySizeInBytes(), kFiveGiB);
  EXPECT_EQ(sub_buffer.offset(), kThreeGiBOffset);

  Buffer moved_buffer(std::move(sub_buffer));
  EXPECT_EQ(moved_buffer.GetMemorySizeInBytes(), kFiveGiB);
  EXPECT_EQ(moved_buffer.offset(), kThreeGiBOffset);
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_EQ(sub_buffer.GetMemorySizeInBytes(), 0);
  EXPECT_EQ(sub_buffer.offset(), 0);
  // NOLINTEND(bugprone-use-after-move)

  GpuResourcesWithValue moved_resources;
  ABSL_ASSERT_OK(moved_buffer.GetGPUResources(&desc, &moved_resources));
  ASSERT_EQ(moved_resources.buffers.size(), 1);
  EXPECT_EQ(moved_resources.buffers[0].second.size, kFiveGiB);
  EXPECT_EQ(moved_resources.buffers[0].second.offset, kThreeGiBOffset);
}

TEST(WebGpuBufferTest, CreateBufferStorageAndWriteDataOnNullBackend) {
  Environment env(wgpu::BackendType::Null);
  ABSL_ASSERT_OK(env.Initialize({}));

  Buffer buffer = CreateBufferStorage(env.device(), /*size_in_bytes=*/4096);
  ASSERT_NE(buffer.GetMemoryHandle().Get(), nullptr);
  EXPECT_EQ(buffer.GetMemorySizeInBytes(), 4096);
  EXPECT_EQ(buffer.offset(), 0);

  std::vector<uint32_t> payload(256, 0x12345678u);
  ABSL_EXPECT_OK(buffer.WriteData(env.queue(), absl::MakeSpan(payload)));

  std::vector<uint32_t> oversized_payload(2048, 0u);
  EXPECT_FALSE(
      buffer.WriteData(env.queue(), absl::MakeSpan(oversized_payload)).ok());
}

TEST(WebGpuBufferTest,
     MemoryManagerBufferAllocationFailureReturnsResourceExhausted) {
  Environment env(wgpu::BackendType::Null);
  ABSL_ASSERT_OK(env.Initialize({}));

  // Dawn's Null backend enforces a 512 MiB device memory cap while reporting a
  // 128 MiB maxStorageBufferBindingSize. Five simultaneously live 120 MiB
  // intermediate buffers fit per-buffer limits individually while exceeding the
  // 512 MiB device cap on the fifth allocation, triggering wgpu::ErrorType::
  // OutOfMemory without allocating host memory.
  GpuModel model;
  TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                        Layout::kBHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1024, 30 * 1024));  // 120 MiB
  ASSERT_LE(desc.GetMemorySizeInBytes(), env.GetInfo().GetMaxBufferSize());

  auto& node = model.nodes.emplace_back();
  for (ValueId id = 1; id <= 5; ++id) {
    model.tensors[id] = desc;
    node.inputs.push_back(id);
  }

  MemoryManager memory_manager;
  ExternalTensorsInfo ext_tensors;
  std::vector<std::unique_ptr<SpatialTensor>> temp_tensors;
  absl::StatusOr<MemoryManager::ModelId> model_id =
      memory_manager.AllocateMemory(env, model, ext_tensors, temp_tensors);
  EXPECT_GT(env.GetDeviceUnhandledErrorsOutOfMemoryCount(), 0);
  EXPECT_TRUE(absl::IsResourceExhausted(model_id.status()))
      << model_id.status();
  EXPECT_THAT(model_id.status().message(), HasSubstr("out of memory"));
}

TEST(WebGpuBufferTest, CreateBufferReturnsEmptyBufferWhenDataCannotBeMapped) {
  Environment env(wgpu::BackendType::Null);
  ABSL_ASSERT_OK(env.Initialize({}));

  // WebGPU only maps ranges whose size is a multiple of 4, so the initial data
  // of a 6-byte buffer cannot be written.
  const std::vector<uint8_t> data(6, 0xAB);
  Buffer buffer =
      CreateBufferStorage(env.device(), /*size_in_bytes=*/6, data.data());

  EXPECT_EQ(buffer.GetMemoryHandle().Get(), nullptr);
  EXPECT_EQ(buffer.GetMemorySizeInBytes(), 0);
}

TEST(WebGpuBufferTest, CreateFromBufferDescriptorCreatesBufferWithData) {
  Environment env(wgpu::BackendType::Null);
  ABSL_ASSERT_OK(env.Initialize({}));

  BufferDescriptor desc;
  desc.size = 8;
  desc.data.assign(8, 0xAB);
  Buffer buffer;
  ABSL_EXPECT_OK(buffer.CreateFromBufferDescriptor(env.device(), desc));

  EXPECT_NE(buffer.GetMemoryHandle().Get(), nullptr);
  EXPECT_EQ(buffer.GetMemorySizeInBytes(), 8);
}

TEST(WebGpuBufferTest, CreateFromBufferDescriptorFailsWhenDataCannotBeMapped) {
  Environment env(wgpu::BackendType::Null);
  ABSL_ASSERT_OK(env.Initialize({}));

  // As above, the data of a 6-byte buffer cannot be written.
  BufferDescriptor desc;
  desc.size = 6;
  desc.data.assign(6, 0xAB);
  Buffer buffer;
  const absl::Status status =
      buffer.CreateFromBufferDescriptor(env.device(), desc);

  EXPECT_TRUE(absl::IsResourceExhausted(status)) << status;
}

TEST(WebGpuBufferTest,
     MemoryManagerRejectsIntermediateBufferLargerThanMaxBufferSize) {
  Environment env(wgpu::BackendType::Null);
  ABSL_ASSERT_OK(env.Initialize({}));

  // Four F32 channels take 16 bytes per element, so this tensor is 16 bytes
  // larger than the largest buffer the device can bind.
  const uint64_t max_buffer_size = env.GetInfo().GetMaxBufferSize();
  const uint64_t width = max_buffer_size / 16 + 1;
  ASSERT_LE(width, uint64_t{std::numeric_limits<int>::max()});
  GpuModel model;
  TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                        Layout::kBHWC);
  desc.SetBHWCShape(BHWC(1, 1, static_cast<int>(width), 4));
  ASSERT_GT(desc.GetMemorySizeInBytes(), max_buffer_size);

  auto& node = model.nodes.emplace_back();
  model.tensors[1] = desc;
  node.inputs.push_back(1);

  MemoryManager memory_manager;
  ExternalTensorsInfo ext_tensors;
  std::vector<std::unique_ptr<SpatialTensor>> temp_tensors;
  absl::StatusOr<MemoryManager::ModelId> model_id =
      memory_manager.AllocateMemory(env, model, ext_tensors, temp_tensors);
  EXPECT_TRUE(absl::IsResourceExhausted(model_id.status()))
      << model_id.status();
  EXPECT_THAT(model_id.status().message(), HasSubstr("max buffer size"));
}

}  // namespace
}  // namespace webgpu
}  // namespace ml_drift
