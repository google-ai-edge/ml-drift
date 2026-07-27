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

#include "ml_drift/webgpu/webgpu_api_util.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/debugging/leak_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/executor.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/instance.h"
#include "ml_drift/webgpu/webgpu_headers.h"  // IWYU pragma: keep

#ifdef __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/bind.h>

// clang-format off
EM_ASYNC_JS(void, ReadBufferDataJs,
            (WGPUBuffer buffer_handle, void* data_ptr), {
              const gpuReadBuffer = WebGPU.getJsObject(buffer_handle);
              await gpuReadBuffer.mapAsync(GPUMapMode.READ);
              const arrayBuffer = gpuReadBuffer.getMappedRange();
              const u8view = new Uint8Array(arrayBuffer);
              HEAPU8.set(u8view, data_ptr >>> 0);
              gpuReadBuffer.unmap();
            });

// clang-format on
#endif  // __EMSCRIPTEN__

namespace ml_drift {
namespace webgpu {
namespace {

absl::Status ReadDataFromMappableBuffer(const wgpu::Device& device,
                                        const wgpu::Queue& queue,
                                        const wgpu::Buffer& mappable_buffer,
                                        uint64_t data_size, void* data_ptr) {
#ifdef __EMSCRIPTEN__
  ReadBufferDataJs(mappable_buffer.Get(), data_ptr);
#else   // __EMSCRIPTEN__
  // TODO: b/369445924 - This chunk should use Instance::Wait() instead of a
  // while loop, but this currently causes a regression.
  wgpu::MapAsyncStatus status;
  std::string message;
  wgpu::Future future = mappable_buffer.MapAsync(
      wgpu::MapMode::Read, 0, AlignByN(data_size, 4),
      wgpu::CallbackMode::WaitAnyOnly,
      [&status, &message](wgpu::MapAsyncStatus s, wgpu::StringView msg) {
        status = s;
        message = std::string(msg);
      });

  wgpu::FutureWaitInfo wait_info = {};
  wait_info.future = future;
  wgpu::WaitStatus wait_status = wgpu::WaitStatus::TimedOut;
  absl::Time start = absl::Now();
  while (wait_status == wgpu::WaitStatus::TimedOut) {
    Instance::ProcessEvents();
    Instance::MaybeRunFlushCallback();
    wait_status = Instance::Get().WaitAny(1u, &wait_info, 0);
    if ((absl::Now() - start) > absl::Seconds(20)) {
      return absl::AbortedError(
          "The timeout was reached while reading back data.");
    }
  }
  ABSL_RETURN_IF_ERROR(VerifyWaitStatus(wait_status));
  if (status != wgpu::MapAsyncStatus::Success) {
    return absl::InternalError(message);
  }

  const void* buf_cpu_ptr =
      mappable_buffer.GetConstMappedRange(0, AlignByN(data_size, 4));
  bool mapping_success = buf_cpu_ptr;
  if (mapping_success) {
    std::memcpy(data_ptr, buf_cpu_ptr, data_size);
  }
  mappable_buffer.Unmap();
  if (!mapping_success) {
    return absl::InternalError("Buffer mapping failed.");
  }
#endif  // __EMSCRIPTEN__
  return absl::OkStatus();
}

}  // namespace

absl::Status ReadDataFromBuffer(const wgpu::Device& device,
                                const wgpu::Queue& queue,
                                const wgpu::Buffer& buffer, uint64_t data_size,
                                void* data_ptr) {
  wgpu::BufferUsage usages = buffer.GetUsage();
  if (usages & wgpu::BufferUsage::MapRead) {
    return ReadDataFromMappableBuffer(device, queue, buffer, data_size,
                                      data_ptr);
  }
  if (!(usages & wgpu::BufferUsage::CopySrc)) {
    return absl::InvalidArgumentError(
        "wgpu::Buffer must be created with BufferUsage::MapRead or "
        "BufferUsage::CopySrc");
  }
  wgpu::BufferDescriptor temp_desc = {
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = AlignByN(data_size, 4),
  };
  wgpu::Buffer temp_mappable_buf = device.CreateBuffer(&temp_desc);

  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  encoder.CopyBufferToBuffer(buffer, 0, temp_mappable_buf, 0,
                             AlignByN(data_size, 4));
  wgpu::CommandBuffer cb = encoder.Finish();
  queue.Submit(1, &cb);

  return ReadDataFromMappableBuffer(device, queue, temp_mappable_buf, data_size,
                                    data_ptr);
}

absl::Status ReadDataFromTexture(const wgpu::Device& device,
                                 const wgpu::Queue& queue,
                                 const wgpu::Texture& texture,
                                 int bytes_per_pixel, int width, int height,
                                 int depth, void* data_ptr) {
  const int bytes_per_row = bytes_per_pixel * width;
  // from spec: "texelCopyBufferInfo.bytesPerRow must be a multiple of 256."
  const uint64_t bytes_per_row_aligned = AlignByN(bytes_per_row, 256);
  const uint64_t data_size_aligned = height * depth * bytes_per_row_aligned;
  wgpu::BufferDescriptor temp_desc = {
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = data_size_aligned,
  };
  wgpu::Buffer temp_buf = device.CreateBuffer(&temp_desc);

  wgpu::Extent3D copy_size;
  copy_size.width = width;
  copy_size.height = height;
  copy_size.depthOrArrayLayers = depth;
  wgpu::TexelCopyTextureInfo src = {};
  src.texture = texture;
  src.aspect = wgpu::TextureAspect::All;
  wgpu::TexelCopyBufferInfo dst = {};
  dst.layout.offset = 0;
  dst.layout.bytesPerRow = bytes_per_row_aligned;
  dst.layout.rowsPerImage = height;
  dst.buffer = temp_buf;
  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  encoder.CopyTextureToBuffer(&src, &dst, &copy_size);
  wgpu::CommandBuffer cb = encoder.Finish();
  queue.Submit(1, &cb);

  std::vector<uint8_t> data_aligned(data_size_aligned, 0);
  ABSL_RETURN_IF_ERROR(ReadDataFromMappableBuffer(
      device, queue, temp_buf, data_size_aligned, data_aligned.data()));

  for (int r = 0; r < height * depth; ++r) {
    memcpy(reinterpret_cast<uint8_t*>(data_ptr) + r * bytes_per_row,
           data_aligned.data() + r * bytes_per_row_aligned, bytes_per_row);
  }
  return absl::OkStatus();
}

void WriteDataToBuffer(const wgpu::Queue& queue, const wgpu::Buffer& buffer,
                       const uint64_t data_size, const void* data_ptr) {
  if (data_size % 4 != 0) {
    ABSL_LOG(WARNING) << "Data size is not a multiple of 4 in "
                         "WriteDataToBuffer. WebGPU requires 4-byte alignment.";
  }
  queue.WriteBuffer(buffer, /*bufferOffset=*/0, data_ptr,
                    AlignByN(data_size, 4));
}

void WriteDataToTexture(const wgpu::Queue& queue, const wgpu::Texture& texture,
                        int bytes_per_pixel, int width, int height, int depth,
                        const void* data_ptr) {
  wgpu::TexelCopyTextureInfo destination = {};
  destination.texture = texture;
  destination.aspect = wgpu::TextureAspect::All;

  wgpu::TexelCopyBufferLayout data_layout;
  data_layout.offset = 0;
  data_layout.bytesPerRow = bytes_per_pixel * width;
  data_layout.rowsPerImage = height;

  wgpu::Extent3D write_size;
  write_size.width = width;
  write_size.height = height;
  write_size.depthOrArrayLayers = depth;

  size_t data_size = width * height * depth * bytes_per_pixel;
  queue.WriteTexture(&destination, data_ptr, data_size, &data_layout,
                     &write_size);
}

absl::Status WaitUntilCompleted(const wgpu::Queue& queue,
                                const wgpu::Device& device,
                                absl::Duration timeout) {
  wgpu::QueueWorkDoneStatus status;
  ABSL_RETURN_IF_ERROR(Instance::Wait(
      queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                                [&status](wgpu::QueueWorkDoneStatus s,
                                          wgpu::StringView) { status = s; }),
      timeout));
  ABSL_RETURN_IF_ERROR(VerifyQueueWorkDoneStatus(status));
  return absl::OkStatus();
}

wgpu::ShaderModule CreateComputeShaderModule(const wgpu::Device& device,
                                             const std::string& src) {
  wgpu::ShaderSourceWGSL wgsl;
  wgsl.code = src.c_str();
  wgpu::ShaderModuleDescriptor descriptor;
  descriptor.nextInChain = &wgsl;
  return device.CreateShaderModule(&descriptor);
}

ComputePipelineHolder::ComputePipelineHolder(
    const wgpu::Device& device, const std::string& code,
    const std::vector<wgpu::ConstantEntry>& pipeline_constants,
    const std::string& entry_point_name, const wgpu::PipelineLayout& layout,
    Executor* executor, bool async_create_call)
    : async_create_call_(async_create_call) {
  thread_data_->code = code;
  thread_data_->start = absl::Now();
  auto create_pipeline = [thread_data = thread_data_, device, code,
                          pipeline_constants, entry_point_name, layout
  // Automated tools complain about unused lambda captures, so we define around
  // this last captured argument for non-web platforms. The auto-formatting
  // tools then result in the odd spacing below.
#ifdef __EMSCRIPTEN__
                          ,
                          async_create_call] {
#else
  ] {
    {
      absl::MutexLock lock(thread_data->mutex);
      if (thread_data->state != ThreadData::State::kUnstarted) {
        return;
      }
      thread_data->state = ThreadData::State::kRunning;
    }
#endif  // __EMSCRIPTEN__
    auto add_debug = [&](const std::string& step) {
      absl::StrAppend(&thread_data->debug_str, step, ": ",
                      absl::Now() - thread_data->start, ", ");
    };
    add_debug("Start");
    wgpu::ComputePipelineDescriptor descriptor = {
        .layout = layout,
        .compute =
            {
                .module = CreateComputeShaderModule(device, code),
                .entryPoint = entry_point_name.c_str(),
                .constantCount = pipeline_constants.size(),
                .constants = pipeline_constants.data(),
            },
    };
    add_debug("Done ShaderModule");

#ifdef __EMSCRIPTEN__
    if (async_create_call) {
      thread_data->pipeline_future = device.CreateComputePipelineAsync(
          &descriptor, wgpu::CallbackMode::WaitAnyOnly,
          [thread_data](wgpu::CreatePipelineAsyncStatus status,
                        wgpu::ComputePipeline pipeline,
                        wgpu::StringView message) {
            if (status != wgpu::CreatePipelineAsyncStatus::Success) {
              thread_data->pipeline = absl::InternalError(std::string(message));
              return;
            }
            thread_data->pipeline = pipeline;
          });
      if (!thread_data->pipeline_future.id) {
        thread_data->pipeline = absl::InternalError(
            "Failed to asynchronously create compute pipeline.");
      }
    } else {
      thread_data->pipeline = device.CreateComputePipeline(&descriptor);
    }
#else
    {  // Leaks on Nvidia sometimes.
      absl::LeakCheckDisabler disabler;
      thread_data->pipeline = device.CreateComputePipeline(&descriptor);
    }
    add_debug("Done Pipeline");

    absl::MutexLock lock(thread_data->mutex);
    thread_data->state = ThreadData::State::kDone;
#endif  // __EMSCRIPTEN__
  };
  thread_data_->fn = create_pipeline;
  if (executor) {
    executor->Schedule(std::move(create_pipeline));
  } else {
    create_pipeline();
  }
}

ComputePipelineHolder::~ComputePipelineHolder() {
  // Wait for the callback here to ensure it's not called with the destructed
  // object.
  Get().IgnoreError();
}

absl::StatusOr<wgpu::ComputePipeline> ComputePipelineHolder::Get() const {
#ifndef __EMSCRIPTEN__
  // Check if the function has been scheduled yet, and if not run on the main
  // thread. This helps the case where many shader compiles may be scheduled
  // before this one, but the result of this pipeline is needed now.
  bool run_on_main_thread = false;
  {
    absl::MutexLock lock(thread_data_->mutex);
    run_on_main_thread = thread_data_->state == ThreadData::State::kUnstarted;
  }
  if (run_on_main_thread) {
    thread_data_->fn();
  }
  thread_data_->fn = {};
  {
    absl::MutexLock lock(thread_data_->mutex);
    auto done = [this] {
      thread_data_->mutex.AssertHeld();
      return thread_data_->state == ThreadData::State::kDone;
    };
    if (!thread_data_->mutex.AwaitWithTimeout(absl::Condition(&done),
                                              absl::Seconds(30))) {
      return absl::AbortedError(absl::StrCat(
          "The 30 second timeout was reached while waiting for pipeline. ",
          "Shader size: ", thread_data_->code.size(),
          ", total time: ", (absl::Now() - thread_data_->start), " timing: ",
          thread_data_->debug_str, ", source: ", thread_data_->code));
    }
  }
#else
  if (async_create_call_ && thread_data_->pipeline_future.id) {
    ABSL_RETURN_IF_ERROR(
        Instance::Wait(thread_data_->pipeline_future, absl::Seconds(30)));
    thread_data_->pipeline_future.id = 0;
  }
#endif  // !__EMSCRIPTEN__
  return thread_data_->pipeline;
}

absl::StatusOr<std::unique_ptr<ComputePipelineHolder>> CreateComputePipeline(
    const wgpu::Device& device, const std::string& code,
    const std::vector<wgpu::ConstantEntry>& pipeline_constants,
    const std::string& entry_point_name, const wgpu::PipelineLayout& layout,
    Executor* executor, bool async_create_call) {
  return std::make_unique<ComputePipelineHolder>(
      device, code, pipeline_constants, entry_point_name, layout, executor,
      async_create_call);
}

wgpu::TextureFormat DataTypeToTextureFormat(DataType data_type,
                                            int channels_count) {
  if (channels_count == 1) {
    switch (data_type) {
      case DataType::FLOAT32:
        return wgpu::TextureFormat::R32Float;
      case DataType::FLOAT16:
        return wgpu::TextureFormat::R16Float;
      case DataType::UINT32:
        return wgpu::TextureFormat::R32Uint;
      case DataType::BFLOAT16:
      case DataType::UINT16:
        return wgpu::TextureFormat::R16Uint;
      case DataType::UINT8:
        return wgpu::TextureFormat::R8Uint;
      case DataType::INT32:
        return wgpu::TextureFormat::R32Sint;
      case DataType::INT16:
        return wgpu::TextureFormat::R16Sint;
      case DataType::INT8:
        return wgpu::TextureFormat::R8Sint;
      case DataType::BOOL:
        return wgpu::TextureFormat::R8Uint;
      default:
        return wgpu::TextureFormat::Undefined;
    }
  } else if (channels_count == 2) {
    switch (data_type) {
      case DataType::FLOAT32:
        return wgpu::TextureFormat::RG32Float;
      case DataType::FLOAT16:
        return wgpu::TextureFormat::RG16Float;
      case DataType::UINT32:
        return wgpu::TextureFormat::RG32Uint;
      case DataType::BFLOAT16:
      case DataType::UINT16:
        return wgpu::TextureFormat::RG16Uint;
      case DataType::UINT8:
        return wgpu::TextureFormat::RG8Uint;
      case DataType::INT32:
        return wgpu::TextureFormat::RG32Sint;
      case DataType::INT16:
        return wgpu::TextureFormat::RG16Sint;
      case DataType::INT8:
        return wgpu::TextureFormat::RG8Sint;
      case DataType::BOOL:
        return wgpu::TextureFormat::RG8Uint;
      default:
        return wgpu::TextureFormat::Undefined;
    }
  } else if (channels_count == 3) {
    return wgpu::TextureFormat::Undefined;
  } else if (channels_count == 4) {
    switch (data_type) {
      case DataType::FLOAT32:
        return wgpu::TextureFormat::RGBA32Float;
      case DataType::FLOAT16:
        return wgpu::TextureFormat::RGBA16Float;
      case DataType::UINT32:
        return wgpu::TextureFormat::RGBA32Uint;
      case DataType::BFLOAT16:
      case DataType::UINT16:
        return wgpu::TextureFormat::RGBA16Uint;
      case DataType::UINT8:
        return wgpu::TextureFormat::RGBA8Uint;
      case DataType::INT32:
        return wgpu::TextureFormat::RGBA32Sint;
      case DataType::INT16:
        return wgpu::TextureFormat::RGBA16Sint;
      case DataType::INT8:
        return wgpu::TextureFormat::RGBA8Sint;
      case DataType::BOOL:
        return wgpu::TextureFormat::RGBA8Uint;
      default:
        return wgpu::TextureFormat::Undefined;
    }
  }
  return wgpu::TextureFormat::Undefined;
}

absl::Status VerifyQueueWorkDoneStatus(wgpu::QueueWorkDoneStatus status) {
  switch (status) {
    case wgpu::QueueWorkDoneStatus::Success:
      return absl::OkStatus();
    case wgpu::QueueWorkDoneStatus::CallbackCancelled:
      return absl::InternalError(
          "wgpu::QueueWorkDoneStatus::CallbackCancelled");
    default:
      return absl::InternalError(
          absl::StrCat("Unknown wgpu::QueueWorkDoneStatus", (int)status));
  }
}

}  // namespace webgpu
}  // namespace ml_drift
