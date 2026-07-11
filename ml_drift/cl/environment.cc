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

#include "ml_drift/cl/environment.h"

#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace cl {
namespace {

bool IsGpuSupportsStorageType(const GpuInfo& gpu_info,
                              TensorStorageType storage_type) {
  switch (storage_type) {
    case TensorStorageType::TEXTURE_2D:
      return !gpu_info.IsAMD();
    case TensorStorageType::BUFFER:
      return true;
    case TensorStorageType::TEXTURE_ARRAY:
      return !gpu_info.IsAMD() && gpu_info.SupportsTextureArray();
    case TensorStorageType::IMAGE_BUFFER:
      return (gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsNvidia()) &&
             gpu_info.SupportsImageBuffer();
    case TensorStorageType::TEXTURE_3D:
      return !gpu_info.IsAMD() && gpu_info.SupportsImage3D();
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return false;
    case TensorStorageType::UNKNOWN:
      return false;
  }
  return false;
}

bool IsGpuSupportsPrecision(const GpuInfo& gpu_info,
                            CalculationsPrecision precision) {
  switch (precision) {
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      return gpu_info.SupportsFP16();
    case CalculationsPrecision::F32:
      return true;
  }
}

}  // namespace

Environment::Environment(CLDevice&& device, CLContext&& context,
                         CLCommandQueue&& queue,
                         ProfilingCommandQueue&& profiling_queue)
    : device_(std::move(device)),
      context_(std::move(context)),
      queue_(std::move(queue)),
      profiling_queue_(std::move(profiling_queue)) {
  if (device_.info_.IsPowerVR() &&
      device_.info_.powervr_info.gpu_version >= PowerVRGpu::kCXT &&
      SupportsImgPixelWaveDot(context_.context())) {
    device_.AddExtension("cl_img_pixel_wave_dot");
  }
}

Environment::Environment(Environment&& environment)
    : device_(std::move(environment.device_)),
      context_(std::move(environment.context_)),
      queue_(std::move(environment.queue_)),
      profiling_queue_(std::move(environment.profiling_queue_)),
      program_cache_(std::move(environment.program_cache_)) {}

Environment& Environment::operator=(Environment&& environment) {
  if (this != &environment) {
    device_ = std::move(environment.device_);
    context_ = std::move(environment.context_);
    queue_ = std::move(environment.queue_);
    profiling_queue_ = std::move(environment.profiling_queue_);
    program_cache_ = std::move(environment.program_cache_);
  }
  return *this;
}

absl::Status Environment::Init() {
  if (device().GetInfo().IsAdreno() &&
      device().GetInfo().SupportsTextureArray()) {
    const auto& adreno_info = device().info_.adreno_info;
    // Some Adreno < 600 have bug with one layer texture array. b/131099086
    // If we have one layer texture array and will write smt from kernel to this
    // texture, we will get zeroes instead of actual values.
    // The same kernel will work, if we use texture array with more than one
    // layer.
    if (adreno_info.IsAdreno3xx() || adreno_info.IsAdreno4xx() ||
        adreno_info.IsAdreno5xx()) {
      GetDevicePtr()->DisableOneLayerTextureArray();
    }
  }
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return absl::InternalError("Could not initialize XNNPack.");
  }
  return absl::OkStatus();
}

std::vector<CalculationsPrecision> Environment::GetSupportedPrecisions() const {
  std::vector<CalculationsPrecision> precisions;
  for (CalculationsPrecision precision :
       {CalculationsPrecision::F32, CalculationsPrecision::F32_F16,
        CalculationsPrecision::F16}) {
    if (IsSupported(precision)) {
      precisions.push_back(precision);
    }
  }
  return precisions;
}

bool Environment::IsSupported(CalculationsPrecision precision) const {
  return IsGpuSupportsPrecision(device_.GetInfo(), precision);
}

std::vector<TensorStorageType> Environment::GetSupportedStorages() const {
  std::vector<TensorStorageType> storage_types;
  for (auto storage_type :
       {TensorStorageType::TEXTURE_2D, TensorStorageType::BUFFER,
        TensorStorageType::TEXTURE_ARRAY, TensorStorageType::IMAGE_BUFFER,
        TensorStorageType::TEXTURE_3D}) {
    if (IsSupported(storage_type)) {
      storage_types.push_back(storage_type);
    }
  }
  return storage_types;
}

std::vector<TensorStorageType>
Environment::GetSupportedStoragesWithHWZeroClampSupport() const {
  std::vector<TensorStorageType> storage_types;
  for (auto storage_type :
       {TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_ARRAY,
        TensorStorageType::TEXTURE_3D}) {
    if (IsSupported(storage_type)) {
      storage_types.push_back(storage_type);
    }
  }
  return storage_types;
}

bool Environment::IsSupported(TensorStorageType storage_type) const {
  return IsGpuSupportsStorageType(device_.GetInfo(), storage_type);
}

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno()) {
    return TensorStorageType::TEXTURE_2D;
  } else if (gpu_info.IsPowerVR()) {
    return TensorStorageType::TEXTURE_2D;
  } else if (gpu_info.IsMali()) {
    return TensorStorageType::TEXTURE_2D;
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::BUFFER;
  }
  return TensorStorageType::BUFFER;
}

TensorStorageType GetStorageTypeWithMinimalMemoryConsumption(
    const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx() ||
        gpu_info.adreno_info.IsAdreno4xx()) {
      return TensorStorageType::BUFFER;
    } else {
      if (gpu_info.opencl_info.IsImage2dFromBufferSupported()) {
        return TensorStorageType::TEXTURE_2D;
      } else {
        return TensorStorageType::IMAGE_BUFFER;
      }
    }
  } else if (gpu_info.IsPowerVR()) {
    if (gpu_info.opencl_info.IsImage2dFromBufferSupported() &&
        CanUseSubBufferForImage2d(gpu_info)) {
      return TensorStorageType::TEXTURE_2D;
    } else {
      return TensorStorageType::BUFFER;
    }
  } else if (gpu_info.IsMali()) {
    if (gpu_info.opencl_info.IsImage2dFromBufferSupported() &&
        CanUseSubBufferForImage2d(gpu_info)) {
      return TensorStorageType::TEXTURE_2D;
    } else {
      return TensorStorageType::BUFFER;
    }
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::BUFFER;
  }
  return TensorStorageType::BUFFER;
}

bool CanUseSubBufferForImage2d(const GpuInfo& gpu_info) {
  if (!gpu_info.IsCL11OrHigher()) {
    return false;
  }
  if (gpu_info.IsPowerVR() &&
      gpu_info.powervr_info.driver_version.branch_main <= 23) {
    // PowerVR DXT-48-1536-0.5RT2, 24.2@6603887 - works.
    // PowerVR BXM-8-256, 1.15@6133110 - doesn't work.
    //   Segfaults, wrong results at model level.
    return false;
  }
  if (gpu_info.IsNvidia()) {
    return false;
  }
  if (gpu_info.IsMali() &&
      (gpu_info.mali_info.IsBifrost() || gpu_info.mali_info.IsMidgard())) {
    // Known driver issue on some G72 (Bifrost), G76 (Bifrost), T830 (Midgard),
    // and T880 (Midgard) devices.
    return false;
  }
  return true;
}

absl::Status CreateEnvironment(Environment* result,
                               const EnvironmentOptions& options) {
  CLDevice gpu;
  RETURN_IF_ERROR(CreateDefaultGPUDevice(&gpu));

  CLContextOptions context_options;
  context_options.performance = options.performance;
  context_options.priority = options.priority;

  CLContext context;
  RETURN_IF_ERROR(CreateCLContext(gpu, &context, context_options));

  CLCommandQueueOptions queue_options;
  queue_options.priority = options.priority;

  CLCommandQueue queue;
  RETURN_IF_ERROR(CreateCLCommandQueue(gpu, context, &queue, queue_options));
  ProfilingCommandQueue profiling_queue;
  RETURN_IF_ERROR(CreateProfilingCommandQueue(gpu, context, &profiling_queue,
                                              queue_options));

  *result = Environment(std::move(gpu), std::move(context), std::move(queue),
                        std::move(profiling_queue));
  return result->Init();
}

}  // namespace cl
}  // namespace ml_drift
