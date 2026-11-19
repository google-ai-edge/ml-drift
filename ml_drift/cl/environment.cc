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
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_program.h"
#include "ml_drift/cl/util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace cl {
namespace {

bool IsGpuSupportsStorageType(const GpuInfo& gpu_info,
                              TensorStorageType storage_type) {
  switch (storage_type) {
    case TensorStorageType::kTexture2D:
      return !gpu_info.IsAMD();
    case TensorStorageType::kBuffer:
      return true;
    case TensorStorageType::kTextureArray:
      return !gpu_info.IsAMD() && gpu_info.SupportsTextureArray();
    case TensorStorageType::kImageBuffer:
      return (gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsNvidia()) &&
             gpu_info.SupportsImageBuffer();
    case TensorStorageType::kTexture3D:
      return !gpu_info.IsAMD() && gpu_info.SupportsImage3D();
    case TensorStorageType::kSingleTexture2D:
      return false;
    case TensorStorageType::kUnknown:
      return false;
  }
  return false;
}

bool IsGpuSupportsPrecision(const GpuInfo& gpu_info,
                            CalculationsPrecision precision) {
  switch (precision) {
    case CalculationsPrecision::kF32F16:
    case CalculationsPrecision::kF16:
      return gpu_info.SupportsFP16();
    case CalculationsPrecision::kF32:
      return true;
  }
}

bool SupportsImgPixelSubgroupDot(const CLDevice& device,
                                 const CLContext& context) {
  if (device.info_.IsPowerVR()) {
    auto result = CreateCLProgram(GetImgPixelSubgroupDotSample(),
                                  /*compiler_options=*/"", context, device);
    if (result.ok()) {
      return true;
    }
  }
  return false;
}

}  // namespace

Environment::Environment(CLDevice&& device, CLContext&& context,
                         CLCommandQueue&& queue,
                         ProfilingCommandQueue&& profiling_queue)
    : device_(std::move(device)),
      context_(std::move(context)),
      queue_(std::move(queue)),
      profiling_queue_(std::move(profiling_queue)) {
  if (SupportsImgPixelSubgroupDot(device_, context_)) {
    device_.AddExtension("cl_img_pixel_subgroup_dot");
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
    // Some Adreno < 600 have bug with one layer texture array.
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
       {CalculationsPrecision::kF32, CalculationsPrecision::kF32F16,
        CalculationsPrecision::kF16}) {
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
       {TensorStorageType::kTexture2D, TensorStorageType::kBuffer,
        TensorStorageType::kTextureArray, TensorStorageType::kImageBuffer,
        TensorStorageType::kTexture3D}) {
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
       {TensorStorageType::kTexture2D, TensorStorageType::kTextureArray,
        TensorStorageType::kTexture3D}) {
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
    return TensorStorageType::kTexture2D;
  } else if (gpu_info.IsPowerVR()) {
    return TensorStorageType::kTexture2D;
  } else if (gpu_info.IsMali()) {
    return TensorStorageType::kTexture2D;
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::kImageBuffer
                                          : TensorStorageType::kBuffer;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::kImageBuffer
                                          : TensorStorageType::kBuffer;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::kBuffer;
  }
  return TensorStorageType::kBuffer;
}

TensorStorageType GetStorageTypeWithMinimalMemoryConsumption(
    const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx() ||
        gpu_info.adreno_info.IsAdreno4xx()) {
      return TensorStorageType::kBuffer;
    } else {
      if (gpu_info.opencl_info.IsImage2dFromBufferSupported()) {
        return TensorStorageType::kTexture2D;
      } else {
        return TensorStorageType::kImageBuffer;
      }
    }
  } else if (gpu_info.IsPowerVR()) {
    if (gpu_info.opencl_info.IsImage2dFromBufferSupported() &&
        CanUseSubBufferForImage2d(gpu_info)) {
      return TensorStorageType::kTexture2D;
    } else {
      return TensorStorageType::kBuffer;
    }
  } else if (gpu_info.IsMali()) {
    if (gpu_info.opencl_info.IsImage2dFromBufferSupported() &&
        CanUseSubBufferForImage2d(gpu_info)) {
      return TensorStorageType::kTexture2D;
    } else {
      return TensorStorageType::kBuffer;
    }
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::kImageBuffer
                                          : TensorStorageType::kBuffer;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::kImageBuffer
                                          : TensorStorageType::kBuffer;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::kBuffer;
  }
  return TensorStorageType::kBuffer;
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
  ABSL_RETURN_IF_ERROR(CreateDefaultGPUDevice(&gpu));

  CLContextOptions context_options;
  context_options.performance = options.performance;
  context_options.priority = options.priority;

  CLContext context;
  ABSL_RETURN_IF_ERROR(CreateCLContext(gpu, &context, context_options));

  CLCommandQueueOptions queue_options;
  queue_options.priority = options.priority;

  CLCommandQueue queue;
  ABSL_RETURN_IF_ERROR(
      CreateCLCommandQueue(gpu, context, &queue, queue_options));
  ProfilingCommandQueue profiling_queue;
  ABSL_RETURN_IF_ERROR(CreateProfilingCommandQueue(
      gpu, context, &profiling_queue, queue_options));

  *result = Environment(std::move(gpu), std::move(context), std::move(queue),
                        std::move(profiling_queue));
  return result->Init();
}

}  // namespace cl
}  // namespace ml_drift
