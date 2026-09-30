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

#include "ml_drift/webgpu/environment.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"         // IWYU pragma: keep
#include "absl/status/status_macros.h"  // IWYU pragma: keep
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/webgpu/instance.h"
#include "ml_drift/webgpu/webgpu_headers.h"  // IWYU pragma: keep

#ifdef __EMSCRIPTEN__
#include <emscripten/em_asm.h>
#include <emscripten/em_js.h>
#include <webgpu/webgpu.h>
#endif  // __EMSCRIPTEN__

namespace ml_drift {
namespace webgpu {
namespace {

void PrintDeviceError(const wgpu::Device& device, wgpu::ErrorType error,
                      wgpu::StringView msg,
                      DeviceUnhandledErrors* device_unhandled_errors) {
  if (error == wgpu::ErrorType::NoError) {
    return;
  }
  device_unhandled_errors->total_count++;
  switch (error) {
    case wgpu::ErrorType::Validation:
      Environment::OnError(
          absl::StrCat("Validation error: ", std::string_view(msg)));
      return;
    case wgpu::ErrorType::OutOfMemory:
      Environment::OnError(
          absl::StrCat("Out of memory: ", std::string_view(msg)));
      return;
    case wgpu::ErrorType::Internal: {
      auto string_view = std::string_view(msg);
      if (absl::StrContains(string_view, "CreateComputePipelines failed")) {
        device_unhandled_errors->create_compute_pipeline_count++;
      }
      Environment::OnError(
          absl::StrCat("Internal error: ", string_view));
      return;
    }
    case wgpu::ErrorType::Unknown:
    default:
      Environment::OnError(
          absl::StrCat("Unknown error: ", std::string_view(msg)));
      return;
  }
}

void PrintDeviceLost(const wgpu::Device& device, wgpu::DeviceLostReason reason,
                     wgpu::StringView msg) {
  switch (reason) {
    case wgpu::DeviceLostReason::Destroyed:
      // This means the device was destroyed normally (that is,
      // Environment was deleted).
      break;
    case wgpu::DeviceLostReason::CallbackCancelled:
      Environment::OnError(
          absl::StrCat("Callback cancelled: ", std::string_view(msg)));
      return;
    case wgpu::DeviceLostReason::FailedCreation:
      Environment::OnError(
          absl::StrCat("Failed creation: ", std::string_view(msg)));
      return;
    default:
      Environment::OnError(
          absl::StrCat("Unknown reason: ", std::string_view(msg)));
      return;
  }
}

struct RequestAdapterCallbackResult {
  wgpu::RequestAdapterStatus status;
  wgpu::Adapter adapter;
};

void DefaultError(const char* msg) { ABSL_LOG(ERROR) << msg; }

wgpu::BackendType GetPreferredBackendType() {
#ifdef __APPLE__
  return wgpu::BackendType::Metal;
#elif _WIN32
  return wgpu::BackendType::D3D12;
#else
  return wgpu::BackendType::Vulkan;
#endif
}

// Emscripten bindings do not yet have support for this extension, so we must
// polyfill a check ourselves directly through JS. Otherwise we can query
// directly like with any other device feature.
bool DeviceHasSubgroups(const wgpu::Device& device) {
#ifdef __EMSCRIPTEN__
  const int res = EM_ASM_INT(
      {
        const device = WebGPU.getJsObject($0);
        return device.features.has('subgroups');
      },
      device.Get());
  return static_cast<bool>(res);
#else
  return device.HasFeature(wgpu::FeatureName::Subgroups);
#endif  // __EMSCRIPTEN__
}

#ifdef __EMSCRIPTEN__
// clang-format off
EM_JS(int, JsGetDeviceMinSubgroupSize, (void* deviceId), {
  const device = WebGPU.getJsObject(deviceId);
  return device.adapterInfo.subgroupMinSize || device.limits.minSubgroupSize;
});

EM_JS(int, JsGetDeviceMaxSubgroupSize, (void* deviceId), {
  const device = WebGPU.getJsObject(deviceId);
  return device.adapterInfo.subgroupMaxSize || device.limits.maxSubgroupSize;
});
// clang-format on
#endif  // __EMSCRIPTEN

// Safeguard from invalid subgroup limits.
// https://github.com/gpuweb/gpuweb/issues/3950 says 4 - 128 should be the
// guaranteed range, but this is subject to vendors.
constexpr int kMinSubgroupSize = 4;
constexpr int kMaxSubgroupSize = 128;
template <typename T>
void EnsureValidSubgroup(const T& subgroup_limits, WebGpuInfo* webgpu_info) {
  if (subgroup_limits.subgroupMinSize >= kMinSubgroupSize &&
      subgroup_limits.subgroupMaxSize <= kMaxSubgroupSize &&
      subgroup_limits.subgroupMinSize <= subgroup_limits.subgroupMaxSize) {
    webgpu_info->min_subgroup_size = subgroup_limits.subgroupMinSize;
    webgpu_info->max_subgroup_size = subgroup_limits.subgroupMaxSize;
  } else {
    ABSL_LOG_IF(WARNING, webgpu_info->supports_subgroups)
        << "Potentially invalid subgroupMinSize ("
        << subgroup_limits.subgroupMinSize << ") and/or subgroupMaxSize ("
        << subgroup_limits.subgroupMaxSize << "); setting to default values ("
        << kMinSubgroupSize << " and " << kMaxSubgroupSize
        << ", respectively.)";
    webgpu_info->min_subgroup_size = kMinSubgroupSize;
    webgpu_info->max_subgroup_size = kMaxSubgroupSize;
  }
}

void AddGpuLimits(wgpu::Device device, WebGpuInfo& webgpu_info) {
  wgpu::Limits limits;
  device.GetLimits(&limits);

  webgpu_info.max_compute_workgroup_size_x = limits.maxComputeWorkgroupSizeX;
  webgpu_info.max_compute_workgroup_size_y = limits.maxComputeWorkgroupSizeY;
  webgpu_info.max_compute_workgroup_size_z = limits.maxComputeWorkgroupSizeZ;
  webgpu_info.max_compute_invocations_per_workgroup =
      limits.maxComputeInvocationsPerWorkgroup;
  webgpu_info.max_compute_workgroup_storage_size =
      limits.maxComputeWorkgroupStorageSize;
  webgpu_info.max_compute_workgroups_per_dimension =
      limits.maxComputeWorkgroupsPerDimension;

  webgpu_info.max_storage_buffer_binding_size =
      limits.maxStorageBufferBindingSize;
  webgpu_info.max_buffer_size = limits.maxBufferSize;
  webgpu_info.max_storage_buffers_per_shader_stage =
      limits.maxStorageBuffersPerShaderStage;
  webgpu_info.max_storage_textures_per_shader_stage =
      limits.maxStorageTexturesPerShaderStage;
  webgpu_info.max_sampled_textures_per_shader_stage =
      limits.maxSampledTexturesPerShaderStage;

  webgpu_info.max_texture_dimension_1d = limits.maxTextureDimension1D;
  webgpu_info.max_texture_dimension_2d = limits.maxTextureDimension2D;
  webgpu_info.max_texture_dimension_3d = limits.maxTextureDimension3D;
  webgpu_info.max_texture_array_layers = limits.maxTextureArrayLayers;
}

#ifndef __EMSCRIPTEN__
DataType ToDataType(wgpu::SubgroupMatrixComponentType component_type) {
  switch (component_type) {
    case wgpu::SubgroupMatrixComponentType::F32:
      return DataType::kFloat32;
    case wgpu::SubgroupMatrixComponentType::F16:
      return DataType::kFloat16;
    case wgpu::SubgroupMatrixComponentType::I32:
      return DataType::kInt32;
    case wgpu::SubgroupMatrixComponentType::U32:
      return DataType::kUint32;
    case wgpu::SubgroupMatrixComponentType::I8:
      return DataType::kInt8;
    case wgpu::SubgroupMatrixComponentType::U8:
      return DataType::kUint8;
    default:
      return DataType::kUnknown;
  }
}

WaveMatMulOpDescriptor ToWaveMatMulOpDescriptor(
    const wgpu::SubgroupMatrixConfig& src) {
  WaveMatMulOpDescriptor dst;
  dst.m_size = src.M;
  dst.n_size = src.N;
  dst.k_size = src.K;
  dst.left_type = ToDataType(src.componentType);
  dst.right_type = ToDataType(src.componentType);
  dst.result_type = ToDataType(src.resultComponentType);
  return dst;
}

std::string ToString(wgpu::BackendType backendType) {
  switch (backendType) {
    case wgpu::BackendType::D3D12: return "Direct3D 12";
    case wgpu::BackendType::Metal: return "Metal";
    case wgpu::BackendType::Vulkan: return "Vulkan";
    case wgpu::BackendType::OpenGL: return "OpenGL";
    case wgpu::BackendType::OpenGLES: return "OpenGLES";
    default: return "Unknown";
  }
}

std::string ToString(wgpu::AdapterType adapterType) {
  switch (adapterType) {
    case wgpu::AdapterType::DiscreteGPU: return "Discrete GPU";
    case wgpu::AdapterType::IntegratedGPU: return "Integrated GPU";
    case wgpu::AdapterType::CPU: return "CPU / Software";
    default: return "Other";
  }
}

std::string ToString(const wgpu::AdapterInfo& adapter_info) {
  return absl::StrCat(std::string_view(adapter_info.device), ", arch=",
                      std::string_view(adapter_info.architecture), ", vendor=",
                      std::string_view(adapter_info.vendor), ", backend=",
                      ToString(adapter_info.backendType), ", adapterType=",
                      ToString(adapter_info.adapterType));
}

std::string ToString(const wgpu::Adapter& adapter) {
  wgpu::AdapterInfo adapter_info;
  return adapter.GetInfo(&adapter_info) == wgpu::Status::Success
             ? ToString(adapter_info)
             : "Unknown";
}
#endif  // !__EMSCRIPTEN__

}  // namespace

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info) {
  return !gpu_info.SupportsImages() || gpu_info.IsApple() ||
                 gpu_info.IsNvidia() || gpu_info.IsAMD()
             ? TensorStorageType::kBuffer
             : TensorStorageType::kTexture2D;
}

Environment::ErrorFn Environment::error_fn_ = &DefaultError;

Environment::Environment() : Environment(GetPreferredBackendType()) {}

Environment::Environment(wgpu::BackendType preferred_backend_type)
    : preferred_backend_type_(preferred_backend_type) {
  xnn_initialize(/*allocator=*/nullptr);
}

// static
void Environment::SetErrorFn(ErrorFn error_fn) { error_fn_ = error_fn; }

// static
// The error function may have been set from a separate library, so make sure
// that doesn't cause CFI errors.
ABSL_ATTRIBUTE_NO_SANITIZE_CFI
void Environment::OnError(const std::string& msg) { error_fn_(msg.c_str()); }

void Environment::RequestExtension(const std::string& extensions) {
  requested_extensions_.push_back(extensions);
}

absl::Status Environment::Initialize(const wgpu::Device& device,
                                     const wgpu::AdapterInfo& adapter_info) {
  device_ = device;
  queue_ = device.GetQueue();

  // On web, WebGPU returns limited device information. In particular, it can
  // report simply "apple" for vendor name and "common-3" for architecture, even
  // for MBP M-series chips. However, since we currently only run browser-based
  // MLDrift-WebGPU on Apple laptops and desktops (no mWeb on Apple devices), as
  // a temporary patch to allow for closer-to-proper behavior, we can fall back
  // to the lowest common denominator for our use case as being an M1 GPU.
  // TODO: We should detect GPU variations on web better, either
  // through new WebGPU APIs or perhaps using detected capabilities/features.
  std::string_view architecture_with_overrides(adapter_info.architecture);
  if (std::string(adapter_info.vendor) == "apple" &&
      (architecture_with_overrides == "metal-3" ||
       architecture_with_overrides == "common-3")) {
    architecture_with_overrides = "m1";
  }

  const auto gpu_description = absl::StrCat(
      std::string_view(adapter_info.vendor), " ", architecture_with_overrides,
      " ", std::string_view(adapter_info.device), " ",
      std::string_view(adapter_info.description));
  SetPlatformDescription(gpu_description);
  GetGpuInfoFromDeviceDescription(gpu_description, GpuApi::kWebGpu, &gpu_info_);
  for (const auto& requested : requested_extensions_) {
    if (requested == "shader_f16" &&
        device_.HasFeature(wgpu::FeatureName::ShaderF16)) {
      gpu_info_.webgpu_info.supports_fp16 = true;
    } else if (requested == "subgroups" && DeviceHasSubgroups(device_)) {
      ABSL_LOG(INFO) << "Subgroups Enabled!";
      gpu_info_.webgpu_info.supports_subgroups = true;
    }
  }
#ifdef __EMSCRIPTEN__
  struct {
    int subgroupMinSize;
    int subgroupMaxSize;
  } subgroup_limits;
  subgroup_limits.subgroupMinSize =
      JsGetDeviceMinSubgroupSize(reinterpret_cast<void*>(device_.Get()));
  subgroup_limits.subgroupMaxSize =
      JsGetDeviceMaxSubgroupSize(reinterpret_cast<void*>(device_.Get()));
  EnsureValidSubgroup(subgroup_limits, &gpu_info_.webgpu_info);
#else
  EnsureValidSubgroup(adapter_info, &gpu_info_.webgpu_info);
#endif  // __EMSCRIPTEN__

  AddGpuLimits(device_, gpu_info_.webgpu_info);
  return absl::OkStatus();
}

#ifndef __EMSCRIPTEN__
absl::Status Environment::Initialize(const InitParams& params) {
  wgpu::Adapter adapter;
  ABSL_RETURN_IF_ERROR(
      InitializeInstanceAndAdapter(&adapter, params.use_low_power));

  std::vector<wgpu::FeatureName> features;
  for (const auto& extension : requested_extensions_) {
    if (extension == "timestamp_query") {
      features.push_back(wgpu::FeatureName::TimestampQuery);
    } else if (extension == "shader_f16") {
      // This is unconditionally requested below.
    } else {
      return absl::InternalError("Unsupported feature requested.");
    }
  }

  // Request optional features.
  if (params.enable_host_mapped_pointer &&
      adapter.HasFeature(wgpu::FeatureName::HostMappedPointer)) {
    features.push_back(wgpu::FeatureName::HostMappedPointer);
  }
  if (adapter.HasFeature(wgpu::FeatureName::Subgroups)) {
    features.push_back(wgpu::FeatureName::Subgroups);
  }
  if (adapter.HasFeature(wgpu::FeatureName::ShaderF16)) {
    features.push_back(wgpu::FeatureName::ShaderF16);
  }
  if (adapter.HasFeature(wgpu::FeatureName::TimestampQuery)) {
    features.push_back(wgpu::FeatureName::TimestampQuery);
  }
  if (adapter.HasFeature(
          wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
    features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
  }

  // Buffer uploads may happen on other threads, so enable synchronization.
  features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);

  std::vector<const char*> enabled_toggles;
  enabled_toggles.push_back("disable_robustness");
  enabled_toggles.push_back("fxc_optimizations");
  enabled_toggles.push_back("d3d_disable_ieee_strictness");
  enabled_toggles.push_back("disable_polyfills_on_integer_div_and_mod");
  enabled_toggles.push_back("dump_shaders_on_failure");
  wgpu::DawnTogglesDescriptor toggles_desc;
  toggles_desc.enabledToggles = enabled_toggles.data();
  toggles_desc.enabledToggleCount = enabled_toggles.size();

  wgpu::DawnCacheDeviceDescriptor cache_desc;
  if (params.cache_descriptor) {
    cache_desc = *params.cache_descriptor;
    toggles_desc.nextInChain = &cache_desc;
  }

  wgpu::Limits adapter_limits;
  adapter.GetLimits(&adapter_limits);

  // If limits are set in the InitParams use those, otherwise use the maximum
  // adapter limits.
  wgpu::Limits requested_limits;
  if (params.limits) {
    requested_limits = *params.limits;
  } else {
    requested_limits = adapter_limits;
  }

  // Request the maximum possible workgroup storage size.
  requested_limits.maxComputeWorkgroupStorageSize =
      adapter_limits.maxComputeWorkgroupStorageSize;
  // Request the maximum possible number of storage buffers per shader stage.
  requested_limits.maxStorageBuffersPerShaderStage =
      adapter_limits.maxStorageBuffersPerShaderStage;

  wgpu::DeviceDescriptor wgpu_device_descriptor;
  wgpu_device_descriptor.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous, PrintDeviceLost);
  wgpu_device_descriptor.SetUncapturedErrorCallback(PrintDeviceError,
                                                    &device_unhandled_errors_);
  wgpu_device_descriptor.requiredFeatures = features.data();
  wgpu_device_descriptor.requiredFeatureCount = features.size();
  wgpu_device_descriptor.requiredLimits = &requested_limits;
  wgpu_device_descriptor.label = nullptr;
  wgpu_device_descriptor.nextInChain = &toggles_desc;
  device_ = adapter.CreateDevice(&wgpu_device_descriptor);

  if (!device_) {
    return absl::InternalError("Device could not be created");
  }

  queue_ = device_.GetQueue();

  wgpu::AdapterInfo info;
  adapter.GetInfo(&info);
  const auto gpu_description = absl::StrCat(
      std::string_view(info.vendor), " ", std::string_view(info.device), " ",
      std::string_view(info.architecture));
  SetPlatformDescription(gpu_description);
  GetGpuInfoFromDeviceDescription(gpu_description, GpuApi::kWebGpu, &gpu_info_);
  gpu_info_.webgpu_info.supports_subgroups =
      device_.HasFeature(wgpu::FeatureName::Subgroups);
  gpu_info_.webgpu_info.supports_fp16 =
      device_.HasFeature(wgpu::FeatureName::ShaderF16);
  gpu_info_.webgpu_info.supports_timestamp_query =
      device_.HasFeature(wgpu::FeatureName::TimestampQuery);
  gpu_info_.webgpu_info.supports_subgroup_matrix =
      device_.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
  gpu_info_.webgpu_info.supports_host_mapped_pointer =
      device_.HasFeature(wgpu::FeatureName::HostMappedPointer);

  EnsureValidSubgroup(info, &gpu_info_.webgpu_info);

  if (gpu_info_.webgpu_info.supports_subgroup_matrix) {
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs;
    wgpu::AdapterInfo info;
    info.nextInChain = &subgroup_matrix_configs;
    device_.GetAdapterInfo(&info);
    for (int i = 0; i < subgroup_matrix_configs.configCount; ++i) {
      gpu_info_.wave_mat_mul_ops.push_back(
          ToWaveMatMulOpDescriptor(subgroup_matrix_configs.configs[i]));
    }
  }

  gpu_info_.webgpu_info.is_integrated_gpu =
      info.adapterType == wgpu::AdapterType::IntegratedGPU;
  AddGpuLimits(device_, gpu_info_.webgpu_info);
  return absl::OkStatus();
}

absl::Status Environment::InitializeInstanceAndAdapter(wgpu::Adapter* adapter,
                                                       bool use_low_power) {
  // When there are multiple GPUs, e.g. in a laptop, prefer the discrete GPU
  // over integrated.
  wgpu::RequestAdapterOptions options;
  if (use_low_power) {
    options.powerPreference = wgpu::PowerPreference::LowPower;
  } else {
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
  }
  options.backendType = preferred_backend_type_;

  std::vector<const char*> enabled_toggles;
  wgpu::DawnTogglesDescriptor toggles_desc = {};
  if (preferred_backend_type_ == wgpu::BackendType::Vulkan) {
    // Required for ShaderF16
    enabled_toggles.push_back("vulkan_enable_f16_on_nvidia");
    // Required for SubgroupMatrix
    enabled_toggles.push_back("use_vulkan_memory_model");
  }
  toggles_desc.enabledToggles = enabled_toggles.data();
  toggles_desc.enabledToggleCount = enabled_toggles.size();
  options.nextInChain = &toggles_desc;

  RequestAdapterCallbackResult result;
  const wgpu::Instance& instance = Instance::Get();
  if (!instance) {
    return absl::InternalError("Error creating instance");
  }

  instance.RequestAdapter(
      &options, wgpu::CallbackMode::AllowSpontaneous,
      [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
         wgpu::StringView message, RequestAdapterCallbackResult* result) {
        result->status = status;
        result->adapter = adapter;
      },
      &result);
  if (result.status != wgpu::RequestAdapterStatus::Success) {
    return absl::InternalError("No adapters found");
  }
  // Adapters are sorted in order of preference, so choose the first available.
  *adapter = std::move(result.adapter);
  ABSL_LOG(INFO) << "Selected adapter: " << ToString(*adapter);
  return absl::OkStatus();
}
#endif  // __EMSCRIPTEN__

void Environment::Tick() const { Instance::Get(device_).ProcessEvents(); }

}  // namespace webgpu
}  // namespace ml_drift
