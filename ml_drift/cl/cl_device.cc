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

#include "ml_drift/cl/cl_device.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {
namespace {
OpenClInfo::ClKhrIntegerDotProductInfo GetClKhrIntegerDotProductInfo(
    cl_device_id id) {
  OpenClInfo::ClKhrIntegerDotProductInfo khr_8bit_info;
  auto caps = GetDeviceInfo<cl_device_integer_dot_product_capabilities_khr>(
      id, CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR);
  khr_8bit_info.suppports_4x8bit_packed =
      caps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR;
  khr_8bit_info.suppports_4x8bit =
      caps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR;
  if (khr_8bit_info.suppports_4x8bit) {
    auto accelerate_cap = GetDeviceInfo<
        cl_device_integer_dot_product_acceleration_properties_khr>(
        id, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR);
    khr_8bit_info.acceleration_info_4x8bit.signed_accelerated =
        accelerate_cap.signed_accelerated;
    khr_8bit_info.acceleration_info_4x8bit.unsigned_accelerated =
        accelerate_cap.unsigned_accelerated;
    khr_8bit_info.acceleration_info_4x8bit.mixed_signedness_accelerated =
        accelerate_cap.mixed_signedness_accelerated;
    khr_8bit_info.acceleration_info_4x8bit
        .accumulating_saturating_signed_accelerated =
        accelerate_cap.accumulating_saturating_signed_accelerated;
    khr_8bit_info.acceleration_info_4x8bit
        .accumulating_saturating_unsigned_accelerated =
        accelerate_cap.accumulating_saturating_unsigned_accelerated;
    khr_8bit_info.acceleration_info_4x8bit
        .accumulating_saturating_mixed_signedness_accelerated =
        accelerate_cap.accumulating_saturating_mixed_signedness_accelerated;
  }
  if (khr_8bit_info.suppports_4x8bit_packed) {
    // clang-format off
    auto accelerate_cap = GetDeviceInfo<
      cl_device_integer_dot_product_acceleration_properties_khr>(
      id,
      CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR);
    // clang-format on
    khr_8bit_info.acceleration_info_4x8bit_packed.signed_accelerated =
        accelerate_cap.signed_accelerated;
    khr_8bit_info.acceleration_info_4x8bit_packed.unsigned_accelerated =
        accelerate_cap.unsigned_accelerated;
    khr_8bit_info.acceleration_info_4x8bit_packed.mixed_signedness_accelerated =
        accelerate_cap.mixed_signedness_accelerated;
    khr_8bit_info.acceleration_info_4x8bit_packed
        .accumulating_saturating_signed_accelerated =
        accelerate_cap.accumulating_saturating_signed_accelerated;
    khr_8bit_info.acceleration_info_4x8bit_packed
        .accumulating_saturating_unsigned_accelerated =
        accelerate_cap.accumulating_saturating_unsigned_accelerated;
    khr_8bit_info.acceleration_info_4x8bit_packed
        .accumulating_saturating_mixed_signedness_accelerated =
        accelerate_cap.accumulating_saturating_mixed_signedness_accelerated;
  }
  return khr_8bit_info;
}
}  // namespace

static void ParsePowerVRDriverVersion(const std::string& cl_driver_version,
                                      PowerVRInfo::DriverVersion& result) {
  size_t position = cl_driver_version.find('@');
  if (position == std::string::npos) {
    return;
  }

  // string format: "*.**@*******" where * is digit
  int main = 0;
  size_t curpos = 0;
  while (curpos < position && absl::ascii_isdigit(cl_driver_version[curpos])) {
    main = main * 10 + cl_driver_version[curpos] - '0';
    ++curpos;
  }

  ++curpos;
  int minor = 0;
  while (curpos < position) {
    minor = minor * 10 + cl_driver_version[curpos] - '0';
    ++curpos;
  }

  curpos = position + 1;
  int id = 0;
  while (curpos < cl_driver_version.length()) {
    id = id * 10 + cl_driver_version[curpos] - '0';
    ++curpos;
  }
  result.branch_main = main;
  result.branch_minor = minor;
  result.id = id;
}

template <>
std::string GetDeviceInfo<std::string>(cl_device_id id, cl_device_info info) {
  size_t size;
  cl_int error = clGetDeviceInfo(id, info, 0, nullptr, &size);
  if (error != CL_SUCCESS) {
    return "";
  }

  std::string result(size - 1, 0);
  error = clGetDeviceInfo(id, info, size, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return "";
  }
  return result;
}

namespace {
template <typename T>
T GetPlatformInfo(cl_platform_id id, cl_platform_info info) {
  T result;
  cl_int error = clGetPlatformInfo(id, info, sizeof(T), &result, nullptr);
  if (error != CL_SUCCESS) {
    return -1;
  }
  return result;
}

std::string GetPlatformInfo(cl_platform_id id, cl_platform_info info) {
  size_t size;
  cl_int error = clGetPlatformInfo(id, info, 0, nullptr, &size);
  if (error != CL_SUCCESS) {
    return "";
  }

  std::string result(size - 1, 0);
  error = clGetPlatformInfo(id, info, size, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return "";
  }
  return result;
}

void GetDeviceWorkDimsSizes(cl_device_id id, int3* result) {
  int dims_count =
      GetDeviceInfo<cl_uint>(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
  if (dims_count < 3) {
    return;
  }
  std::vector<size_t> limits(dims_count);
  cl_int error =
      clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      sizeof(size_t) * dims_count, limits.data(), nullptr);
  if (error != CL_SUCCESS) {
    return;
  }
  // dims_count must be at least 3 according to spec
  result->x = limits[0];
  result->y = limits[1];
  result->z = limits[2];
}

OpenClVersion ParseCLVersion(const std::string& version) {
  const auto first_dot_pos = version.find_first_of('.');
  if (first_dot_pos == std::string::npos) {
    return OpenClVersion::kCl1_0;
  }
  const int major = version[first_dot_pos - 1] - '0';
  const int minor = version[first_dot_pos + 1] - '0';

  if (major == 1) {
    if (minor == 2) {
      return OpenClVersion::kCl1_2;
    } else if (minor == 1) {
      return OpenClVersion::kCl1_1;
    } else {
      return OpenClVersion::kCl1_0;
    }
  } else if (major == 2) {
    if (minor == 2) {
      return OpenClVersion::kCl2_2;
    } else if (minor == 1) {
      return OpenClVersion::kCl2_1;
    } else {
      return OpenClVersion::kCl2_0;
    }
  } else if (major == 3) {
    return OpenClVersion::kCl3_0;
  } else {
    return OpenClVersion::kCl1_0;
  }
}

GpuInfo GpuInfoFromDeviceID(cl_device_id id, cl_platform_id platform_id) {
  GpuInfo info;
  info.opencl_info.platform_version =
      GetPlatformInfo(platform_id, CL_PLATFORM_VERSION);
  info.opencl_info.device_name = GetDeviceInfo<std::string>(id, CL_DEVICE_NAME);
  info.opencl_info.vendor_name =
      GetDeviceInfo<std::string>(id, CL_DEVICE_VENDOR);
  info.opencl_info.opencl_c_version =
      GetDeviceInfo<std::string>(id, CL_DEVICE_OPENCL_C_VERSION);
  info.opencl_info.driver_version =
      GetDeviceInfo<std::string>(id, CL_DRIVER_VERSION);
  const std::string gpu_description = absl::StrCat(
      info.opencl_info.device_name, " ", info.opencl_info.vendor_name, " ",
      info.opencl_info.opencl_c_version);
  GetGpuInfoFromDeviceDescription(gpu_description, GpuApi::kOpenCl, &info);
  if (info.IsAdreno()) {
    info.adreno_info.cl_compiler_version =
        GetQualcommOpenClCompilerVersion(info.opencl_info.driver_version);
  } else if (info.IsPowerVR()) {
    ParsePowerVRDriverVersion(info.opencl_info.driver_version,
                              info.powervr_info.driver_version);
  }
  info.opencl_info.cl_version =
      ParseCLVersion(info.opencl_info.opencl_c_version);
  info.opencl_info.extensions =
      absl::StrSplit(GetDeviceInfo<std::string>(id, CL_DEVICE_EXTENSIONS), ' ');
  const std::vector<std::string> unsupported_extensions =
      GetUnsupportedExtensions();
  for (const auto& unsupported_extension : unsupported_extensions) {
    for (auto it = info.opencl_info.extensions.begin();
         it != info.opencl_info.extensions.end();) {
      if (*it == unsupported_extension) {
        it = info.opencl_info.extensions.erase(it);
      } else {
        ++it;
      }
    }
  }
  const std::vector<std::string> private_extensions =
      GetPrivateExtensions(info);
  info.opencl_info.extensions.insert(info.opencl_info.extensions.end(),
                                     private_extensions.begin(),
                                     private_extensions.end());
  info.opencl_info.supports_fp16 = false;
  info.opencl_info.supports_image3d_writes = false;
  for (const auto& ext : info.opencl_info.extensions) {
    if (ext == "cl_khr_fp16") {
      info.opencl_info.supports_fp16 = true;
    }
    if (ext == "cl_khr_3d_image_writes") {
      info.opencl_info.supports_image3d_writes = true;
    }
  }

  info.opencl_info.supports_images =
      GetDeviceInfo<cl_bool>(id, CL_DEVICE_IMAGE_SUPPORT);

  cl_device_fp_config f32_config =
      GetDeviceInfo<cl_device_fp_config>(id, CL_DEVICE_SINGLE_FP_CONFIG);
  info.opencl_info.supports_fp32_rtn = f32_config & CL_FP_ROUND_TO_NEAREST;

  if (info.opencl_info.supports_fp16) {
    cl_device_fp_config f16_config;
    auto status = GetDeviceInfo<cl_device_fp_config>(
        id, CL_DEVICE_HALF_FP_CONFIG, &f16_config);
    if (status.ok()) {
      info.opencl_info.is_fp16_ftz_hardware_forced =
          !(f16_config & CL_FP_DENORM);
    }
    // AMD supports cl_khr_fp16 but CL_DEVICE_HALF_FP_CONFIG is empty.
    if (status.ok() && !info.IsAMD()) {
      info.opencl_info.supports_fp16_rtn = f16_config & CL_FP_ROUND_TO_NEAREST;
    } else {  // happens on PowerVR
      f16_config = f32_config;
      info.opencl_info.supports_fp16_rtn = info.opencl_info.supports_fp32_rtn;
    }
  } else {
    info.opencl_info.supports_fp16_rtn = false;
  }

  if (info.IsPowerVR() && !info.opencl_info.supports_fp16) {
    // PowerVR doesn't have full support of fp16 and so doesn't list this
    // extension. But it can support fp16 in MADs and as buffers/textures types,
    // so we will use it.
    info.opencl_info.supports_fp16 = true;
    info.opencl_info.supports_fp16_rtn = info.opencl_info.supports_fp32_rtn;
  }

  if (!info.opencl_info.supports_image3d_writes &&
      ((info.IsAdreno() && info.adreno_info.IsAdreno4xx()) ||
       info.IsNvidia())) {
    // in local tests Adreno 430 can write in image 3d, at least on small sizes,
    // but it doesn't have cl_khr_3d_image_writes in list of available
    // extensions
    // The same for NVidia
    info.opencl_info.supports_image3d_writes = true;
  }

  if (info.SupportsExtension("cl_khr_integer_dot_product")) {
    info.opencl_info.cl_khr_integer_dot_product_info =
        GetClKhrIntegerDotProductInfo(id);
  }
  info.opencl_info.compute_units_count =
      GetDeviceInfo<cl_uint>(id, CL_DEVICE_MAX_COMPUTE_UNITS);
  info.opencl_info.image2d_max_width =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_WIDTH);
  info.opencl_info.image2d_max_height =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
  info.opencl_info.buffer_max_size =
      GetDeviceInfo<cl_ulong>(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
  info.opencl_info.max_allocation_size =
      GetDeviceInfo<cl_ulong>(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
  if (info.opencl_info.cl_version >= OpenClVersion::kCl1_2) {
    info.opencl_info.image_buffer_max_size =
        GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
    info.opencl_info.image_array_max_layers =
        GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE);
  }
  info.opencl_info.image3d_max_width =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE3D_MAX_WIDTH);
  info.opencl_info.image3d_max_height =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
  info.opencl_info.image3d_max_depth =
      GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE3D_MAX_DEPTH);
  int3 max_work_group_sizes;
  GetDeviceWorkDimsSizes(id, &max_work_group_sizes);
  info.opencl_info.max_work_group_size_x = max_work_group_sizes.x;
  info.opencl_info.max_work_group_size_y = max_work_group_sizes.y;
  info.opencl_info.max_work_group_size_z = max_work_group_sizes.z;
  info.opencl_info.max_work_group_total_size =
      GetDeviceInfo<size_t>(id, CL_DEVICE_MAX_WORK_GROUP_SIZE);
  info.opencl_info.dedicated_local_memory =
      (GetDeviceInfo<cl_device_local_mem_type>(id, CL_DEVICE_LOCAL_MEM_TYPE) ==
       CL_LOCAL);

  info.opencl_info.base_addr_align_in_bits =
      GetDeviceInfo<cl_uint>(id, CL_DEVICE_MEM_BASE_ADDR_ALIGN);
  info.opencl_info.image_pitch_alignment = 0;
  if (info.opencl_info.cl_version == OpenClVersion::kCl2_0 ||
      info.opencl_info.cl_version == OpenClVersion::kCl2_1 ||
      info.opencl_info.cl_version == OpenClVersion::kCl2_2) {
    info.opencl_info.image_pitch_alignment =
        GetDeviceInfo<cl_uint>(id, CL_DEVICE_IMAGE_PITCH_ALIGNMENT);
    info.opencl_info.image_base_address_alignment =
        GetDeviceInfo<cl_uint>(id, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT);
  } else if (info.SupportsExtension("cl_khr_image2d_from_buffer")) {
    cl_uint result = 0;
    auto status =
        GetDeviceInfo(id, CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR, &result);
    if (status.ok()) {
      info.opencl_info.image_pitch_alignment = result;
    }
    result = 0;
    status =
        GetDeviceInfo(id, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR, &result);
    if (status.ok()) {
      info.opencl_info.image_base_address_alignment = result;
    }
  }

  if (info.SupportsExtension("cl_arm_scheduling_controls")) {
    auto capabilities =
        GetDeviceInfo<cl_device_scheduling_controls_capabilities_arm>(
            id, CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM);
    info.opencl_info.supports_register_allocation_arm =
        capabilities & CL_DEVICE_SCHEDULING_REGISTER_ALLOCATION_ARM;
  }

  if (info.SupportsExtension("cl_khr_device_uuid")) {
    info.opencl_info.device_uuid_khr.resize(CL_UUID_SIZE_KHR, 0);
    info.opencl_info.driver_uuid_khr.resize(CL_UUID_SIZE_KHR, 0);

    cl_int error = clGetDeviceInfo(id, CL_DEVICE_UUID_KHR, CL_UUID_SIZE_KHR,
                                   info.opencl_info.device_uuid_khr.data(),
                                   /*param_value_size_ret=*/nullptr);
    if (error != CL_SUCCESS) {
      info.opencl_info.device_uuid_khr.clear();
    }
    error = clGetDeviceInfo(id, CL_DRIVER_UUID_KHR, CL_UUID_SIZE_KHR,
                            info.opencl_info.driver_uuid_khr.data(),
                            /*param_value_size_ret=*/nullptr);
    if (error != CL_SUCCESS) {
      info.opencl_info.driver_uuid_khr.clear();
    }
  }

  if (info.SupportsExtension("cl_intel_required_subgroup_size")) {
    size_t sub_groups_ret_size;
    constexpr cl_device_info kClDeviceSubGroupSizesIntel = 0x4108;
    cl_int status = clGetDeviceInfo(id, kClDeviceSubGroupSizesIntel, 0, nullptr,
                                    &sub_groups_ret_size);
    if (status == CL_SUCCESS) {
      auto sub_groups_count = sub_groups_ret_size / sizeof(size_t);
      std::vector<size_t> sub_group_sizes(sub_groups_count);
      status =
          clGetDeviceInfo(id, kClDeviceSubGroupSizesIntel, sub_groups_ret_size,
                          sub_group_sizes.data(), nullptr);
      if (status == CL_SUCCESS) {
        info.supported_wave_sizes.clear();
        for (int i = 0; i < sub_groups_count; ++i) {
          info.supported_wave_sizes.push_back(sub_group_sizes[i]);
        }
      }
    }
  }

  return info;
}

}  // namespace

CLDevice::CLDevice(cl_device_id id, cl_platform_id platform_id)
    : info_(GpuInfoFromDeviceID(id, platform_id)),
      id_(id),
      platform_id_(platform_id) {
  LoadOpenCLFunctionExtensions(platform_id_);
}

CLDevice::CLDevice(const CLDevice& device)
    : info_(device.info_), id_(device.id_), platform_id_(device.platform_id_) {}

CLDevice& CLDevice::operator=(const CLDevice& device) {
  if (this != &device) {
    info_ = device.info_;
    id_ = device.id_;
    platform_id_ = device.platform_id_;
  }
  return *this;
}

CLDevice::CLDevice(CLDevice&& device)
    : info_(std::move(device.info_)),
      id_(device.id_),
      platform_id_(device.platform_id_) {
  device.id_ = nullptr;
  device.platform_id_ = nullptr;
}

CLDevice& CLDevice::operator=(CLDevice&& device) {
  if (this != &device) {
    id_ = nullptr;
    platform_id_ = nullptr;
    info_ = std::move(device.info_);
    std::swap(id_, device.id_);
    std::swap(platform_id_, device.platform_id_);
  }
  return *this;
}

std::string CLDevice::GetPlatformVersion() const {
  return GetPlatformInfo(platform_id_, CL_PLATFORM_VERSION);
}

void CLDevice::DisableOneLayerTextureArray() {
  info_.adreno_info.support_one_layer_texture_array = false;
}

absl::Status CreateDefaultGPUDevice(CLDevice* result) {
  cl_uint num_platforms;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }
  if (num_platforms == 0) {
    return absl::UnknownError("No supported OpenCL platform.");
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }

  cl_platform_id platform_id = platforms[0];
  cl_uint num_devices;
  status =
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }
  if (num_devices == 0) {
    return absl::UnknownError("No GPU on current platform.");
  }

  std::vector<cl_device_id> devices(num_devices);
  status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices,
                          devices.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }

  *result = CLDevice(devices[0], platform_id);
  const auto& gpu_info = result->GetInfo();
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno8xx() &&
      gpu_info.adreno_info.cl_compiler_version.major == 47 &&
      gpu_info.adreno_info.cl_compiler_version.minor == 12) {
    // Observed on E031.47.12.01
    ABSL_LOG(WARNING)
        << "Adreno 8xx with compiler E031.47.12.xxx is not well supported in "
           "MLDrift OpenCL backend. Various kernel compilation errors and "
           "incorrect results. Update your driver to a newer version.";
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace ml_drift
