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

#ifndef ML_DRIFT_COMMON_GPU_INFO_H_
#define ML_DRIFT_COMMON_GPU_INFO_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

// The VendorID returned by the GPU driver.
enum class GpuVendor {
  // go/keep-sorted start
  kAMD,
  kApple,
  kBroadcom,
  kHuawei,
  kIntel,
  kLlvmPipe,
  kMali,
  kNvidia,
  kPowerVR,
  kQualcomm,
  // go/keep-sorted end
  kUnknown
};

enum class GpuApi {
  kUnknown,
  kOpenCl,
  kMetal,
  kVulkan,
  kOpenGl,
  kWebGpu,
  kCuda,
};

enum class AdrenoGpu {
  // Adreno 8xx series
  kAdreno840,
  kAdreno830,
  // Adreno 7xx series
  kAdreno750,
  kAdreno740,
  kAdreno735,
  kAdreno732,
  kAdreno730,
  kAdreno725,
  kAdreno720,
  kAdreno710,
  kAdreno702,
  // Adreno 6xx series
  kAdreno690,
  kAdreno685,
  kAdreno680,
  kAdreno675,
  kAdreno660,
  kAdreno650,
  kAdreno644,
  kAdreno642,
  kAdreno640,
  kAdreno630,
  kAdreno620,
  kAdreno619,
  kAdreno618,
  kAdreno616,
  kAdreno615,
  kAdreno613,
  kAdreno612,
  kAdreno610,
  kAdreno605,
  // Adreno 5xx series
  kAdreno540,
  kAdreno530,
  kAdreno512,
  kAdreno510,
  kAdreno509,
  kAdreno508,
  kAdreno506,
  kAdreno505,
  kAdreno504,
  // Adreno 4xx series
  kAdreno430,
  kAdreno420,
  kAdreno418,
  kAdreno405,
  // Adreno 3xx series
  kAdreno330,
  kAdreno320,
  kAdreno308,
  kAdreno306,
  kAdreno305,
  kAdreno304,
  // Adreno 2xx series
  kAdreno225,
  kAdreno220,
  kAdreno205,
  kAdreno203,
  kAdreno200,
  // Adreno 1xx series
  kAdreno130,
  kAdreno120,
  kUnknown
};

struct AdrenoInfo {
  struct OpenClCompilerVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;
  };
  struct OpenGlDriverVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;
  };
  enum class Generation {
    kUnknown = 0,
    kGen1 = 1,
    kGen2 = 2,
    kGen3 = 3,
    kGen4 = 4,
    kGen5 = 5,
    kGen6 = 6,
    kGen7 = 7,
    kGen8 = 8,
  };

  AdrenoGpu adreno_gpu;
  Generation generation;
  // compute units count using GPU name. Use GpuInfo::GetComputeUnitsCount().
  int compute_units = 1;

  bool IsAdreno1xx() const;
  bool IsAdreno2xx() const;
  bool IsAdreno3xx() const;
  bool IsAdreno4xx() const;
  bool IsAdreno5xx() const;
  bool IsAdreno6xx() const;
  bool IsAdreno7xx() const;
  bool IsAdreno8xx() const;

  bool IsLowEnd() const;

  int GetWaveSize(bool full_wave) const;

  // Not supported on some Adreno devices with specific driver version.
  // b/131099086
  bool support_one_layer_texture_array = true;
  OpenClCompilerVersion cl_compiler_version;
  OpenGlDriverVersion opengl_driver_version;

  // b/442857864
  bool IsGlDriverMajor615Minor88_97() const {
    return opengl_driver_version.major == 615 &&
           (opengl_driver_version.minor >= 88 &&
            opengl_driver_version.minor <= 97);
  }
};

// This has a small selection of AMD GPUs needed for the IsNavi10() and
// IsVega20() checks in AMDInfo.
enum class AMDGpu {
  kUnknown,
  kRadeonProVegaII,
  kRadeonVII,
  kRadeonRX5600,
  kRadeonRX5700,
  kRadeonPro5700,
  kRadeonProW5700,
};

enum class AMDArchitecture {
  kUnknown,
  kGcn1,
  kGcn2,
  kGcn3,
  kGcn4,
  kGcn5,
  kRdna1,
  kRdna2,
  kRdna3,
  kRdna4,
};

struct AMDInfo {
  explicit AMDInfo(const std::string& gpu_description);
  AMDInfo() = default;
  int shader_engines = 0;
  int compute_units_per_shader_engine = 0;
  int GetComputeUnitsCount() const {
    return shader_engines * compute_units_per_shader_engine;
  }

  AMDArchitecture architecture;
  AMDGpu amd_gpu;

  bool IsArchitectureOrNewer(AMDArchitecture arch) const;
  bool IsNavi10() const;
  bool IsVega20() const;
};

enum class AppleGpu {
  kUnknown,
  kA7,
  kA8,
  kA8X,
  kA9,
  kA9X,
  kA10,
  kA10X,
  kA11,
  kA12,
  kA12X,
  kA12Z,
  kA13,
  kA14,
  kA15,
  kA16,
  kA17Pro,
  kA18,
  kA18Pro,
  kA19,
  kA19Pro,
  kM1,
  kM1Pro,
  kM1Max,
  kM1Ultra,
  kM2,
  kM2Pro,
  kM2Max,
  kM2Ultra,
  kM3,
  kM3Pro,
  kM3Max,
  kM4,
  kM4Pro,
  kM4Max,
  kM5,
  kM5Pro,
  kM5Max,
};

struct AppleInfo {
  // https://developer.apple.com/documentation/metal/mtlgpufamily
  enum class Family {
    kApple10 = 10,
    kApple9 = 9,
    kApple8 = 8,
    kApple7 = 7,
    kApple6 = 6,
    kApple5 = 5,
    kApple4 = 4,
    kApple3 = 3,
    kApple2 = 2,
    kApple1 = 1,
  };
  AppleInfo() = default;
  explicit AppleInfo(const std::string& gpu_description);
  AppleGpu gpu_type;
  Family gpu_family;

  bool IsFamilyApple1() const;
  bool IsFamilyApple2() const;
  bool IsFamilyApple3() const;
  bool IsFamilyApple4() const;
  bool IsFamilyApple5() const;
  bool IsFamilyApple6() const;
  bool IsFamilyApple7() const;
  bool IsFamilyApple8() const;
  bool IsFamilyApple9() const;
  bool IsFamilyApple10() const;

  bool IsFamilyOrLower(Family family) const;

  bool IsLocalMemoryPreferredOverGlobal() const;

  bool IsBionic() const;
  bool IsMSeries() const;
  bool IsM1Series() const;
  bool IsM2Series() const;
  bool IsM3Series() const;
  bool IsM4Series() const;
  bool IsM5Series() const;

  bool IsSIMDMatMulSupported() const;
  // Often, fp32 alu performance is 1/2 of fp16 alu performance
  // But, on some devices, fp32 alu performance equal to fp16 alu performance,
  // at least in some scenarios.
  // This method returns true if SIMDMatMul performance in fp32 equal to fp16
  bool IsSIMDMatMulFp32Perf2x() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  int GetComputeUnitsCount() const;

  // do not use, for internal usage
  void SetComputeUnits(int compute_units_count);

 private:
  Family GetGpuFamily() const;
  int compute_units = -1;
};

enum class BroadcomGpu {
  kUnknown,
  kVideoCore7,
};

struct BroadcomInfo {
  enum class Gen {
    kUnknown = 0,
  };
  BroadcomGpu gpu_version;
  Gen generation;

  // returns approximate compute units count using GPU name
  int GetApproximateComputeUnitsCount() const;

  bool IsVideoCore7() const;
};

enum class MaliGpu {
  kUnknown,
  kT604,
  kT622,
  kT624,
  kT628,
  kT658,
  kT678,
  kT720,
  kT760,
  kT820,
  kT830,
  kT860,
  kT880,
  kG31,
  kG51,
  kG71,
  kG52,
  kG72,
  kG76,
  kG57,
  kG77,
  kG68,
  kG78,
  kG310,
  kG510,
  kG610,
  kG710,
  kG615,
  kG715,
  kG620,
  kG720,
  kG925,
  kG1Pro,
  kG1Premium,
  kG1Ultra,
};

struct MaliInfo {
  enum class Gen {
    kUnknown = 0,
    kMidgardV1 = 1,
    kMidgardV2 = 2,
    kMidgardV3 = 3,
    kBifrostV1 = 4,
    kBifrostV2 = 5,
    kBifrostV3 = 6,
    kValhallV1 = 7,
    kValhallV2 = 8,
    kValhallV3 = 9,
    kValhallV4 = 10,
    kV5 = 11,
  };
  MaliGpu gpu_version;
  Gen generation;

  bool IsMaliT6xx() const;
  bool IsMaliT7xx() const;
  bool IsMaliT8xx() const;
  bool IsMidgard() const;
  bool IsBifrostGen1() const;
  bool IsBifrostGen2() const;
  bool IsBifrostGen3() const;
  bool IsBifrost() const;
  bool IsValhallGen1() const;
  bool IsValhallGen2() const;
  bool IsValhallGen3() const;
  bool IsValhallGen4() const;
  bool IsValhall() const;
  bool IsGen5() const;
  bool IsMaliG1() const;

  // returns approximate compute units count using GPU name
  int GetApproximateComputeUnitsCount() const;
};

enum class PowerVRGpu {
  kUnknown,
  kRogue,
  // New generation of IMG gpus after 2019:
  kAXE,
  kAXM,
  kAXT,
  kBXE,
  kBXM,
  kBXS,
  kBXT,
  kCXT,
  kDXT,
};

struct PowerVRInfo {
  struct DriverVersion {
    int branch_main = 0;
    int branch_minor = 0;
    int id = 0;
    bool IsHigherOrEqual(const DriverVersion& other) const {
      if (branch_main > other.branch_main) return true;
      if (branch_main < other.branch_main) return false;
      if (branch_minor > other.branch_minor) return true;
      if (branch_minor < other.branch_minor) return false;
      return id >= other.id;
    }
  };
  PowerVRInfo() = default;
  explicit PowerVRInfo(const std::string& gpu_description);
  PowerVRGpu gpu_version;
  DriverVersion driver_version;

  bool IsRogue() const;
  bool IsImgAxx() const;
  bool IsImgBxx() const;
  bool IsImgCxx() const;
  bool IsImgDxx() const;
};

enum class NvidiaArchitecture {
  kUnknown,
  kFermi,
  kKepler,
  kMaxwell,
  kPascal,
  kVolta,
  kTuring,
  kAmpere,
  kLovelace,
  kHopper,
  kBlackwell,
};

enum class NvidiaGpu {
  kUnknown,
  // Fermi
  kGTX580,
  // Kepler
  kGTX780,
  kK40,
  kK80,
  // Maxwell
  kGTX960,
  kGTX970,
  kGTX980,
  kGTX980Ti,
  kTitanX,
  kM40,
  // Pascal
  kGTX1050,
  kGTX1050Ti,
  kGTX1060,
  kGTX1070,
  kGTX1070Ti,
  kGTX1080,
  kGTX1080Ti,
  kTitanXp,
  kP4,
  kP40,
  kP100,
  // Volta
  kV100,
  kTitanV,
  // Turing
  kGTX1650,
  kGTX1650Super,
  kGTX1660,
  kGTX1660Super,
  kGTX1660Ti,
  kRTX2060,
  kRTX2060Super,
  kRTX2070,
  kRTX2070Super,
  kRTX2080,
  kRTX2080Super,
  kRTX2080Ti,
  kTitanRTX,
  kT4,
  kRTX4000Turing,
  kRTX5000Turing,
  kRTX6000Turing,
  kRTX8000Turing,
  // Ampere
  kRTX3050,
  kRTX3060,
  kRTX3060Ti,
  kRTX3070,
  kRTX3070Ti,
  kRTX3080,
  kRTX3080Ti,
  kRTX3090,
  kRTX3090Ti,
  kA2,
  kA16,
  kA10G,
  kA30,
  kA40,
  kA100,
  kA800,
  kRTXA2000,
  kRTXA4000,
  kRTXA4500,
  kRTXA5000,
  kRTXA6000,
  // Lovelace
  kRTX4060,
  kRTX4060Ti,
  kRTX4070,
  kRTX4070Super,
  kRTX4070Ti,
  kRTX4070TiSuper,
  kRTX4080,
  kRTX4080Super,
  kRTX4090,
  kL4,
  kL20,
  kL40,
  kL40S,
  kRTX2000Ada,
  kRTX3500Ada,
  kRTX4000Ada,
  kRTX5000Ada,
  kRTX6000Ada,
  // Hopper
  kH100,
  kH200,
  kH800,
  kGH200,
  // Blackwell
  kRTX5060,
  kRTX5060Ti,
  kRTX5070,
  kRTX5070Ti,
  kRTX5080,
  kRTX5090,
  kB100,
  kB200,
  kB800,
  kGB200,
};

struct NvidiaInfo {
  NvidiaInfo() = default;
  explicit NvidiaInfo(const std::string& gpu_description);

  NvidiaArchitecture architecture;
  NvidiaGpu gpu_type = NvidiaGpu::kUnknown;

  bool IsArchitectureOrNewer(NvidiaArchitecture arch) const;
  int GetComputeUnitsCount() const;
};

struct OpenGlInfo {
  std::string renderer_name;
  std::string vendor_name;
  std::string version;
  int major_version = -1;
  int minor_version = -1;

  int max_ssbo_bindings = 0;
  int max_image_bindings = 0;
  int max_texture_bindings = 0;
  int max_work_group_invocations = 0;
  int max_texture_size = 0;
  int max_array_texture_layers = 0;
  int max_fragment_uniform_vec4_count = 0;
  int max_color_atttachments = 0;
  int max_viewport_width = 0;
  int max_viewport_height = 0;
  int max_renderbuffer_size = 0;
  int max_shader_storage_block_size = 0;

  std::vector<std::string> extensions;
  int max_compute_work_group_size_x;
  int max_compute_work_group_size_y;
  int max_compute_work_group_size_z;

  struct SubgroupInfo {
    int size = -1;
    bool basic = false;
    bool vote = false;
    bool arithmetic = false;
    bool ballot = false;
    bool shuffle = false;
    bool shuffle_relative = false;
    bool clustered = false;
    bool quad = false;
  } subgroup_info;

  bool SupportsExplicitFp16() const;

  bool IsApiOpenGl31OrAbove() const;
  bool IsApiOpenGl32OrAbove() const;
};

struct VulkanInfo {
  std::string vendor_name;
  uint32_t api_version = -1;
  uint32_t api_version_major = -1;
  uint32_t api_version_minor = -1;
  uint32_t api_version_patch = -1;

  uint32_t max_per_stage_descriptor_sampled_images = 0;
  uint32_t max_per_stage_descriptor_storage_images = 0;
  uint32_t max_per_stage_descriptor_storage_buffers = 0;
  uint32_t max_compute_work_group_invocations;
  uint32_t max_image_dimension_1d;
  uint32_t max_image_dimension_2d;
  uint32_t max_image_dimension_3d;
  uint32_t max_image_array_layers;
  uint64_t max_texel_buffer_elements;
  uint64_t max_uniform_buffer_range;
  uint64_t max_storage_buffer_range;
  uint64_t max_push_constants_size;

  uint32_t subgroup_size = 0;
  bool supports_subgroup_arithmetic = false;

  std::vector<std::string> extensions;
  int max_compute_work_group_size_x;
  int max_compute_work_group_size_y;
  int max_compute_work_group_size_z;

  bool SupportsExplicitFp16() const;
};

enum class OpenClVersion {
  kCl1_0,
  kCl1_1,
  kCl1_2,
  kCl2_0,
  kCl2_1,
  kCl2_2,
  kCl3_0,
  kUnknown,
};
std::string OpenClVersionToString(OpenClVersion version);

struct OpenClInfo {
  std::string device_name;
  std::string vendor_name;
  std::string opencl_c_version;
  std::string platform_version;
  std::string driver_version;

  OpenClVersion cl_version;

  std::vector<std::string> extensions;
  bool supports_fp16;
  bool supports_image3d_writes;
  bool supports_images;
  int compute_units_count;
  uint64_t buffer_max_size;
  uint64_t max_allocation_size;
  uint64_t image2d_max_width;
  uint64_t image2d_max_height;
  uint64_t image_buffer_max_size;
  uint64_t image_array_max_layers;
  uint64_t image3d_max_width;
  uint64_t image3d_max_height;
  uint64_t image3d_max_depth;
  int max_work_group_size_x;
  int max_work_group_size_y;
  int max_work_group_size_z;
  int max_work_group_total_size;
  bool dedicated_local_memory;

  // The row pitch alignment size in pixels for 2D images created from a buffer.
  // The value must be a power of 2.
  uint64_t image_pitch_alignment = 0;
  // The minimum alignment in pixels. The value must be a power of 2.
  uint64_t image_base_address_alignment = 0;
  uint64_t base_addr_align_in_bits;

  // rtn is ROUND_TO_NEAREST
  // with rtn precision is much better then with rtz (ROUND_TO_ZERO)
  // Adreno 3xx supports only rtz, Adreno 4xx and more support rtn
  // Mali from T6xx supports rtn
  // PowerVR supports only rtz
  bool supports_fp32_rtn;
  bool supports_fp16_rtn;

  bool supports_register_allocation_arm = false;

  // Whether or not the device flushes denormals to zero for fp16.
  bool is_fp16_ftz_hardware_forced = true;

  // cl_khr_device_uuid extension values
  std::vector<uint8_t> device_uuid_khr;
  std::vector<uint8_t> driver_uuid_khr;

  struct ClKhrIntegerDotProductInfo {
    bool suppports_4x8bit_packed = false;
    bool suppports_4x8bit = false;
    struct AccelerationProperties {
      bool signed_accelerated;
      bool unsigned_accelerated;
      bool mixed_signedness_accelerated;
      bool accumulating_saturating_signed_accelerated;
      bool accumulating_saturating_unsigned_accelerated;
      bool accumulating_saturating_mixed_signedness_accelerated;
    };
    AccelerationProperties acceleration_info_4x8bit_packed;
    AccelerationProperties acceleration_info_4x8bit;
  } cl_khr_integer_dot_product_info;

  struct SupportedImage2dTypes {
    absl::flat_hash_set<DataType> r_layout;
    absl::flat_hash_set<DataType> rg_layout;
    absl::flat_hash_set<DataType> rgb_layout;
    absl::flat_hash_set<DataType> rgba_layout;

    bool SupportsImage2D(DataType data_type, int channels) const;
  };

  SupportedImage2dTypes supported_images_2d;

  bool IsImage2dFromBufferSupported() const;

  bool IsCLVK() const { return absl::StrContains(platform_version, "clvk"); }
};

enum class MetalLanguageVersion {
  kMetal1_0,
  kMetal1_1,
  kMetal1_2,
  kMetal2_0,
  kMetal2_1,
  kMetal2_2,
  kMetal2_3,
  kMetal2_4,
  kMetal3_0,
  kMetal3_1,
  kMetal3_2,
  kMetal4_0,
  kUnknown,
};

struct MetalInfo {
  MetalLanguageVersion language_version;

  int max_work_group_size_x;
  int max_work_group_size_y;
  int max_work_group_size_z;

  uint64_t buffer_max_size;

  uint64_t image2d_max_width;
  uint64_t image2d_max_height;
  uint64_t image_array_max_layers;
  uint64_t image3d_max_width;
  uint64_t image3d_max_height;
  uint64_t image3d_max_depth;

  bool is_simulator;

  bool IsSIMDMatMulSupported() const;
  // MSL is Metal shading language
  bool IsMslVersionEqualOrHigher(int major, int minor = 0) const;

  bool IsNativeBfloatSupported() const;
};

struct WebGpuInfo {
  // The maximum value of the workgroup_size X dimension for a compute stage
  // GPUShaderModule entry-point.
  int max_compute_workgroup_size_x = 256;
  // The maximum value of the workgroup_size Y dimensions for a compute stage
  // GPUShaderModule entry-point.
  int max_compute_workgroup_size_y = 256;
  // The maximum value of the workgroup_size Z dimensions for a compute stage
  // GPUShaderModule entry-point.
  int max_compute_workgroup_size_z = 64;
  // The maximum value of the product of the workgroup_size dimensions for a
  // compute stage GPUShaderModule entry-point.
  int max_compute_invocations_per_workgroup = 256;
  // The maximum total workgroup storage size.
  int max_compute_workgroup_storage_size = 16384;
  // The maximum number of compute workgroups per dimension.
  uint32_t max_compute_workgroups_per_dimension = 65535u;

  // The maximum GPUBufferBinding.size for bindings with a
  // GPUBindGroupLayoutEntry entry for which entry.buffer?.type is "storage" or
  // "read-only-storage".
  uint64_t max_storage_buffer_binding_size = 134217728;  // 128mb
  // The maximum size of size when creating a GPUBuffer.
  uint64_t max_buffer_size = 268435456;  // 256mb

  // The maximum number of storage buffers for a single shader stage.
  uint32_t max_storage_buffers_per_shader_stage = 8;
  // The maximum number of storage textures for a single shader stage.
  uint32_t max_storage_textures_per_shader_stage = 4;
  // The maximum number of sampled textures for a single shader stage.
  uint32_t max_sampled_textures_per_shader_stage = 16;

  // The maximum allowed value for the size.width of a texture created with
  // dimension "1d".
  uint64_t max_texture_dimension_1d = 8192;
  // The maximum allowed value for the size.width and size.height of a texture
  // created with dimension "2d".
  uint64_t max_texture_dimension_2d = 8192;
  // The maximum allowed value for the size.width, size.height and
  // size.depthOrArrayLayers of a texture created with dimension "3d".
  uint64_t max_texture_dimension_3d = 2048;
  uint64_t max_texture_array_layers = 256;
  // Whether or not the device supports subgroups.
  // Subgroups with f16 operands are supported when both subgroups and
  // fp16 are supported.
  bool supports_subgroups = false;
  uint32_t min_subgroup_size = 4;
  uint32_t max_subgroup_size = 128;
  // Whether or not the device supports fp16 math.
  bool supports_fp16 = false;
  bool supports_timestamp_query = false;

  bool supports_subgroup_matrix = false;

  // Whether or not the device is using an integrated GPU.
  bool is_integrated_gpu = false;

  // Whether or not the device supports the HostMappedPointer feature.
  bool supports_host_mapped_pointer = false;
};

struct CudaInfo {
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int multiprocessor_count = 1;
};

enum class IntelGeneration {
  kUnknown,
  kGen9,
  kGen10,
  kGen11,
  kGen12,
  kGen13,
  kGen14,
};

struct IntelInfo {
  IntelInfo() = default;
  explicit IntelInfo(const std::string& gpu_description);

  IntelGeneration generation;

  bool IsGenerationOrNewer(IntelGeneration gen) const;
};

enum class MaleoonGpu {
  kMaleoon910,
  kMaleoon920,
  kMaleoon930,
  kUnknown,
};

struct MaleoonInfo {
  MaleoonGpu gpu;
};

// Describes the parameters of a wave matrix multiplication (with accumulation)
// operation.
// RESULT (MxN) += LEFT (MxK) * RIGHT (KxN)
struct WaveMatMulOpDescriptor {
  int m_size;
  int n_size;
  int k_size;
  DataType left_type;
  DataType right_type;
  DataType result_type;

  bool operator==(const WaveMatMulOpDescriptor& d) const {
    return d.m_size == m_size && d.n_size == n_size && d.k_size == k_size &&
           d.left_type == left_type && d.right_type == right_type &&
           d.result_type == result_type;
  }
};

struct GpuInfo {
  bool IsAdreno() const;
  bool IsApple() const;
  bool IsBroadcom() const;
  bool IsMali() const;
  bool IsPowerVR() const;
  bool IsNvidia() const;
  bool IsAMD() const;
  bool IsIntel() const;
  bool IsMaleoon() const;
  bool IsLlvmPipe() const;

  bool IsGlsl() const;
  bool IsGlslSupportsExplicitFp16() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  // We often use this blocks(R, S, W0-W3 is vec4) in code generation:
  // Block in MAD form:
  //   R += S.x * W0;
  //   R += S.y * W1;
  //   R += S.z * W2;
  //   R += S.w * W3;
  // Block in MAD form(W0t-W3t can be received from W0-W3 by transposition):
  //   R.x += dot(S, W0t);
  //   R.y += dot(S, W1t);
  //   R.z += dot(S, W2t);
  //   R.w += dot(S, W3t);
  // Some devices(compilers) 'prefer' MAD(usually) or DOT forms
  // This function is to determine prefered form
  bool IsDotPreferred() const;

  bool SupportsFP16() const;

  bool SupportsImages() const;
  bool SupportsTextureArray() const;
  bool SupportsImageBuffer() const;
  bool SupportsImage3D() const;

  bool SupportsPointersInKernels() const;

  // returns true if device have fixed wave size equal to 32
  bool IsWaveSizeEqualTo32() const;
  bool SupportsSubGroupWithSize(int sub_group_size) const;

  bool SupportsImage2D(DataType data_type, int channels) const;
  bool SupportsExtension(const std::string& extension) const;

  bool SupportsZeroClampForImageBuffer() const;
  bool SupportsZeroClampForImages() const;

  bool SupportsAcceleratedDp4a() const;

  int GetComputeUnitsCount() const;

  int GetMaxImageArguments() const;
  int GetMaxTextureArguments() const;
  int GetMaxBufferArguments() const;

  int GetMaxWorkGroupSizeForX() const;
  int GetMaxWorkGroupSizeForY() const;
  int GetMaxWorkGroupSizeForZ() const;
  int GetMaxWorkGroupTotalSize() const;

  uint32_t GetMaxWorkGroupsCountForX() const;
  uint32_t GetMaxWorkGroupsCountForY() const;
  uint32_t GetMaxWorkGroupsCountForZ() const;

  uint64_t GetMaxImage2DWidth() const;
  uint64_t GetMaxImage2DHeight() const;
  uint64_t GetMaxImage2DArrayLayers() const;
  uint64_t GetMaxImage3DWidth() const;
  uint64_t GetMaxImage3DHeight() const;
  uint64_t GetMaxImage3DDepth() const;
  uint64_t GetMaxBufferSize() const;
  uint64_t GetMaxMemoryAllocationSize() const;
  uint64_t GetMaxImageBufferWidth() const;

  GpuVendor vendor = GpuVendor::kUnknown;
  GpuApi gpu_api = GpuApi::kUnknown;

  std::vector<int> supported_wave_sizes;
  std::vector<WaveMatMulOpDescriptor>
      wave_mat_mul_ops;  // supported wave matrix multiplication operations on
                         // this GPU

  bool SupportsWaveMatMulOp(const WaveMatMulOpDescriptor& op_descriptor) const;

  AdrenoInfo adreno_info;
  AMDInfo amd_info;
  AppleInfo apple_info;
  BroadcomInfo broadcom_info;
  MaliInfo mali_info;
  PowerVRInfo powervr_info;
  IntelInfo intel_info;
  NvidiaInfo nvidia_info;
  MaleoonInfo maleoon_info;

  // OpenGL specific, gpu_api should be kOpenGl
  OpenGlInfo opengl_info;
  bool IsApiOpenGl() const;
  bool IsApiOpenGl31OrAbove() const;

  // Vulkan specific, gpu_api should be kVulkan
  VulkanInfo vulkan_info;
  bool IsApiVulkan() const;

  MetalInfo metal_info;
  bool IsApiMetal() const;

  OpenClInfo opencl_info;
  bool IsApiOpenCl() const;
  bool IsCL11OrHigher() const;
  bool IsCL20OrHigher() const;
  bool IsCL30OrHigher() const;

  // WebGpu
  WebGpuInfo webgpu_info;
  bool IsApiWebGpu() const;

  // Cuda
  CudaInfo cuda_info;
  bool IsApiCuda() const;
};

// Currently it initializes:
// vendor
// AdrenoInfo if vendor is kQualcomm
// AppleInfo if vendor is kApple
// MaliInfo if vendor is kMali
// PowerVRInfo if vendor is kPowerVR
void GetGpuInfoFromDeviceDescription(const std::string& gpu_description,
                                     GpuApi gpu_api, GpuInfo* gpu_info);

bool IsValidWorkgroup(const GpuInfo& gpu_info, const int3& wg_size);

std::string ToString(const OpenClInfo::ClKhrIntegerDotProductInfo& int8_info);
std::string ToString(const WaveMatMulOpDescriptor& op_descriptor);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_INFO_H_
