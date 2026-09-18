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

#include "ml_drift/common/gpu_info.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace {

GpuVendor GetGpuVendor(const std::string& gpu_description) {
  const std::map<std::string, GpuVendor> kMapping = {
      {"adreno", GpuVendor::kQualcomm},
      {"apple", GpuVendor::kApple},
      {"qualcomm", GpuVendor::kQualcomm},
      {"mali", GpuVendor::kMali},
      {"powervr", GpuVendor::kPowerVR},
      {"advanced micro devices", GpuVendor::kAMD},
      {"intel", GpuVendor::kIntel},
      {"nvidia", GpuVendor::kNvidia},
      {"quadro", GpuVendor::kNvidia},
      {"tesla", GpuVendor::kNvidia},
      {"amd", GpuVendor::kAMD},
      {"radeon", GpuVendor::kAMD},
      {"xclipse", GpuVendor::kAMD},
      {"power", GpuVendor::kPowerVR},
      {"maleoon", GpuVendor::kHuawei},
      {"broadcom", GpuVendor::kBroadcom},
      {"lavapipe", GpuVendor::kLlvmPipe},
      {"llvmpipe", GpuVendor::kLlvmPipe},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return GpuVendor::kUnknown;
}

AdrenoInfo GetAdrenoInfo(const std::string& gpu_description) {
  const std::map<std::string, AdrenoInfo> kMapping = {
      // Adreno 8xx series
      {"840", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno840,
                         /*generation=*/AdrenoInfo::Generation::kGen8,
                         /*compute_units=*/12}},
      {"831", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno831,
                         /*generation=*/AdrenoInfo::Generation::kGen8,
                         /*compute_units=*/12}},
      {"830", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno830,
                         /*generation=*/AdrenoInfo::Generation::kGen8,
                         /*compute_units=*/12}},
      {"829", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno829,
                         /*generation=*/AdrenoInfo::Generation::kGen8,
                         /*compute_units=*/8}},
      {"825", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno825,
                         /*generation=*/AdrenoInfo::Generation::kGen8,
                         /*compute_units=*/8}},
      {"812", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno812,
                         /*generation=*/AdrenoInfo::Generation::kGen8,
                         /*compute_units=*/2}},
      {"810", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno810,
                         /*generation=*/AdrenoInfo::Generation::kGen8,
                         /*compute_units=*/2}},
      // Adreno 7xx series
      {"750", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno750,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/6}},
      {"740", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno740,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/6}},
      {"735", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno735,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/3}},
      {"732", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno732,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/3}},
      {"730", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno730,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/4}},
      {"725", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno725,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/4}},
      {"720", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno720,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/3}},
      {"710", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno710,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/2}},
      {"702", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno702,
                         /*generation=*/AdrenoInfo::Generation::kGen7,
                         /*compute_units=*/1}},
      // Adreno 6xx series
      {"690", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno690,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/4}},
      {"685", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno685,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/4}},
      {"680", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno680,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/4}},
      {"675", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno675,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/4}},
      {"660", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno660,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/3}},
      {"650", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno650,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/3}},
      {"644", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno644,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/2}},
      {"642", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno642,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/2}},
      {"640", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno640,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/2}},
      {"630", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno630,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/2}},
      {"620", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno620,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"619", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno619,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"618", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno618,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"616", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno616,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"615", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno615,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"613", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno613,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"612", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno612,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"610", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno610,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      {"605", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno605,
                         /*generation=*/AdrenoInfo::Generation::kGen6,
                         /*compute_units=*/1}},
      // Adreno 5xx series
      {"540", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno540,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/4}},
      {"530", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno530,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/4}},
      {"512", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno512,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/2}},
      {"510", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno510,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/2}},
      {"509", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno509,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/2}},
      {"508", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno508,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/1}},
      {"506", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno506,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/1}},
      {"505", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno505,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/1}},
      {"504", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno504,
                         /*generation=*/AdrenoInfo::Generation::kGen5,
                         /*compute_units=*/1}},
      // Adreno 4xx series
      {"430", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno430,
                         /*generation=*/AdrenoInfo::Generation::kGen4,
                         /*compute_units=*/4}},
      {"420", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno420,
                         /*generation=*/AdrenoInfo::Generation::kGen4,
                         /*compute_units=*/4}},
      {"418", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno418,
                         /*generation=*/AdrenoInfo::Generation::kGen4,
                         /*compute_units=*/2}},
      {"405", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno405,
                         /*generation=*/AdrenoInfo::Generation::kGen4,
                         /*compute_units=*/1}},
      // Adreno 3xx series
      {"330", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno330,
                         /*generation=*/AdrenoInfo::Generation::kGen3,
                         /*compute_units=*/4}},
      {"320", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno320,
                         /*generation=*/AdrenoInfo::Generation::kGen3,
                         /*compute_units=*/2}},
      {"308", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno308,
                         /*generation=*/AdrenoInfo::Generation::kGen3,
                         /*compute_units=*/1}},
      {"306", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno306,
                         /*generation=*/AdrenoInfo::Generation::kGen3,
                         /*compute_units=*/1}},
      {"305", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno305,
                         /*generation=*/AdrenoInfo::Generation::kGen3,
                         /*compute_units=*/1}},
      {"304", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno304,
                         /*generation=*/AdrenoInfo::Generation::kGen3,
                         /*compute_units=*/1}},
      // Adreno 2xx series
      {"225", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno225,
                         /*generation=*/AdrenoInfo::Generation::kGen2,
                         /*compute_units=*/1}},
      {"220", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno220,
                         /*generation=*/AdrenoInfo::Generation::kGen2,
                         /*compute_units=*/1}},
      {"205", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno205,
                         /*generation=*/AdrenoInfo::Generation::kGen2,
                         /*compute_units=*/1}},
      {"203", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno203,
                         /*generation=*/AdrenoInfo::Generation::kGen2,
                         /*compute_units=*/1}},
      {"200", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno200,
                         /*generation=*/AdrenoInfo::Generation::kGen2,
                         /*compute_units=*/1}},
      // Adreno 1xx series
      {"130", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno130,
                         /*generation=*/AdrenoInfo::Generation::kGen1,
                         /*compute_units=*/1}},
      {"120", AdrenoInfo{/*adreno_gpu=*/AdrenoGpu::kAdreno120,
                         /*generation=*/AdrenoInfo::Generation::kGen1,
                         /*compute_units=*/1}},
  };

  AdrenoInfo adreno_info;
  adreno_info.adreno_gpu = AdrenoGpu::kUnknown;
  adreno_info.compute_units = 1;
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      adreno_info = v.second;
      break;
    }
  }
  return adreno_info;
}

BroadcomInfo GetBroadcomInfo(const std::string& gpu_description) {
  const std::vector<std::pair<std::string, BroadcomInfo>> kMapping = {
      {"v3d 7", BroadcomInfo{/*gpu_version=*/BroadcomGpu::kVideoCore7,
                             /*generation=*/BroadcomInfo::Gen::kUnknown}},
  };
  BroadcomInfo broadcom_info;
  broadcom_info.gpu_version = BroadcomGpu::kUnknown;
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      broadcom_info = v.second;
      break;
    }
  }
  return broadcom_info;
}

MaliInfo GetMaliInfo(const std::string& gpu_description) {
  // Order must be preserved
  const std::vector<std::pair<std::string, MaliInfo>> kMapping = {
      {"t604", MaliInfo{/*gpu_version=*/MaliGpu::kT604,
                        /*generation=*/MaliInfo::Gen::kMidgardV1}},
      {"t622", MaliInfo{/*gpu_version=*/MaliGpu::kT622,
                        /*generation=*/MaliInfo::Gen::kMidgardV1}},
      {"t624", MaliInfo{/*gpu_version=*/MaliGpu::kT624,
                        /*generation=*/MaliInfo::Gen::kMidgardV1}},
      {"t628", MaliInfo{/*gpu_version=*/MaliGpu::kT628,
                        /*generation=*/MaliInfo::Gen::kMidgardV1}},
      {"t658", MaliInfo{/*gpu_version=*/MaliGpu::kT658,
                        /*generation=*/MaliInfo::Gen::kMidgardV1}},
      {"t678", MaliInfo{/*gpu_version=*/MaliGpu::kT678,
                        /*generation=*/MaliInfo::Gen::kMidgardV1}},
      {"t720", MaliInfo{/*gpu_version=*/MaliGpu::kT720,
                        /*generation=*/MaliInfo::Gen::kMidgardV2}},
      {"t760", MaliInfo{/*gpu_version=*/MaliGpu::kT760,
                        /*generation=*/MaliInfo::Gen::kMidgardV2}},
      {"t820", MaliInfo{/*gpu_version=*/MaliGpu::kT820,
                        /*generation=*/MaliInfo::Gen::kMidgardV3}},
      {"t830", MaliInfo{/*gpu_version=*/MaliGpu::kT830,
                        /*generation=*/MaliInfo::Gen::kMidgardV3}},
      {"t860", MaliInfo{/*gpu_version=*/MaliGpu::kT860,
                        /*generation=*/MaliInfo::Gen::kMidgardV3}},
      {"t880", MaliInfo{/*gpu_version=*/MaliGpu::kT880,
                        /*generation=*/MaliInfo::Gen::kMidgardV3}},
      {"g310", MaliInfo{/*gpu_version=*/MaliGpu::kG310,
                        /*generation=*/MaliInfo::Gen::kValhallV3}},
      {"g31", MaliInfo{/*gpu_version=*/MaliGpu::kG31,
                       /*generation=*/MaliInfo::Gen::kBifrostV1}},
      {"g510", MaliInfo{/*gpu_version=*/MaliGpu::kG510,
                        /*generation=*/MaliInfo::Gen::kValhallV3}},
      {"g51", MaliInfo{/*gpu_version=*/MaliGpu::kG51,
                       /*generation=*/MaliInfo::Gen::kBifrostV1}},
      {"g52", MaliInfo{/*gpu_version=*/MaliGpu::kG52,
                       /*generation=*/MaliInfo::Gen::kBifrostV2}},
      {"g57", MaliInfo{/*gpu_version=*/MaliGpu::kG57,
                       /*generation=*/MaliInfo::Gen::kValhallV1}},
      {"g610", MaliInfo{/*gpu_version=*/MaliGpu::kG610,
                        /*generation=*/MaliInfo::Gen::kValhallV3}},
      {"g615", MaliInfo{/*gpu_version=*/MaliGpu::kG615,
                        /*generation=*/MaliInfo::Gen::kValhallV4}},
      {"g620", MaliInfo{/*gpu_version=*/MaliGpu::kG620,
                        /*generation=*/MaliInfo::Gen::kV5}},
      {"g68", MaliInfo{/*gpu_version=*/MaliGpu::kG68,
                       /*generation=*/MaliInfo::Gen::kValhallV2}},
      {"g710", MaliInfo{/*gpu_version=*/MaliGpu::kG710,
                        /*generation=*/MaliInfo::Gen::kValhallV3}},
      {"g715", MaliInfo{/*gpu_version=*/MaliGpu::kG715,
                        /*generation=*/MaliInfo::Gen::kValhallV4}},
      {"g720", MaliInfo{/*gpu_version=*/MaliGpu::kG720,
                        /*generation=*/MaliInfo::Gen::kV5}},
      {"g71", MaliInfo{/*gpu_version=*/MaliGpu::kG71,
                       /*generation=*/MaliInfo::Gen::kBifrostV1}},
      {"g72", MaliInfo{/*gpu_version=*/MaliGpu::kG72,
                       /*generation=*/MaliInfo::Gen::kBifrostV2}},
      {"g76", MaliInfo{/*gpu_version=*/MaliGpu::kG76,
                       /*generation=*/MaliInfo::Gen::kBifrostV3}},
      {"g77", MaliInfo{/*gpu_version=*/MaliGpu::kG77,
                       /*generation=*/MaliInfo::Gen::kValhallV1}},
      {"g78", MaliInfo{/*gpu_version=*/MaliGpu::kG78,
                       /*generation=*/MaliInfo::Gen::kValhallV2}},
      {"g925", MaliInfo{/*gpu_version=*/MaliGpu::kG925,
                        /*generation=*/MaliInfo::Gen::kV5}},
      {"g1-pro", MaliInfo{/*gpu_version=*/MaliGpu::kG1Pro,
                          /*generation=*/MaliInfo::Gen::kV5}},
      {"g1-premium", MaliInfo{/*gpu_version=*/MaliGpu::kG1Premium,
                              /*generation=*/MaliInfo::Gen::kV5}},
      {"g1-ultra", MaliInfo{/*gpu_version=*/MaliGpu::kG1Ultra,
                            /*generation=*/MaliInfo::Gen::kV5}}};
  MaliInfo mali_info;
  mali_info.gpu_version = MaliGpu::kUnknown;
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      mali_info = v.second;
      break;
    }
  }
  return mali_info;
}

PowerVRGpu GetPowerVRGpuVersion(const std::string& gpu_description) {
  // Order must be preserved
  const std::vector<std::pair<std::string, PowerVRGpu>> kMapping = {
      {"rogue", PowerVRGpu::kRogue},     {"axe", PowerVRGpu::kAXE},
      {"axm", PowerVRGpu::kAXM},         {"axt", PowerVRGpu::kAXT},
      {"bxe", PowerVRGpu::kBXE},         {"bxm", PowerVRGpu::kBXM},
      {"bxs", PowerVRGpu::kBXS},         {"bxt", PowerVRGpu::kBXT},
      {"cxt", PowerVRGpu::kCXT},         {"dxt", PowerVRGpu::kDXT},
      {"powervr g", PowerVRGpu::kRogue},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }

  return PowerVRGpu::kUnknown;
}

IntelGeneration GetIntelGeneration(const std::string& gpu_description) {
  // Order must be preserved
  const std::vector<std::pair<std::string, IntelGeneration>> kMapping = {
      {"gen-9", IntelGeneration::kGen9},   {"gen-10", IntelGeneration::kGen10},
      {"gen-11", IntelGeneration::kGen11}, {"gen-12", IntelGeneration::kGen12},
      {"xe-lpg", IntelGeneration::kGen12}, {"gen-13", IntelGeneration::kGen13},
      {"xe-2", IntelGeneration::kGen13},   {"xe-3", IntelGeneration::kGen14},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }

  return IntelGeneration::kUnknown;
}

NvidiaArchitecture GetNvidiaArchitecture(const std::string& gpu_description) {
  // Order must be preserved
  const std::vector<std::pair<std::string, NvidiaArchitecture>> kMapping = {
      {"fermi", NvidiaArchitecture::kFermi},
      {"kepler", NvidiaArchitecture::kKepler},
      {"maxwell", NvidiaArchitecture::kMaxwell},
      {"pascal", NvidiaArchitecture::kPascal},
      {"volta", NvidiaArchitecture::kVolta},
      {"turing", NvidiaArchitecture::kTuring},
      {"ampere", NvidiaArchitecture::kAmpere},
      {"lovelace", NvidiaArchitecture::kLovelace},
      {"hopper", NvidiaArchitecture::kHopper},
      {"blackwell", NvidiaArchitecture::kBlackwell},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }

  return NvidiaArchitecture::kUnknown;
}

NvidiaGpu GetNvidiaGpu(const std::string& gpu_description) {
  // Order must be preserved (more specific names before general ones, e.g., 980
  // ti before 980)
  const std::vector<std::pair<std::string, NvidiaGpu>> kMapping = {
      // Lovelace / Ada Workstation
      {"rtx 6000 ada", NvidiaGpu::kRTX6000Ada},
      {"rtx 5000 ada", NvidiaGpu::kRTX5000Ada},
      {"rtx 4000 ada", NvidiaGpu::kRTX4000Ada},
      {"rtx 3500 ada", NvidiaGpu::kRTX3500Ada},
      {"rtx 2000 ada", NvidiaGpu::kRTX2000Ada},
      // Ampere Workstation
      {"rtx a6000", NvidiaGpu::kRTXA6000},
      {"rtx a5000", NvidiaGpu::kRTXA5000},
      {"rtx a4500", NvidiaGpu::kRTXA4500},
      {"rtx a4000", NvidiaGpu::kRTXA4000},
      {"rtx a2000", NvidiaGpu::kRTXA2000},
      // Blackwell
      {"5090", NvidiaGpu::kRTX5090},
      {"5080", NvidiaGpu::kRTX5080},
      {"5070 ti", NvidiaGpu::kRTX5070Ti},
      {"5070", NvidiaGpu::kRTX5070},
      {"5060 ti", NvidiaGpu::kRTX5060Ti},
      {"5060", NvidiaGpu::kRTX5060},
      {"gb200", NvidiaGpu::kGB200},
      {"b800", NvidiaGpu::kB800},
      {"b200", NvidiaGpu::kB200},
      {"b100", NvidiaGpu::kB100},
      // Hopper
      {"gh200", NvidiaGpu::kGH200},
      {"h800", NvidiaGpu::kH800},
      {"h200", NvidiaGpu::kH200},
      {"h100", NvidiaGpu::kH100},
      // Lovelace
      {"4090", NvidiaGpu::kRTX4090},
      {"4080 super", NvidiaGpu::kRTX4080Super},
      {"4080", NvidiaGpu::kRTX4080},
      {"4070 ti super", NvidiaGpu::kRTX4070TiSuper},
      {"4070 ti", NvidiaGpu::kRTX4070Ti},
      {"4070 super", NvidiaGpu::kRTX4070Super},
      {"4070", NvidiaGpu::kRTX4070},
      {"4060 ti", NvidiaGpu::kRTX4060Ti},
      {"4060", NvidiaGpu::kRTX4060},
      {"l40s", NvidiaGpu::kL40S},
      {"l40", NvidiaGpu::kL40},
      {"l20", NvidiaGpu::kL20},
      {"l4", NvidiaGpu::kL4},
      // Ampere
      {"3090 ti", NvidiaGpu::kRTX3090Ti},
      {"3090", NvidiaGpu::kRTX3090},
      {"3080 ti", NvidiaGpu::kRTX3080Ti},
      {"3080", NvidiaGpu::kRTX3080},
      {"3070 ti", NvidiaGpu::kRTX3070Ti},
      {"3070", NvidiaGpu::kRTX3070},
      {"3060 ti", NvidiaGpu::kRTX3060Ti},
      {"3060", NvidiaGpu::kRTX3060},
      {"3050", NvidiaGpu::kRTX3050},
      {"a800", NvidiaGpu::kA800},
      {"a100", NvidiaGpu::kA100},
      {"a40", NvidiaGpu::kA40},
      {"a30", NvidiaGpu::kA30},
      {"a16", NvidiaGpu::kA16},
      {"a10g", NvidiaGpu::kA10G},
      {"a10", NvidiaGpu::kA10G},
      {"a2", NvidiaGpu::kA2},
      // Turing Workstation / Gaming
      {"titan rtx", NvidiaGpu::kTitanRTX},
      {"rtx 8000", NvidiaGpu::kRTX8000Turing},
      {"rtx 6000", NvidiaGpu::kRTX6000Turing},
      {"rtx 5000", NvidiaGpu::kRTX5000Turing},
      {"rtx 4000", NvidiaGpu::kRTX4000Turing},
      {"2080 ti", NvidiaGpu::kRTX2080Ti},
      {"2080 super", NvidiaGpu::kRTX2080Super},
      {"2080", NvidiaGpu::kRTX2080},
      {"2070 super", NvidiaGpu::kRTX2070Super},
      {"2070", NvidiaGpu::kRTX2070},
      {"2060 super", NvidiaGpu::kRTX2060Super},
      {"2060", NvidiaGpu::kRTX2060},
      {"1660 ti", NvidiaGpu::kGTX1660Ti},
      {"1660 super", NvidiaGpu::kGTX1660Super},
      {"1660", NvidiaGpu::kGTX1660},
      {"1650 super", NvidiaGpu::kGTX1650Super},
      {"1650", NvidiaGpu::kGTX1650},
      {"t4", NvidiaGpu::kT4},
      // Volta
      {"v100", NvidiaGpu::kV100},
      {"titan v", NvidiaGpu::kTitanV},
      // Pascal
      {"titan xp", NvidiaGpu::kTitanXp},
      {"1080 ti", NvidiaGpu::kGTX1080Ti},
      {"1080", NvidiaGpu::kGTX1080},
      {"1070 ti", NvidiaGpu::kGTX1070Ti},
      {"1070", NvidiaGpu::kGTX1070},
      {"1060", NvidiaGpu::kGTX1060},
      {"1050 ti", NvidiaGpu::kGTX1050Ti},
      {"1050", NvidiaGpu::kGTX1050},
      {"p100", NvidiaGpu::kP100},
      {"p40", NvidiaGpu::kP40},
      {"p4", NvidiaGpu::kP4},
      // Maxwell
      {"titan x", NvidiaGpu::kTitanX},
      {"980 ti", NvidiaGpu::kGTX980Ti},
      {"980", NvidiaGpu::kGTX980},
      {"970", NvidiaGpu::kGTX970},
      {"960", NvidiaGpu::kGTX960},
      {"m40", NvidiaGpu::kM40},
      // Kepler
      {"780", NvidiaGpu::kGTX780},
      {"k80", NvidiaGpu::kK80},
      {"k40", NvidiaGpu::kK40},
      // Fermi
      {"580", NvidiaGpu::kGTX580},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return NvidiaGpu::kUnknown;
}

AMDArchitecture GetAMDArchitecture(const std::string& gpu_description) {
  // Order must be preserved
  const std::vector<std::pair<std::string, AMDArchitecture>> kMapping = {
      {"gcn-1", AMDArchitecture::kGcn1},   {"gcn-2", AMDArchitecture::kGcn2},
      {"gcn-3", AMDArchitecture::kGcn3},   {"gcn-4", AMDArchitecture::kGcn4},
      {"gcn-5", AMDArchitecture::kGcn5},   {"rdna-1", AMDArchitecture::kRdna1},
      {"rdna-2", AMDArchitecture::kRdna2}, {"rdna-3", AMDArchitecture::kRdna3},
      {"rdna-4", AMDArchitecture::kRdna4},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }

  return AMDArchitecture::kUnknown;
}

AMDGpu GetAMDGpu(const std::string& gpu_description) {
  // Order must be preserved
  const std::vector<std::pair<std::string, AMDGpu>> kMapping = {
      {"radeon pro vega ii", AMDGpu::kRadeonProVegaII},
      {"radeon vii", AMDGpu::kRadeonVII},
      {"radeon rx 5600", AMDGpu::kRadeonRX5600},
      {"radeon rx 5700", AMDGpu::kRadeonRX5700},
      {"radeon pro 5700", AMDGpu::kRadeonPro5700},
      {"radeon pro w5700", AMDGpu::kRadeonProW5700},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }

  return AMDGpu::kUnknown;
}

MaleoonInfo GetMaleoonInfo(const std::string& gpu_description) {
  const std::map<std::string, MaleoonInfo> kMapping = {
      {"maleoon 910", MaleoonInfo{.gpu = MaleoonGpu::kMaleoon910}},
      {"maleoon 920", MaleoonInfo{.gpu = MaleoonGpu::kMaleoon920}},
      {"maleoon 930", MaleoonInfo{.gpu = MaleoonGpu::kMaleoon930}},
  };

  MaleoonInfo maleoon_info;
  maleoon_info.gpu = MaleoonGpu::kUnknown;
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      maleoon_info = v.second;
      break;
    }
  }
  return maleoon_info;
}

}  // namespace

bool AdrenoInfo::IsAdreno1xx() const {
  return generation == AdrenoInfo::Generation::kGen1;
}

bool AdrenoInfo::IsAdreno2xx() const {
  return generation == AdrenoInfo::Generation::kGen2;
}

bool AdrenoInfo::IsAdreno3xx() const {
  return generation == AdrenoInfo::Generation::kGen3;
}

bool AdrenoInfo::IsAdreno4xx() const {
  return generation == AdrenoInfo::Generation::kGen4;
}

bool AdrenoInfo::IsAdreno5xx() const {
  return generation == AdrenoInfo::Generation::kGen5;
}

bool AdrenoInfo::IsAdreno6xx() const {
  return generation == AdrenoInfo::Generation::kGen6;
}

bool AdrenoInfo::IsAdreno7xx() const {
  return generation == AdrenoInfo::Generation::kGen7;
}

bool AdrenoInfo::IsAdreno8xx() const {
  return generation == AdrenoInfo::Generation::kGen8;
}

bool AdrenoInfo::IsLowEnd() const { return compute_units == 1; }

int AdrenoInfo::GetWaveSize(bool full_wave) const {
  if (generation <= AdrenoInfo::Generation::kGen3) {
    return full_wave ? 32 : 16;
  } else if (generation <= AdrenoInfo::Generation::kGen5) {
    return full_wave ? 64 : 32;
  } else if (generation <= AdrenoInfo::Generation::kGen7) {
    return full_wave ? 128 : 64;
  } else {
    return 64;
  }
}

AMDInfo::AMDInfo(const std::string& gpu_description)
    : architecture(GetAMDArchitecture(gpu_description)),
      amd_gpu(GetAMDGpu(gpu_description)) {}

bool AMDInfo::IsArchitectureOrNewer(AMDArchitecture arch) const {
  return architecture >= arch;
}

bool AMDInfo::IsNavi10() const {
  return amd_gpu == AMDGpu::kRadeonRX5600 || amd_gpu == AMDGpu::kRadeonRX5700 ||
         amd_gpu == AMDGpu::kRadeonPro5700 ||
         amd_gpu == AMDGpu::kRadeonProW5700;
}

bool AMDInfo::IsVega20() const {
  return amd_gpu == AMDGpu::kRadeonProVegaII || amd_gpu == AMDGpu::kRadeonVII;
}

AppleInfo::AppleInfo(const std::string& gpu_description) {
  const std::vector<std::pair<std::string, AppleGpu>> kMapping = {
      {"apple a7", AppleGpu::kA7},
      {"apple a8", AppleGpu::kA8},
      {"apple a8x", AppleGpu::kA8X},
      {"apple a9", AppleGpu::kA9},
      {"apple a9x", AppleGpu::kA9X},
      {"apple a10", AppleGpu::kA10},
      {"apple a10x", AppleGpu::kA10X},
      {"apple a11", AppleGpu::kA11},
      {"apple a12", AppleGpu::kA12},
      {"apple a12x", AppleGpu::kA12X},
      {"apple a12z", AppleGpu::kA12Z},
      {"apple a13", AppleGpu::kA13},
      {"apple a14", AppleGpu::kA14},
      {"apple a15", AppleGpu::kA15},
      {"apple a16", AppleGpu::kA16},
      {"apple a17 pro", AppleGpu::kA17Pro},
      {"apple a18", AppleGpu::kA18},
      {"apple a18 pro", AppleGpu::kA18Pro},
      {"apple a19", AppleGpu::kA19},
      {"apple a19 pro", AppleGpu::kA19Pro},
      {"apple a20 pro", AppleGpu::kA20Pro},
      {"apple m1", AppleGpu::kM1},
      {"apple m1 pro", AppleGpu::kM1Pro},
      {"apple m1 max", AppleGpu::kM1Max},
      {"apple m1 ultra", AppleGpu::kM1Ultra},
      {"apple m2", AppleGpu::kM2},
      {"apple m2 pro", AppleGpu::kM2Pro},
      {"apple m2 max", AppleGpu::kM2Max},
      {"apple m2 ultra", AppleGpu::kM2Ultra},
      {"apple m3", AppleGpu::kM3},
      {"apple m3 pro", AppleGpu::kM3Pro},
      {"apple m3 max", AppleGpu::kM3Max},
      {"apple m3 ultra", AppleGpu::kM3Ultra},
      {"apple m4", AppleGpu::kM4},
      {"apple m4 pro", AppleGpu::kM4Pro},
      {"apple m4 max", AppleGpu::kM4Max},
      {"apple m5", AppleGpu::kM5},
      {"apple m5 pro", AppleGpu::kM5Pro},
      {"apple m5 max", AppleGpu::kM5Max},
      {"apple m5 ultra", AppleGpu::kM5Ultra},
      {"apple m6", AppleGpu::kM6},
  };
  gpu_type = AppleGpu::kUnknown;
  std::string gpu_name = "";
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos &&
        v.first.size() > gpu_name.size()) {
      gpu_name = v.first;
      gpu_type = v.second;
    }
  }
  gpu_family = GetGpuFamily();
}

AppleInfo::Family AppleInfo::GetGpuFamily() const {
  if (gpu_type == AppleGpu::kA7) {
    return AppleInfo::Family::kApple1;
  } else if (gpu_type == AppleGpu::kA8 || gpu_type == AppleGpu::kA8X) {
    return AppleInfo::Family::kApple2;
  } else if (gpu_type == AppleGpu::kA9 || gpu_type == AppleGpu::kA9X ||
             gpu_type == AppleGpu::kA10 || gpu_type == AppleGpu::kA10X) {
    return AppleInfo::Family::kApple3;
  } else if (gpu_type == AppleGpu::kA11) {
    return AppleInfo::Family::kApple4;
  } else if (gpu_type == AppleGpu::kA12 || gpu_type == AppleGpu::kA12X ||
             gpu_type == AppleGpu::kA12Z) {
    return AppleInfo::Family::kApple5;
  } else if (gpu_type == AppleGpu::kA13) {
    return AppleInfo::Family::kApple6;
  } else if (gpu_type == AppleGpu::kA14 || IsM1Series()) {
    return AppleInfo::Family::kApple7;
  } else if (gpu_type == AppleGpu::kA15 || gpu_type == AppleGpu::kA16 ||
             IsM2Series()) {
    return AppleInfo::Family::kApple8;
  } else if (gpu_type == AppleGpu::kA17Pro || gpu_type == AppleGpu::kA18 ||
             gpu_type == AppleGpu::kA18Pro || IsM3Series() || IsM4Series()) {
    return AppleInfo::Family::kApple9;
  } else if (gpu_type == AppleGpu::kA19 || gpu_type == AppleGpu::kA19Pro ||
             gpu_type == AppleGpu::kA20Pro || IsM5Series() || IsM6Series()) {
    return AppleInfo::Family::kApple10;
  }
  return AppleInfo::Family::kApple1;
}

bool AppleInfo::IsFamilyApple1() const {
  return gpu_family == AppleInfo::Family::kApple1;
}

bool AppleInfo::IsFamilyApple2() const {
  return gpu_family == AppleInfo::Family::kApple2;
}

bool AppleInfo::IsFamilyApple3() const {
  return gpu_family == AppleInfo::Family::kApple3;
}

bool AppleInfo::IsFamilyApple4() const {
  return gpu_family == AppleInfo::Family::kApple4;
}

bool AppleInfo::IsFamilyApple5() const {
  return gpu_family == AppleInfo::Family::kApple5;
}

bool AppleInfo::IsFamilyApple6() const {
  return gpu_family == AppleInfo::Family::kApple6;
}

bool AppleInfo::IsFamilyApple7() const {
  return gpu_family == AppleInfo::Family::kApple7;
}

bool AppleInfo::IsFamilyApple8() const {
  return gpu_family == AppleInfo::Family::kApple8;
}

bool AppleInfo::IsFamilyApple9() const {
  return gpu_family == AppleInfo::Family::kApple9;
}

bool AppleInfo::IsFamilyApple10() const {
  return gpu_family == AppleInfo::Family::kApple10;
}

bool AppleInfo::IsFamilyOrLower(AppleInfo::Family family) const {
  return gpu_family <= family;
}

bool AppleInfo::IsLocalMemoryPreferredOverGlobal() const {
  return IsFamilyOrLower(AppleInfo::Family::kApple2);
}

bool AppleInfo::IsMSeries() const {
  return IsM1Series() || IsM2Series() || IsM3Series() || IsM4Series() ||
         IsM5Series() || IsM6Series();
}

bool AppleInfo::IsM1Series() const {
  return gpu_type == AppleGpu::kM1 || gpu_type == AppleGpu::kM1Pro ||
         gpu_type == AppleGpu::kM1Max || gpu_type == AppleGpu::kM1Ultra;
}

bool AppleInfo::IsM2Series() const {
  return gpu_type == AppleGpu::kM2 || gpu_type == AppleGpu::kM2Pro ||
         gpu_type == AppleGpu::kM2Max || gpu_type == AppleGpu::kM2Ultra;
}

bool AppleInfo::IsM3Series() const {
  return gpu_type == AppleGpu::kM3 || gpu_type == AppleGpu::kM3Pro ||
         gpu_type == AppleGpu::kM3Max || gpu_type == AppleGpu::kM3Ultra;
}

bool AppleInfo::IsM4Series() const {
  return gpu_type == AppleGpu::kM4 || gpu_type == AppleGpu::kM4Pro ||
         gpu_type == AppleGpu::kM4Max;
}

bool AppleInfo::IsM5Series() const {
  return gpu_type == AppleGpu::kM5 || gpu_type == AppleGpu::kM5Pro ||
         gpu_type == AppleGpu::kM5Max || gpu_type == AppleGpu::kM5Ultra;
}

bool AppleInfo::IsM6Series() const { return gpu_type == AppleGpu::kM6; }

bool AppleInfo::IsBionic() const {
  return gpu_family >= AppleInfo::Family::kApple4;
}

bool AppleInfo::IsSIMDMatMulSupported() const {
  return gpu_family >= AppleInfo::Family::kApple7;
}

bool AppleInfo::IsSIMDMatMulFp32Perf2x() const {
  return gpu_family >= AppleInfo::Family::kApple8 || IsM1Series();
}

bool AppleInfo::IsRoundToNearestSupported() const { return IsBionic(); }

int AppleInfo::GetComputeUnitsCount() const {
  switch (gpu_type) {
    case AppleGpu::kA7:
      return 4;
    case AppleGpu::kA8:
      return 4;
    case AppleGpu::kA8X:
      return 8;
    case AppleGpu::kA9:
      return 6;
    case AppleGpu::kA9X:
      return 12;
    case AppleGpu::kA10:
      return 6;
    case AppleGpu::kA10X:
      return 12;
    case AppleGpu::kA11:
      return 3;
    case AppleGpu::kA12:
      return 4;
    case AppleGpu::kA12X:
      return 7;
    case AppleGpu::kA12Z:
      return 8;
    case AppleGpu::kA13:
      return 4;
    case AppleGpu::kA14:
      return 4;
    // For some apple GPUs we can not receive exact CU count from name.
    // No official Metal API to receive this info.
    case AppleGpu::kA15:
      if (compute_units != -1) {
        return compute_units;
      }
      return 5;
    case AppleGpu::kA16:
      return 5;
    case AppleGpu::kA17Pro:
      return 6;
    case AppleGpu::kA18:
      return 5;
    case AppleGpu::kA18Pro:
      return 6;
    case AppleGpu::kA19:
      return 5;
    case AppleGpu::kA19Pro:
      return 6;
    case AppleGpu::kA20Pro:
      return 7;
    case AppleGpu::kM1:
      // approximate, can be 7 or 8
      return 8;
    case AppleGpu::kM1Pro:
      // approximate, can be 14 or 16
      return 16;
    case AppleGpu::kM1Max:
      // approximate, can be 24 or 32
      return 32;
    case AppleGpu::kM1Ultra:
      // approximate, 64 is max possible
      return 64;
    case AppleGpu::kM2:
      // approximate
      return 10;
    case AppleGpu::kM2Pro:
      // approximate
      return 19;
    case AppleGpu::kM2Max:
      // approximate
      return 38;
    case AppleGpu::kM2Ultra:
      // approximate
      return 76;
    case AppleGpu::kM3:
      // approximate
      return 10;
    case AppleGpu::kM3Pro:
      // approximate
      return 18;
    case AppleGpu::kM3Max:
      // approximate
      return 40;
    case AppleGpu::kM3Ultra:
      // approximate
      return 80;
    case AppleGpu::kM4:
      // approximate
      return 10;
    case AppleGpu::kM4Pro:
      // approximate
      return 20;
    case AppleGpu::kM4Max:
      // approximate
      return 40;
    case AppleGpu::kM5:
      // approximate
      return 10;
    case AppleGpu::kM5Pro:
      // approximate
      return 20;
    case AppleGpu::kM5Max:
      // approximate
      return 40;
    case AppleGpu::kM5Ultra:
      // approximate
      return 80;
    case AppleGpu::kM6:
      // approximate
      return 12;
    case AppleGpu::kUnknown:
      return 4;
  }
}

void AppleInfo::SetComputeUnits(int compute_units_count) {
  compute_units = compute_units_count;
}

int BroadcomInfo::GetApproximateComputeUnitsCount() const {
  if (gpu_version == BroadcomGpu::kVideoCore7) {
    return 12;
  }
  return 4;
}

bool BroadcomInfo::IsVideoCore7() const {
  return gpu_version == BroadcomGpu::kVideoCore7;
}

bool MaliInfo::IsMaliT6xx() const {
  return generation == MaliInfo::Gen::kMidgardV1;
}

bool MaliInfo::IsMaliT7xx() const {
  return generation == MaliInfo::Gen::kMidgardV2;
}

bool MaliInfo::IsMaliT8xx() const {
  return generation == MaliInfo::Gen::kMidgardV3;
}

bool MaliInfo::IsMidgard() const {
  return IsMaliT6xx() || IsMaliT7xx() || IsMaliT8xx();
}

bool MaliInfo::IsBifrostGen1() const {
  return generation == MaliInfo::Gen::kBifrostV1;
}

bool MaliInfo::IsBifrostGen2() const {
  return generation == MaliInfo::Gen::kBifrostV2;
}

bool MaliInfo::IsBifrostGen3() const {
  return generation == MaliInfo::Gen::kBifrostV3;
}

bool MaliInfo::IsBifrost() const {
  return IsBifrostGen1() || IsBifrostGen2() || IsBifrostGen3();
}

bool MaliInfo::IsValhallGen1() const {
  return generation == MaliInfo::Gen::kValhallV1;
}

bool MaliInfo::IsValhallGen2() const {
  return generation == MaliInfo::Gen::kValhallV2;
}

bool MaliInfo::IsValhallGen3() const {
  return generation == MaliInfo::Gen::kValhallV3;
}

bool MaliInfo::IsValhallGen4() const {
  return generation == MaliInfo::Gen::kValhallV4;
}

bool MaliInfo::IsValhall() const {
  return IsValhallGen1() || IsValhallGen2() || IsValhallGen3() ||
         IsValhallGen4();
}

bool MaliInfo::IsGen5() const { return generation == MaliInfo::Gen::kV5; }

bool MaliInfo::IsMaliG1() const {
  return gpu_version == MaliGpu::kG1Pro || gpu_version == MaliGpu::kG1Premium ||
         gpu_version == MaliGpu::kG1Ultra;
}

int MaliInfo::GetApproximateComputeUnitsCount() const {
  if (IsMidgard()) {
    // Mali Midgard can have 1-16 cores
    return 8;
  } else if (IsBifrost()) {
    // Mali Bifrost can have 1-32 cores
    return 16;
  } else if (IsValhall()) {
    if (gpu_version == MaliGpu::kG57) {
      return 6;  // Mali-G57 can have 1-6 cores
    } else if (gpu_version == MaliGpu::kG77) {
      return 16;  // Mali-G77 can have 7-16 cores
    } else if (gpu_version == MaliGpu::kG68) {
      return 6;  // Mali-G68 can have 4-6 cores
    } else if (gpu_version == MaliGpu::kG78) {
      return 16;  // Mali-G78 can have 7-24 cores
    } else if (gpu_version == MaliGpu::kG310 || gpu_version == MaliGpu::kG510 ||
               gpu_version == MaliGpu::kG610 || gpu_version == MaliGpu::kG615 ||
               gpu_version == MaliGpu::kG620) {
      return 6;  // Mali-G310/G510/G610/G615/G620 can have up to 6 cores
    } else if (gpu_version == MaliGpu::kG710 || gpu_version == MaliGpu::kG715 ||
               gpu_version == MaliGpu::kG720) {
      // Mali-G710/G715 can have 7–16 cores
      // Mali-G720 7-9 cores, Immortalis-G720 - 10-16 cores
      return 10;
    } else if (gpu_version == MaliGpu::kG925) {
      return 12;
    } else if (gpu_version == MaliGpu::kG1Pro) {
      return 2;
    } else if (gpu_version == MaliGpu::kG1Premium) {
      return 6;
    } else if (gpu_version == MaliGpu::kG1Ultra) {
      return 12;
    }
  }
  return 4;
}

PowerVRInfo::PowerVRInfo(const std::string& gpu_description)
    : gpu_version(GetPowerVRGpuVersion(gpu_description)) {}

bool PowerVRInfo::IsRogue() const { return gpu_version == PowerVRGpu::kRogue; }

bool PowerVRInfo::IsImgAxx() const {
  return gpu_version == PowerVRGpu::kAXE || gpu_version == PowerVRGpu::kAXM ||
         gpu_version == PowerVRGpu::kAXT;
}

bool PowerVRInfo::IsImgBxx() const {
  return gpu_version == PowerVRGpu::kBXE || gpu_version == PowerVRGpu::kBXM ||
         gpu_version == PowerVRGpu::kBXS || gpu_version == PowerVRGpu::kBXT;
}

bool PowerVRInfo::IsImgCxx() const { return gpu_version == PowerVRGpu::kCXT; }

bool PowerVRInfo::IsImgDxx() const { return gpu_version == PowerVRGpu::kDXT; }

NvidiaInfo::NvidiaInfo(const std::string& gpu_description) {
  std::string lowered = gpu_description;
  absl::AsciiStrToLower(&lowered);
  architecture = GetNvidiaArchitecture(lowered);
  gpu_type = GetNvidiaGpu(lowered);
}

bool NvidiaInfo::IsArchitectureOrNewer(NvidiaArchitecture arch) const {
  return architecture >= arch;
}

int NvidiaInfo::GetComputeUnitsCount() const {
  switch (gpu_type) {
    // Fermi
    case NvidiaGpu::kGTX580:
      return 16;
    // Kepler
    case NvidiaGpu::kGTX780:
      return 12;
    case NvidiaGpu::kK80:
      return 13;
    case NvidiaGpu::kK40:
      return 15;
    // Maxwell
    case NvidiaGpu::kGTX960:
      return 8;
    case NvidiaGpu::kGTX970:
      return 13;
    case NvidiaGpu::kGTX980:
      return 16;
    case NvidiaGpu::kGTX980Ti:
      return 22;
    case NvidiaGpu::kTitanX:
    case NvidiaGpu::kM40:
      return 24;
    // Pascal
    case NvidiaGpu::kGTX1050:
      return 5;
    case NvidiaGpu::kGTX1050Ti:
      return 6;
    case NvidiaGpu::kGTX1060:
      return 10;
    case NvidiaGpu::kGTX1070:
      return 15;
    case NvidiaGpu::kGTX1070Ti:
      return 19;
    case NvidiaGpu::kGTX1080:
    case NvidiaGpu::kP4:
      return 20;
    case NvidiaGpu::kGTX1080Ti:
      return 28;
    case NvidiaGpu::kTitanXp:
    case NvidiaGpu::kP40:
      return 30;
    case NvidiaGpu::kP100:
      return 56;
    // Volta
    case NvidiaGpu::kV100:
    case NvidiaGpu::kTitanV:
      return 80;
    // Turing
    case NvidiaGpu::kGTX1650:
      return 14;
    case NvidiaGpu::kGTX1650Super:
      return 20;
    case NvidiaGpu::kGTX1660:
    case NvidiaGpu::kGTX1660Super:
      return 22;
    case NvidiaGpu::kGTX1660Ti:
      return 24;
    case NvidiaGpu::kRTX2060:
      return 30;
    case NvidiaGpu::kRTX2060Super:
      return 34;
    case NvidiaGpu::kRTX2070:
    case NvidiaGpu::kRTX4000Turing:
      return 36;
    case NvidiaGpu::kRTX2070Super:
    case NvidiaGpu::kT4:
      return 40;
    case NvidiaGpu::kRTX2080:
      return 46;
    case NvidiaGpu::kRTX2080Super:
    case NvidiaGpu::kRTX5000Turing:
      return 48;
    case NvidiaGpu::kRTX2080Ti:
      return 68;
    case NvidiaGpu::kTitanRTX:
    case NvidiaGpu::kRTX6000Turing:
    case NvidiaGpu::kRTX8000Turing:
      return 72;
    // Ampere
    case NvidiaGpu::kA2:
    case NvidiaGpu::kA16:
      return 10;
    case NvidiaGpu::kRTX3050:
      return 20;
    case NvidiaGpu::kRTXA2000:
      return 26;
    case NvidiaGpu::kRTX3060:
      return 28;
    case NvidiaGpu::kRTX3060Ti:
      return 38;
    case NvidiaGpu::kRTX3070:
      return 46;
    case NvidiaGpu::kRTX3070Ti:
    case NvidiaGpu::kRTXA4000:
      return 48;
    case NvidiaGpu::kA30:
    case NvidiaGpu::kRTXA4500:
      return 56;
    case NvidiaGpu::kRTXA5000:
      return 64;
    case NvidiaGpu::kRTX3080:
      return 68;
    case NvidiaGpu::kRTX3080Ti:
    case NvidiaGpu::kA10G:
      return 80;
    case NvidiaGpu::kRTX3090:
      return 82;
    case NvidiaGpu::kRTX3090Ti:
    case NvidiaGpu::kA40:
    case NvidiaGpu::kRTXA6000:
      return 84;
    case NvidiaGpu::kA100:
    case NvidiaGpu::kA800:
      return 108;
    // Lovelace
    case NvidiaGpu::kRTX2000Ada:
      return 22;
    case NvidiaGpu::kRTX4060:
      return 24;
    case NvidiaGpu::kRTX4060Ti:
      return 34;
    case NvidiaGpu::kRTX3500Ada:
      return 40;
    case NvidiaGpu::kRTX4070:
      return 46;
    case NvidiaGpu::kRTX4000Ada:
      return 48;
    case NvidiaGpu::kRTX4070Super:
      return 56;
    case NvidiaGpu::kL4:
      return 58;
    case NvidiaGpu::kRTX4070Ti:
      return 60;
    case NvidiaGpu::kRTX4070TiSuper:
      return 66;
    case NvidiaGpu::kRTX4080:
      return 76;
    case NvidiaGpu::kRTX4080Super:
      return 80;
    case NvidiaGpu::kL20:
      return 92;
    case NvidiaGpu::kRTX5000Ada:
      return 100;
    case NvidiaGpu::kRTX4090:
      return 128;
    case NvidiaGpu::kL40S:
    case NvidiaGpu::kL40:
    case NvidiaGpu::kRTX6000Ada:
      return 142;
    // Hopper
    case NvidiaGpu::kH100:
    case NvidiaGpu::kH200:
    case NvidiaGpu::kH800:
    case NvidiaGpu::kGH200:
      return 132;
    // Blackwell
    case NvidiaGpu::kRTX5060:
      return 30;
    case NvidiaGpu::kRTX5060Ti:
      return 36;
    case NvidiaGpu::kRTX5070:
      return 48;
    case NvidiaGpu::kRTX5070Ti:
      return 70;
    case NvidiaGpu::kRTX5080:
      return 84;
    case NvidiaGpu::kB100:
      return 132;
    case NvidiaGpu::kB200:
    case NvidiaGpu::kB800:
    case NvidiaGpu::kGB200:
      return 148;
    case NvidiaGpu::kRTX5090:
      return 170;
    case NvidiaGpu::kUnknown:
      break;
  }

  // If the exact GPU model was not identified, fallback to a representative
  // SM (Streaming Multiprocessor) count for a typical flagship/server GPU
  // belonging to the detected architecture generation.
  switch (architecture) {
    case NvidiaArchitecture::kFermi:
      return 16;  // GF100/GF110 (e.g., GTX 580).
    case NvidiaArchitecture::kKepler:
      return 15;  // GK110 (e.g., GTX 780 Ti / K40).
    case NvidiaArchitecture::kMaxwell:
      return 24;  // GM200 (e.g., Titan X / GTX 980 Ti).
    case NvidiaArchitecture::kPascal:
      return 28;  // GP102 (e.g., GTX 1080 Ti).
    case NvidiaArchitecture::kVolta:
      return 80;  // GV100 (e.g., V100 / Titan V).
    case NvidiaArchitecture::kTuring:
      return 40;  // TU104 (e.g., T4), ubiquitous in cloud inference.
    case NvidiaArchitecture::kAmpere:
      return 80;  // GA102 (e.g., A10G / RTX 3080 12GB).
    case NvidiaArchitecture::kLovelace:
      return 128;  // AD102 (e.g., RTX 4090).
    case NvidiaArchitecture::kHopper:
      return 132;  // GH100 (e.g., H100 SXM / H200).
    case NvidiaArchitecture::kBlackwell:
      return 148;  // GB200/B200 (per die SM count).
    default:
      // Default fallback of 16 SMs provides a safe, moderate baseline for
      // unknown NVIDIA GPU architectures.
      return 16;
  }
}

void GetGpuInfoFromDeviceDescription(const std::string& gpu_description,
                                     GpuApi gpu_api, GpuInfo* gpu_info) {
  gpu_info->gpu_api = gpu_api;
  std::string lowered = gpu_description;
  absl::AsciiStrToLower(&lowered);
  gpu_info->vendor = GetGpuVendor(lowered);

  if (gpu_info->IsAdreno()) {
    gpu_info->adreno_info = GetAdrenoInfo(lowered);
  } else if (gpu_info->IsApple()) {
    gpu_info->apple_info = AppleInfo(lowered);
    gpu_info->supported_wave_sizes = {32};
  } else if (gpu_info->IsBroadcom()) {
    gpu_info->broadcom_info = GetBroadcomInfo(lowered);
  } else if (gpu_info->IsMali()) {
    gpu_info->mali_info = GetMaliInfo(lowered);
  } else if (gpu_info->IsPowerVR()) {
    gpu_info->powervr_info = PowerVRInfo(lowered);
    const int wave_size =
        gpu_info->powervr_info.gpu_version <= PowerVRGpu::kRogue ? 32 : 128;
    gpu_info->supported_wave_sizes = {wave_size};
  } else if (gpu_info->IsIntel()) {
    gpu_info->intel_info = IntelInfo(lowered);
  } else if (gpu_info->IsNvidia()) {
    gpu_info->nvidia_info = NvidiaInfo(lowered);
    gpu_info->supported_wave_sizes = {32};
  } else if (gpu_info->IsAMD()) {
    gpu_info->amd_info = AMDInfo(lowered);
  } else if (gpu_info->IsMaleoon()) {
    gpu_info->maleoon_info = GetMaleoonInfo(lowered);
    gpu_info->supported_wave_sizes = {32, 64};
  }
}

std::string OpenClVersionToString(OpenClVersion version) {
  switch (version) {
    case OpenClVersion::kCl1_0:
      return "1.0";
    case OpenClVersion::kCl1_1:
      return "1.1";
    case OpenClVersion::kCl1_2:
      return "1.2";
    case OpenClVersion::kCl2_0:
      return "2.0";
    case OpenClVersion::kCl2_1:
      return "2.1";
    case OpenClVersion::kCl2_2:
      return "2.2";
    case OpenClVersion::kCl3_0:
      return "3.0";
    default:
      return "Unknown OpenCL version";
  }
}

bool OpenGlInfo::SupportsExplicitFp16() const {
  bool supports_f16_alu = false;
  bool supports_f16_storage = false;
  for (const auto& ext : extensions) {
    if (ext == "GL_EXT_shader_explicit_arithmetic_types_float16") {
      supports_f16_alu = true;
    }
    if (ext == "GL_EXT_shader_16bit_storage") {
      supports_f16_storage = true;
    }
  }
  return supports_f16_alu && supports_f16_storage;
}

bool OpenGlInfo::IsApiOpenGl31OrAbove() const {
  return (major_version == 3 && minor_version >= 1) || major_version > 3;
}

bool OpenGlInfo::IsApiOpenGl32OrAbove() const {
  return (major_version == 3 && minor_version >= 2) || major_version > 3;
}

bool VulkanInfo::SupportsExplicitFp16() const {
  bool supports_f16_alu = false;
  bool supports_f16_storage = false;
  for (const auto& ext : extensions) {
    if (ext == "VK_KHR_shader_float16_int8") {
      supports_f16_alu = true;
    }
    if (ext == "VK_KHR_16bit_storage") {
      supports_f16_storage = true;
    }
  }
  return supports_f16_alu && supports_f16_storage;
}

bool OpenClInfo::SupportedImage2dTypes::SupportsImage2D(DataType data_type,
                                                        int channels) const {
  if (channels == 1) {
    return r_layout.find(data_type) != r_layout.end();
  } else if (channels == 2) {
    return rg_layout.find(data_type) != rg_layout.end();
  } else if (channels == 3) {
    return rgb_layout.find(data_type) != rgb_layout.end();
  } else if (channels == 4) {
    return rgba_layout.find(data_type) != rgba_layout.end();
  } else {
    return false;
  }
}

bool OpenClInfo::IsImage2dFromBufferSupported() const {
  if (image_pitch_alignment == 0) {
    return false;
  }
  if (image_base_address_alignment == 0) {
    return false;
  }
  if (cl_version == OpenClVersion::kCl2_0 ||
      cl_version == OpenClVersion::kCl2_1 ||
      cl_version == OpenClVersion::kCl2_2) {
    return true;
  }
  for (const auto& ext : extensions) {
    if (ext == "cl_khr_image2d_from_buffer") {
      return true;
    }
  }
  return false;
}

bool MetalInfo::IsSIMDMatMulSupported() const {
  return IsMslVersionEqualOrHigher(2, 3);
}

bool MetalInfo::IsMslVersionEqualOrHigher(int major, int minor) const {
  const std::map<MetalLanguageVersion, std::pair<int, int>> kMapping = {
      {MetalLanguageVersion::kUnknown, {1, 0}},
      {MetalLanguageVersion::kMetal1_0, {1, 0}},
      {MetalLanguageVersion::kMetal1_1, {1, 1}},
      {MetalLanguageVersion::kMetal1_2, {1, 2}},
      {MetalLanguageVersion::kMetal2_0, {2, 0}},
      {MetalLanguageVersion::kMetal2_1, {2, 1}},
      {MetalLanguageVersion::kMetal2_2, {2, 2}},
      {MetalLanguageVersion::kMetal2_3, {2, 3}},
      {MetalLanguageVersion::kMetal2_4, {2, 4}},
      {MetalLanguageVersion::kMetal3_0, {3, 0}},
      {MetalLanguageVersion::kMetal3_1, {3, 1}},
      {MetalLanguageVersion::kMetal3_2, {3, 2}},
      {MetalLanguageVersion::kMetal4_0, {4, 0}}};
  auto version = kMapping.at(language_version);
  if (version.first > major) {
    return true;
  } else if (version.first == major && version.second >= minor) {
    return true;
  } else {
    return false;
  }
}

bool MetalInfo::IsNativeBfloatSupported() const {
  return IsMslVersionEqualOrHigher(3, 1) && !is_simulator;
}

IntelInfo::IntelInfo(const std::string& gpu_description)
    : generation(GetIntelGeneration(gpu_description)) {}

bool IntelInfo::IsGenerationOrNewer(IntelGeneration gen) const {
  return generation >= gen;
}

bool GpuInfo::IsAdreno() const { return vendor == GpuVendor::kQualcomm; }

bool GpuInfo::IsApple() const { return vendor == GpuVendor::kApple; }

bool GpuInfo::IsBroadcom() const { return vendor == GpuVendor::kBroadcom; }

bool GpuInfo::IsLlvmPipe() const { return vendor == GpuVendor::kLlvmPipe; }

bool GpuInfo::IsMali() const { return vendor == GpuVendor::kMali; }

bool GpuInfo::IsPowerVR() const { return vendor == GpuVendor::kPowerVR; }

bool GpuInfo::IsNvidia() const { return vendor == GpuVendor::kNvidia; }

bool GpuInfo::IsAMD() const { return vendor == GpuVendor::kAMD; }

bool GpuInfo::IsIntel() const { return vendor == GpuVendor::kIntel; }

bool GpuInfo::IsMaleoon() const { return vendor == GpuVendor::kHuawei; }

bool GpuInfo::IsRoundToNearestSupported() const {
  if (IsApiOpenCl()) {
    return opencl_info.supports_fp16_rtn || opencl_info.supports_fp32_rtn;
  }
  if (IsApple()) {
    return apple_info.IsRoundToNearestSupported();
  }
  if (IsAdreno()) {
    if (adreno_info.IsAdreno1xx() || adreno_info.IsAdreno2xx() ||
        adreno_info.IsAdreno3xx()) {
      return false;
    }
  }
  if (IsPowerVR()) {
    return false;
  }
  return true;
}

bool GpuInfo::IsDotPreferred() const {
  return IsApple() || (IsPowerVR() && powervr_info.IsImgBxx());
}

bool GpuInfo::SupportsFP16() const {
  if (IsApiOpenCl()) {
    return opencl_info.supports_fp16;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.supports_fp16;
  }
  return true;
}

bool GpuInfo::SupportsTextureArray() const {
  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    return opencl_info.cl_version >= OpenClVersion::kCl1_2;
  }
  return true;
}

bool GpuInfo::SupportsImageBuffer() const {
  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    return opencl_info.cl_version >= OpenClVersion::kCl1_2;
  }
  if (IsApiWebGpu()) {
    return false;
  }
  if (IsApiMetal()) {
    return !metal_info.is_simulator;
  }
  return true;
}

bool GpuInfo::SupportsImage3D() const {
  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    if (IsMali() && mali_info.IsMidgard()) {
      // On Mali T880 read_imageh doesn't compile with image3d_t
      return false;
    }
    return opencl_info.supports_image3d_writes;
  }
  return true;
}

bool GpuInfo::SupportsImages() const {
  if (IsApiOpenCl()) {
    return opencl_info.supports_images;
  }
  return true;
}

bool GpuInfo::SupportsPointersInKernels() const {
  return IsApiOpenCl() || IsApiMetal();
}

bool GpuInfo::SupportsZeroClampForImageBuffer() const {
  if (IsApiMetal() || IsApiOpenCl()) {
    return true;
  } else {
    return false;
  }
}

bool GpuInfo::SupportsZeroClampForImages() const {
  if (IsApiMetal()) {
    return true;
  } else if (IsApiOpenCl()) {
    return true;
  } else if (IsApiVulkan()) {
    return false;
  } else if (IsApiOpenGl()) {
    return false;
  } else {
    return false;
  }
}

bool GpuInfo::SupportsAcceleratedDp4a() const {
  // Metal does not have a builtin function for dp4a.
  if (IsApiMetal()) {
    return false;
  }
#if defined(__APPLE__)
  // WebGPU will use Metal on Apple devices.
  if (IsApiWebGpu()) {
    return false;
  }
#endif
  // TODO(b/367828817): We disable dp4a on web for now, until correctness issues
  // are resolved.
#ifndef __EMSCRIPTEN__
  if (IsIntel()) {
    return intel_info.IsGenerationOrNewer(IntelGeneration::kGen12);
  } else if (IsNvidia()) {
    return nvidia_info.IsArchitectureOrNewer(NvidiaArchitecture::kPascal);
  } else if (IsAMD()) {
    return amd_info.IsVega20() ||
           (amd_info.IsArchitectureOrNewer(AMDArchitecture::kRdna1) &&
            !amd_info.IsNavi10());
  } else if (IsMali()) {
    return mali_info.IsValhall();
  } else if (IsBroadcom()) {
    return broadcom_info.IsVideoCore7();
  }
#endif  // !__EMSCRIPTEN__
  return false;
}

bool GpuInfo::IsWaveSizeEqualTo32() const {
  return supported_wave_sizes.size() == 1 && supported_wave_sizes[0] == 32;
}

bool GpuInfo::SupportsExtension(const std::string& extension) const {
  const std::vector<std::string>* extensions = nullptr;
  if (IsApiMetal() && extension == "ucl_wave_memory") {
    return metal_info.IsMslVersionEqualOrHigher(2);
  }
  if (IsApiWebGpu() && extension == "ucl_wave_memory" && !IsAdreno()) {
    return webgpu_info.supports_subgroups;
  }
  if (IsApiOpenGl()) {
    extensions = &opengl_info.extensions;
  } else if (IsApiVulkan()) {
    extensions = &vulkan_info.extensions;
  } else if (IsApiOpenCl()) {
    extensions = &opencl_info.extensions;
  }
  if (!extensions) {
    return false;
  }
  for (const auto& ext : *extensions) {
    if (ext == extension) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::SupportsSubGroupWithSize(int sub_group_size) const {
  for (auto subgroup_size : supported_wave_sizes) {
    if (sub_group_size == subgroup_size) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::SupportsImage2D(DataType data_type, int channels) const {
  if (IsApiOpenCl()) {
    return opencl_info.supported_images_2d.SupportsImage2D(data_type, channels);
  }
  if (IsApiWebGpu()) {
    return channels == 1 || channels == 2 || channels == 4;
  }
  if (IsApiMetal()) {
    return channels == 1 || channels == 2 || channels == 4;
  }
  return false;
}

int GpuInfo::GetComputeUnitsCount() const {
  if (IsApiOpenCl()) {
    return opencl_info.compute_units_count;
  }
  if (IsApiCuda()) {
    return cuda_info.multiprocessor_count;
  }
  if (IsApple()) {
    return apple_info.GetComputeUnitsCount();
  }
  if (IsNvidia()) {
    return nvidia_info.GetComputeUnitsCount();
  }
  if (IsAMD()) {
    if (amd_info.GetComputeUnitsCount() != 0) {
      return amd_info.GetComputeUnitsCount();
    } else {
      // approximate number
      return 16;
    }
  }
  if (IsAdreno()) {
    return adreno_info.compute_units;
  }
  if (IsMali()) {
    return mali_info.GetApproximateComputeUnitsCount();
  }
  if (IsBroadcom()) {
    return broadcom_info.GetApproximateComputeUnitsCount();
  }
  return 4;
}

int GpuInfo::GetMaxWorkGroupSizeForX() const {
  int size = 256;
  if (IsApiOpenGl()) {
    size = opengl_info.max_compute_work_group_size_x;
  }
  if (IsApiVulkan()) {
    size = vulkan_info.max_compute_work_group_size_x;
  }
  if (IsApiOpenCl()) {
    size = opencl_info.max_work_group_size_x;
  }
  if (IsApiMetal()) {
    size = metal_info.max_work_group_size_x;
  }
  if (IsApiWebGpu()) {
    size = webgpu_info.max_compute_workgroup_size_x;
  }
  // There is a possible driver bug where the work group size for a single
  // dimension is reported as more than the maximum total size.
  return std::min(GetMaxWorkGroupTotalSize(), size);
}

int GpuInfo::GetMaxWorkGroupSizeForY() const {
  int size = 256;
  if (IsApiOpenGl()) {
    size = opengl_info.max_compute_work_group_size_y;
  }
  if (IsApiVulkan()) {
    size = vulkan_info.max_compute_work_group_size_y;
  }
  if (IsApiOpenCl()) {
    size = opencl_info.max_work_group_size_y;
  }
  if (IsApiMetal()) {
    size = metal_info.max_work_group_size_y;
  }
  if (IsApiWebGpu()) {
    size = webgpu_info.max_compute_workgroup_size_y;
  }
  // There is a possible driver bug where the work group size for a single
  // dimension is reported as more than the maximum total size.
  return std::min(GetMaxWorkGroupTotalSize(), size);
}

int GpuInfo::GetMaxWorkGroupSizeForZ() const {
  int size = 64;
  if (IsApiOpenGl()) {
    size = opengl_info.max_compute_work_group_size_z;
  }
  if (IsApiVulkan()) {
    size = vulkan_info.max_compute_work_group_size_z;
  }
  if (IsApiOpenCl()) {
    size = opencl_info.max_work_group_size_z;
  }
  if (IsApiMetal()) {
    size = metal_info.max_work_group_size_z;
  }
  if (IsApiWebGpu()) {
    size = webgpu_info.max_compute_workgroup_size_z;
  }
  // There is a possible driver bug where the work group size for a single
  // dimension is reported as more than the maximum total size.
  return std::min(GetMaxWorkGroupTotalSize(), size);
}

int GpuInfo::GetMaxWorkGroupTotalSize() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_work_group_invocations;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_invocations;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_total_size;
  }
  if (IsApiMetal()) {
    int max_size = metal_info.max_work_group_size_x;
    max_size = std::max(max_size, metal_info.max_work_group_size_y);
    max_size = std::max(max_size, metal_info.max_work_group_size_z);
    return max_size;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_compute_invocations_per_workgroup;
  }
  return 256;
}

uint32_t GpuInfo::GetMaxWorkGroupsCountForX() const {
  if (IsApiMetal() || IsApiCuda() || IsApiOpenCl()) {
    return 2147483647u;  // 2^31 - 1
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_compute_workgroups_per_dimension;
  }
  return 65535u;  // 2^16 - 1
}

uint32_t GpuInfo::GetMaxWorkGroupsCountForY() const {
  if (IsApiMetal() || IsApiOpenCl()) {
    return 2147483647u;  // 2^31 - 1
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_compute_workgroups_per_dimension;
  }
  return 65535u;  // 2^16 - 1
}

uint32_t GpuInfo::GetMaxWorkGroupsCountForZ() const {
  if (IsApiMetal() || IsApiOpenCl()) {
    return 2147483647u;  // 2^31 - 1
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_compute_workgroups_per_dimension;
  }
  return 65535u;  // 2^16 - 1
}

uint64_t GpuInfo::GetMaxImage2DWidth() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_texture_size;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_2d;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image2d_max_width;
  }
  if (IsApiMetal()) {
    return metal_info.image2d_max_width;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_texture_dimension_2d;
  }
  return 2048;
}

uint64_t GpuInfo::GetMaxImage2DHeight() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_texture_size;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_2d;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image2d_max_height;
  }
  if (IsApiMetal()) {
    return metal_info.image2d_max_height;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_texture_dimension_2d;
  }
  return 2048;
}

uint64_t GpuInfo::GetMaxImage2DArrayLayers() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_array_texture_layers;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_array_layers;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image_array_max_layers;
  }
  if (IsApiMetal()) {
    return metal_info.image_array_max_layers;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_texture_array_layers;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DWidth() const {
  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_width;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_width;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  } else if (IsApiWebGpu()) {
    return webgpu_info.max_texture_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DHeight() const {
  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_height;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_height;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  } else if (IsApiWebGpu()) {
    return webgpu_info.max_texture_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DDepth() const {
  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_depth;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_depth;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  } else if (IsApiWebGpu()) {
    return webgpu_info.max_texture_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxBufferSize() const {
  if (IsApiOpenCl()) {
    return opencl_info.buffer_max_size;
  } else if (IsApiMetal()) {
    return metal_info.buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_storage_buffer_range;
  } else if (IsApiWebGpu()) {
    return std::min(webgpu_info.max_buffer_size,
                    webgpu_info.max_storage_buffer_binding_size);
  } else if (IsApiOpenGl()) {
    return opengl_info.max_shader_storage_block_size;
  }
  return 128 * 1024 * 1024;
}

uint64_t GpuInfo::GetMaxMemoryAllocationSize() const {
  if (IsApiOpenCl()) {
    return opencl_info.max_allocation_size;
  } else if (IsApiMetal()) {
    return metal_info.buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_storage_buffer_range;
  } else if (IsApiWebGpu()) {
    return webgpu_info.max_buffer_size;
  } else if (IsApiOpenGl()) {
    return opengl_info.max_shader_storage_block_size;
  }
  return 128 * 1024 * 1024;
}

uint64_t GpuInfo::GetMaxImageBufferWidth() const {
  if (IsApiOpenCl()) {
    return opencl_info.image_buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_texel_buffer_elements;
  }
  return 64 * 1024;
}

int GpuInfo::GetMaxImageArguments() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_image_bindings;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_per_stage_descriptor_storage_images;
  }
  if (IsApiMetal()) {
    return 32;
  }
  if (IsApiOpenCl()) {
    return 128;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_storage_textures_per_shader_stage;
  }
  return 1;
}

int GpuInfo::GetMaxTextureArguments() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_texture_bindings;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_per_stage_descriptor_sampled_images;
  }
  if (IsApiMetal()) {
    return 32;
  }
  if (IsApiOpenCl()) {
    return 128;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_sampled_textures_per_shader_stage;
  }
  return 1;
}

int GpuInfo::GetMaxBufferArguments() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_ssbo_bindings;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_per_stage_descriptor_storage_buffers;
  }
  if (IsApiMetal()) {
    return 32;
  }
  if (IsApiOpenCl()) {
    return 128;
  }
  if (IsApiWebGpu()) {
    return webgpu_info.max_storage_buffers_per_shader_stage;
  }
  return 1;
}

bool GpuInfo::SupportsWaveMatMulOp(
    const WaveMatMulOpDescriptor& op_descriptor) const {
  return std::find(wave_mat_mul_ops.begin(), wave_mat_mul_ops.end(),
                   op_descriptor) != wave_mat_mul_ops.end();
}

bool GpuInfo::IsApiOpenGl() const { return gpu_api == GpuApi::kOpenGl; }

bool GpuInfo::IsApiOpenGl31OrAbove() const {
  if (!IsApiOpenGl()) {
    return false;
  }
  return opengl_info.IsApiOpenGl31OrAbove();
}

bool GpuInfo::IsApiVulkan() const { return gpu_api == GpuApi::kVulkan; }

bool GpuInfo::IsApiMetal() const { return gpu_api == GpuApi::kMetal; }

bool GpuInfo::IsApiOpenCl() const { return gpu_api == GpuApi::kOpenCl; }

bool GpuInfo::IsApiWebGpu() const { return gpu_api == GpuApi::kWebGpu; }

bool GpuInfo::IsApiCuda() const { return gpu_api == GpuApi::kCuda; }

bool GpuInfo::IsGlsl() const { return IsApiOpenGl() || IsApiVulkan(); }

bool GpuInfo::IsGlslSupportsExplicitFp16() const {
  if (IsApiOpenGl() && opengl_info.SupportsExplicitFp16()) {
    return true;
  }
  if (IsApiVulkan() && vulkan_info.SupportsExplicitFp16()) {
    return true;
  }
  return false;
}

bool GpuInfo::IsCL11OrHigher() const {
  if (!IsApiOpenCl()) {
    return false;
  }
  return opencl_info.cl_version != OpenClVersion::kCl1_0;
}

bool GpuInfo::IsCL20OrHigher() const {
  if (!IsApiOpenCl()) {
    return false;
  }
  return opencl_info.cl_version != OpenClVersion::kCl1_0 &&
         opencl_info.cl_version != OpenClVersion::kCl1_1 &&
         opencl_info.cl_version != OpenClVersion::kCl1_2;
}

bool GpuInfo::IsCL30OrHigher() const {
  if (!IsApiOpenCl()) {
    return false;
  }
  return IsCL20OrHigher() && opencl_info.cl_version != OpenClVersion::kCl2_0 &&
         opencl_info.cl_version != OpenClVersion::kCl2_1 &&
         opencl_info.cl_version != OpenClVersion::kCl2_2;
}

bool IsValidWorkgroup(const GpuInfo& gpu_info, const int3& wg_size) {
  return wg_size.x <= gpu_info.GetMaxWorkGroupSizeForX() &&
         wg_size.y <= gpu_info.GetMaxWorkGroupSizeForY() &&
         wg_size.z <= gpu_info.GetMaxWorkGroupSizeForZ() &&
         wg_size.x * wg_size.y * wg_size.z <=
             gpu_info.GetMaxWorkGroupTotalSize();
}

std::string ToString(const OpenClInfo::ClKhrIntegerDotProductInfo& int8_info) {
  std::string result;
  if (int8_info.suppports_4x8bit_packed) {
    result += "suppports_4x8bit_packed\n";
    if (int8_info.acceleration_info_4x8bit_packed.signed_accelerated) {
      result += "  4x8bit_packed_signed_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit_packed.unsigned_accelerated) {
      result += "  4x8bit_packed_unsigned_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit_packed
            .mixed_signedness_accelerated) {
      result += "  4x8bit_packed_mixed_signedness_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit_packed
            .accumulating_saturating_signed_accelerated) {
      result += "  4x8bit_packed_accumulating_saturating_signed_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit_packed
            .accumulating_saturating_unsigned_accelerated) {
      result +=
          "  4x8bit_packed_accumulating_saturating_unsigned_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit_packed
            .accumulating_saturating_mixed_signedness_accelerated) {
      result +=
          "  "
          "4x8bit_packed_accumulating_saturating_mixed_signedness_"
          "accelerated\n";
    }
  }
  if (int8_info.suppports_4x8bit) {
    result += "suppports_4x8bit\n";
    if (int8_info.acceleration_info_4x8bit.signed_accelerated) {
      result += "  4x8bit_signed_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit.unsigned_accelerated) {
      result += "  4x8bit_unsigned_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit.mixed_signedness_accelerated) {
      result += "  4x8bit_mixed_signedness_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit
            .accumulating_saturating_signed_accelerated) {
      result += "  4x8bit_accumulating_saturating_signed_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit
            .accumulating_saturating_unsigned_accelerated) {
      result += "  4x8bit_accumulating_saturating_unsigned_accelerated\n";
    }
    if (int8_info.acceleration_info_4x8bit
            .accumulating_saturating_mixed_signedness_accelerated) {
      result +=
          "  4x8bit_accumulating_saturating_mixed_signedness_accelerated\n";
    }
  }
  return result;
}

std::string ToString(const WaveMatMulOpDescriptor& op_descriptor) {
  return absl::Substitute(
      "$3($0x$2) += $4($0x$1) * $5($1x$2), MxKxN($0x$1x$2)",
      op_descriptor.m_size, op_descriptor.k_size, op_descriptor.n_size,
      ToString(op_descriptor.result_type), ToString(op_descriptor.left_type),
      ToString(op_descriptor.right_type));
}

}  // namespace ml_drift
