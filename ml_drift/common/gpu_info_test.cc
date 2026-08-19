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

#include <string>

#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"

namespace ml_drift {
namespace {

TEST(NvidiaInfoTest, TestComputeUnitsCount) {
  // Fermi
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 580").GetComputeUnitsCount(), 16);

  // Kepler
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 780").GetComputeUnitsCount(), 12);
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla K80").GetComputeUnitsCount(), 13);
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla K40").GetComputeUnitsCount(), 15);

  // Maxwell
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 960").GetComputeUnitsCount(), 8);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 970").GetComputeUnitsCount(), 13);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 980").GetComputeUnitsCount(), 16);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 980 Ti").GetComputeUnitsCount(), 22);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX Titan X").GetComputeUnitsCount(),
            24);
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla M40").GetComputeUnitsCount(), 24);

  // Pascal
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1050").GetComputeUnitsCount(), 5);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1050 Ti").GetComputeUnitsCount(), 6);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1060").GetComputeUnitsCount(), 10);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1070").GetComputeUnitsCount(), 15);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1070 Ti").GetComputeUnitsCount(),
            19);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1080").GetComputeUnitsCount(), 20);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1080 Ti").GetComputeUnitsCount(),
            28);
  EXPECT_EQ(NvidiaInfo("NVIDIA Titan Xp").GetComputeUnitsCount(), 30);
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla P4").GetComputeUnitsCount(), 20);
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla P40").GetComputeUnitsCount(), 30);
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla P100").GetComputeUnitsCount(), 56);

  // Volta
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla V100-SXM2-16GB").GetComputeUnitsCount(),
            80);
  EXPECT_EQ(NvidiaInfo("NVIDIA Titan V").GetComputeUnitsCount(), 80);

  // Turing
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1650").GetComputeUnitsCount(), 14);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1650 Super").GetComputeUnitsCount(),
            20);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1660").GetComputeUnitsCount(), 22);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1660 Super").GetComputeUnitsCount(),
            22);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce GTX 1660 Ti").GetComputeUnitsCount(),
            24);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 2060").GetComputeUnitsCount(), 30);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 2060 Super").GetComputeUnitsCount(),
            34);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 2070").GetComputeUnitsCount(), 36);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 2070 Super").GetComputeUnitsCount(),
            40);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 2080").GetComputeUnitsCount(), 46);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 2080 Super").GetComputeUnitsCount(),
            48);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 2080 Ti").GetComputeUnitsCount(),
            68);
  EXPECT_EQ(NvidiaInfo("NVIDIA Titan RTX").GetComputeUnitsCount(), 72);
  EXPECT_EQ(NvidiaInfo("NVIDIA Tesla T4").GetComputeUnitsCount(), 40);
  EXPECT_EQ(NvidiaInfo("NVIDIA Quadro RTX 4000").GetComputeUnitsCount(), 36);
  EXPECT_EQ(NvidiaInfo("NVIDIA Quadro RTX 5000").GetComputeUnitsCount(), 48);
  EXPECT_EQ(NvidiaInfo("NVIDIA Quadro RTX 6000").GetComputeUnitsCount(), 72);
  EXPECT_EQ(NvidiaInfo("NVIDIA Quadro RTX 8000").GetComputeUnitsCount(), 72);

  // Ampere
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3050").GetComputeUnitsCount(), 20);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3060").GetComputeUnitsCount(), 28);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3060 Ti").GetComputeUnitsCount(),
            38);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3070").GetComputeUnitsCount(), 46);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3070 Ti").GetComputeUnitsCount(),
            48);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3080").GetComputeUnitsCount(), 68);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3080 Ti").GetComputeUnitsCount(),
            80);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3090").GetComputeUnitsCount(), 82);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 3090 Ti").GetComputeUnitsCount(),
            84);
  EXPECT_EQ(NvidiaInfo("NVIDIA A2").GetComputeUnitsCount(), 10);
  EXPECT_EQ(NvidiaInfo("NVIDIA A16").GetComputeUnitsCount(), 10);
  EXPECT_EQ(NvidiaInfo("NVIDIA A10G").GetComputeUnitsCount(), 80);
  EXPECT_EQ(NvidiaInfo("NVIDIA A30").GetComputeUnitsCount(), 56);
  EXPECT_EQ(NvidiaInfo("NVIDIA A40").GetComputeUnitsCount(), 84);
  EXPECT_EQ(NvidiaInfo("NVIDIA A100-SXM4-40GB").GetComputeUnitsCount(), 108);
  EXPECT_EQ(NvidiaInfo("NVIDIA A800").GetComputeUnitsCount(), 108);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX A2000").GetComputeUnitsCount(), 26);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX A4000").GetComputeUnitsCount(), 48);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX A4500").GetComputeUnitsCount(), 56);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX A5000").GetComputeUnitsCount(), 64);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX A6000").GetComputeUnitsCount(), 84);

  // Lovelace / Ada Lovelace
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4060").GetComputeUnitsCount(), 24);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4060 Ti").GetComputeUnitsCount(),
            34);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4070").GetComputeUnitsCount(), 46);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4070 Super").GetComputeUnitsCount(),
            56);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4070 Ti").GetComputeUnitsCount(),
            60);
  EXPECT_EQ(
      NvidiaInfo("NVIDIA GeForce RTX 4070 Ti Super").GetComputeUnitsCount(),
      66);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4080").GetComputeUnitsCount(), 76);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4080 Super").GetComputeUnitsCount(),
            80);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 4090").GetComputeUnitsCount(), 128);
  EXPECT_EQ(NvidiaInfo("NVIDIA L4").GetComputeUnitsCount(), 58);
  EXPECT_EQ(NvidiaInfo("NVIDIA L20").GetComputeUnitsCount(), 92);
  EXPECT_EQ(NvidiaInfo("NVIDIA L40").GetComputeUnitsCount(), 142);
  EXPECT_EQ(NvidiaInfo("NVIDIA L40S").GetComputeUnitsCount(), 142);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX 2000 Ada Generation").GetComputeUnitsCount(),
            22);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX 3500 Ada Generation").GetComputeUnitsCount(),
            40);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX 4000 Ada Generation").GetComputeUnitsCount(),
            48);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX 5000 Ada Generation").GetComputeUnitsCount(),
            100);
  EXPECT_EQ(NvidiaInfo("NVIDIA RTX 6000 Ada Generation").GetComputeUnitsCount(),
            142);

  // Hopper
  EXPECT_EQ(NvidiaInfo("NVIDIA H100 80GB HBM3").GetComputeUnitsCount(), 132);
  EXPECT_EQ(NvidiaInfo("NVIDIA H200").GetComputeUnitsCount(), 132);
  EXPECT_EQ(NvidiaInfo("NVIDIA H800").GetComputeUnitsCount(), 132);
  EXPECT_EQ(NvidiaInfo("NVIDIA GH200").GetComputeUnitsCount(), 132);

  // Blackwell
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 5060").GetComputeUnitsCount(), 30);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 5060 Ti").GetComputeUnitsCount(),
            36);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 5070").GetComputeUnitsCount(), 48);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 5070 Ti").GetComputeUnitsCount(),
            70);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 5080").GetComputeUnitsCount(), 84);
  EXPECT_EQ(NvidiaInfo("NVIDIA GeForce RTX 5090").GetComputeUnitsCount(), 170);
  EXPECT_EQ(NvidiaInfo("NVIDIA B100").GetComputeUnitsCount(), 132);
  EXPECT_EQ(NvidiaInfo("NVIDIA B200").GetComputeUnitsCount(), 148);
  EXPECT_EQ(NvidiaInfo("NVIDIA B800").GetComputeUnitsCount(), 148);
  EXPECT_EQ(NvidiaInfo("NVIDIA GB200").GetComputeUnitsCount(), 148);

  // Architecture fallbacks (unknown GPU model with known architecture string)
  EXPECT_EQ(NvidiaInfo("unknown fermi gpu").GetComputeUnitsCount(), 16);
  EXPECT_EQ(NvidiaInfo("unknown kepler gpu").GetComputeUnitsCount(), 15);
  EXPECT_EQ(NvidiaInfo("unknown maxwell gpu").GetComputeUnitsCount(), 24);
  EXPECT_EQ(NvidiaInfo("unknown pascal gpu").GetComputeUnitsCount(), 28);
  EXPECT_EQ(NvidiaInfo("unknown volta gpu").GetComputeUnitsCount(), 80);
  EXPECT_EQ(NvidiaInfo("unknown turing gpu").GetComputeUnitsCount(), 40);
  EXPECT_EQ(NvidiaInfo("unknown ampere gpu").GetComputeUnitsCount(), 80);
  EXPECT_EQ(NvidiaInfo("unknown lovelace gpu").GetComputeUnitsCount(), 128);
  EXPECT_EQ(NvidiaInfo("unknown hopper gpu").GetComputeUnitsCount(), 132);
  EXPECT_EQ(NvidiaInfo("unknown blackwell gpu").GetComputeUnitsCount(), 148);
  EXPECT_EQ(NvidiaInfo("unknown brand new gpu").GetComputeUnitsCount(), 16);
}

AppleInfo GetAppleInfo(const std::string& description) {
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(description, GpuApi::kMetal, &gpu_info);
  return gpu_info.apple_info;
}

TEST(AppleInfoTest, TestComputeUnitsCountAndFamily) {
  // M1 series
  EXPECT_EQ(GetAppleInfo("Apple M1").gpu_type, AppleGpu::kM1);
  EXPECT_EQ(GetAppleInfo("Apple M1").GetComputeUnitsCount(), 8);
  EXPECT_EQ(GetAppleInfo("Apple M1 Pro").gpu_type, AppleGpu::kM1Pro);
  EXPECT_EQ(GetAppleInfo("Apple M1 Pro").GetComputeUnitsCount(), 16);
  EXPECT_EQ(GetAppleInfo("Apple M1 Max").gpu_type, AppleGpu::kM1Max);
  EXPECT_EQ(GetAppleInfo("Apple M1 Max").GetComputeUnitsCount(), 32);
  EXPECT_EQ(GetAppleInfo("Apple M1 Ultra").gpu_type, AppleGpu::kM1Ultra);
  EXPECT_EQ(GetAppleInfo("Apple M1 Ultra").GetComputeUnitsCount(), 64);
  EXPECT_TRUE(GetAppleInfo("Apple M1 Ultra").IsM1Series());
  EXPECT_EQ(GetAppleInfo("Apple M1 Ultra").gpu_family,
            AppleInfo::Family::kApple7);

  // M2 series
  EXPECT_EQ(GetAppleInfo("Apple M2").gpu_type, AppleGpu::kM2);
  EXPECT_EQ(GetAppleInfo("Apple M2").GetComputeUnitsCount(), 10);
  EXPECT_EQ(GetAppleInfo("Apple M2 Pro").gpu_type, AppleGpu::kM2Pro);
  EXPECT_EQ(GetAppleInfo("Apple M2 Pro").GetComputeUnitsCount(), 19);
  EXPECT_EQ(GetAppleInfo("Apple M2 Max").gpu_type, AppleGpu::kM2Max);
  EXPECT_EQ(GetAppleInfo("Apple M2 Max").GetComputeUnitsCount(), 38);
  EXPECT_EQ(GetAppleInfo("Apple M2 Ultra").gpu_type, AppleGpu::kM2Ultra);
  EXPECT_EQ(GetAppleInfo("Apple M2 Ultra").GetComputeUnitsCount(), 76);
  EXPECT_TRUE(GetAppleInfo("Apple M2 Ultra").IsM2Series());
  EXPECT_EQ(GetAppleInfo("Apple M2 Ultra").gpu_family,
            AppleInfo::Family::kApple8);

  // M3 series
  EXPECT_EQ(GetAppleInfo("Apple M3").gpu_type, AppleGpu::kM3);
  EXPECT_EQ(GetAppleInfo("Apple M3").GetComputeUnitsCount(), 10);
  EXPECT_EQ(GetAppleInfo("Apple M3 Pro").gpu_type, AppleGpu::kM3Pro);
  EXPECT_EQ(GetAppleInfo("Apple M3 Pro").GetComputeUnitsCount(), 18);
  EXPECT_EQ(GetAppleInfo("Apple M3 Max").gpu_type, AppleGpu::kM3Max);
  EXPECT_EQ(GetAppleInfo("Apple M3 Max").GetComputeUnitsCount(), 40);
  EXPECT_EQ(GetAppleInfo("Apple M3 Ultra").gpu_type, AppleGpu::kM3Ultra);
  EXPECT_EQ(GetAppleInfo("Apple M3 Ultra").GetComputeUnitsCount(), 80);
  EXPECT_TRUE(GetAppleInfo("Apple M3 Ultra").IsM3Series());
  EXPECT_EQ(GetAppleInfo("Apple M3 Ultra").gpu_family,
            AppleInfo::Family::kApple9);

  // M4 series
  EXPECT_EQ(GetAppleInfo("Apple M4").gpu_type, AppleGpu::kM4);
  EXPECT_EQ(GetAppleInfo("Apple M4").GetComputeUnitsCount(), 10);
  EXPECT_EQ(GetAppleInfo("Apple M4 Pro").gpu_type, AppleGpu::kM4Pro);
  EXPECT_EQ(GetAppleInfo("Apple M4 Pro").GetComputeUnitsCount(), 20);
  EXPECT_EQ(GetAppleInfo("Apple M4 Max").gpu_type, AppleGpu::kM4Max);
  EXPECT_EQ(GetAppleInfo("Apple M4 Max").GetComputeUnitsCount(), 40);

  // M5 series
  EXPECT_EQ(GetAppleInfo("Apple M5").gpu_type, AppleGpu::kM5);
  EXPECT_EQ(GetAppleInfo("Apple M5").GetComputeUnitsCount(), 10);
  EXPECT_EQ(GetAppleInfo("Apple M5 Pro").gpu_type, AppleGpu::kM5Pro);
  EXPECT_EQ(GetAppleInfo("Apple M5 Pro").GetComputeUnitsCount(), 20);
  EXPECT_EQ(GetAppleInfo("Apple M5 Max").gpu_type, AppleGpu::kM5Max);
  EXPECT_EQ(GetAppleInfo("Apple M5 Max").GetComputeUnitsCount(), 40);
}

}  // namespace
}  // namespace ml_drift
