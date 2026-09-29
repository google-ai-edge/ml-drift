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

#include "ml_drift/common/task/work_group_picking.h"

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace {

TEST(WorkGroupPickingTest, GetPossibleWorkGroups) {
  ml_drift::GpuInfo gpu_info;
  gpu_info.gpu_api = ml_drift::GpuApi::kOpenCl;
  gpu_info.opencl_info.max_work_group_size_x = 1536;
  gpu_info.opencl_info.max_work_group_size_y = 1024;
  gpu_info.opencl_info.max_work_group_size_z = 64;
  gpu_info.opencl_info.max_work_group_total_size = 1536;
  ml_drift::KernelInfo kernel_info;
  kernel_info.max_work_group_size = gpu_info.GetMaxWorkGroupTotalSize();

  auto wgs =
      ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFast, gpu_info,
                                      kernel_info, ml_drift::int3(1, 1536, 1));

  for (int i = 0; i < wgs.size(); ++i) {
    EXPECT_LE(wgs[0].x, gpu_info.GetMaxWorkGroupSizeForX());
    EXPECT_LE(wgs[0].y, gpu_info.GetMaxWorkGroupSizeForY());
    EXPECT_LE(wgs[0].z, gpu_info.GetMaxWorkGroupSizeForZ());
    EXPECT_LE(wgs[0].x * wgs[0].y * wgs[0].z,
              gpu_info.GetMaxWorkGroupTotalSize());
  }
}

TEST(WorkGroupPickingTest, PreVoltaNvidiaWebGpuUsesFallbackWorkGroup) {
  ml_drift::GpuInfo gpu_info;
  gpu_info.gpu_api = ml_drift::GpuApi::kWebGpu;
  gpu_info.vendor = ml_drift::GpuVendor::kNvidia;
  gpu_info.nvidia_info.architecture = ml_drift::NvidiaArchitecture::kPascal;
  gpu_info.webgpu_info.max_compute_workgroup_size_x = 1024;
  gpu_info.webgpu_info.max_compute_workgroup_size_y = 1024;
  gpu_info.webgpu_info.max_compute_workgroup_size_z = 64;
  gpu_info.webgpu_info.max_compute_invocations_per_workgroup = 1024;

  ml_drift::KernelInfo kernel_info;
  kernel_info.max_work_group_size = 1536;

  const ml_drift::int3 grid(64, 6, 8);
  auto pascal_wgs = ml_drift::GetPossibleWorkGroups(
      ml_drift::TuningType::kFast, gpu_info, kernel_info, grid);
  ASSERT_EQ(pascal_wgs.size(), 1);
  EXPECT_EQ(pascal_wgs[0], ml_drift::int3(8, 4, 1));

  gpu_info.nvidia_info.architecture = ml_drift::NvidiaArchitecture::kVolta;
  auto volta_wgs = ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFast,
                                                   gpu_info, kernel_info, grid);
  ASSERT_EQ(volta_wgs.size(), 1);
  EXPECT_EQ(volta_wgs[0], ml_drift::int3(32, 6, 8));
}

}  // namespace
