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
    EXPECT_LE(wgs[i].x, gpu_info.GetMaxWorkGroupSizeForX());
    EXPECT_LE(wgs[i].y, gpu_info.GetMaxWorkGroupSizeForY());
    EXPECT_LE(wgs[i].z, gpu_info.GetMaxWorkGroupSizeForZ());
    EXPECT_LE(wgs[i].x * wgs[i].y * wgs[i].z,
              gpu_info.GetMaxWorkGroupTotalSize());
  }
}

TEST(WorkGroupPickingTest, GetPossibleWorkGroupsStableWorkGroupSize) {
  ml_drift::GpuInfo gpu_info;
  gpu_info.gpu_api = ml_drift::GpuApi::kWebGpu;
  gpu_info.webgpu_info.max_compute_workgroup_size_x = 1024;
  gpu_info.webgpu_info.max_compute_workgroup_size_y = 1024;
  gpu_info.webgpu_info.max_compute_workgroup_size_z = 64;
  gpu_info.webgpu_info.max_compute_invocations_per_workgroup = 1024;
  ml_drift::KernelInfo kernel_info;
  kernel_info.max_work_group_size = gpu_info.GetMaxWorkGroupTotalSize();

  // By default the fast work group depends on the exact grid size.
  auto wgs_a =
      ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFast, gpu_info,
                                      kernel_info, ml_drift::int3(514, 3, 1));
  auto wgs_b =
      ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFast, gpu_info,
                                      kernel_info, ml_drift::int3(518, 3, 1));
  ASSERT_EQ(wgs_a.size(), 1);
  ASSERT_EQ(wgs_b.size(), 1);
  EXPECT_NE(wgs_a[0], wgs_b[0]);

  // With kFastStable, grids with the same non-trivial dimensions map to the
  // same work group regardless of their exact sizes.
  wgs_a = ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFastStable,
                                          gpu_info, kernel_info,
                                          ml_drift::int3(514, 3, 1));
  wgs_b = ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFastStable,
                                          gpu_info, kernel_info,
                                          ml_drift::int3(518, 3, 1));
  ASSERT_EQ(wgs_a.size(), 1);
  ASSERT_EQ(wgs_b.size(), 1);
  EXPECT_EQ(wgs_a[0], wgs_b[0]);
  EXPECT_EQ(wgs_a[0], ml_drift::int3(16, 16, 1));

  // 1D and 3D grids, and the total size limit.
  EXPECT_EQ(ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFastStable,
                                            gpu_info, kernel_info,
                                            ml_drift::int3(1, 290, 1))[0],
            ml_drift::int3(1, 256, 1));
  EXPECT_EQ(ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFastStable,
                                            gpu_info, kernel_info,
                                            ml_drift::int3(32, 18, 3))[0],
            ml_drift::int3(8, 8, 4));
  kernel_info.max_work_group_size = 64;
  const auto clamped = ml_drift::GetPossibleWorkGroups(
      ml_drift::TuningType::kFastStable, gpu_info, kernel_info,
      ml_drift::int3(514, 3, 1))[0];
  EXPECT_LE(clamped.x * clamped.y * clamped.z, 64);
}

TEST(WorkGroupPickingTest, GetPossibleWorkGroupsPerDimAndKernelLimits) {
  ml_drift::GpuInfo gpu_info;
  gpu_info.gpu_api = ml_drift::GpuApi::kOpenCl;
  gpu_info.opencl_info.max_work_group_size_x = 1024;
  gpu_info.opencl_info.max_work_group_size_y = 1024;
  gpu_info.opencl_info.max_work_group_size_z = 1024;
  gpu_info.opencl_info.max_work_group_total_size = 2048;
  ml_drift::KernelInfo kernel_info;
  kernel_info.max_work_group_size = 2048;

  // Grid [2704, 1, 1] where DivideRoundUp(2704, 2) = 1352 > 1024.
  auto wgs =
      ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFast, gpu_info,
                                      kernel_info, ml_drift::int3(2704, 1, 1));
  ASSERT_EQ(wgs.size(), 1);
  EXPECT_EQ(wgs[0], ml_drift::int3(1024, 1, 1));

  // Kernel-specific max_work_group_size smaller than device total size.
  kernel_info.max_work_group_size = 512;
  wgs =
      ml_drift::GetPossibleWorkGroups(ml_drift::TuningType::kFast, gpu_info,
                                      kernel_info, ml_drift::int3(2704, 1, 1));
  ASSERT_EQ(wgs.size(), 1);
  EXPECT_EQ(wgs[0], ml_drift::int3(512, 1, 1));

  // Conv work-group picking with smaller per-dimension X limit.
  gpu_info.opencl_info.max_work_group_size_x = 64;
  auto conv_wgs = ml_drift::GetPossibleWorkGroupsConv(
      ml_drift::TuningType::kFast, gpu_info, kernel_info,
      ml_drift::int3(256, 1, 1));
  ASSERT_EQ(conv_wgs.size(), 1);
  EXPECT_LE(conv_wgs[0].x, gpu_info.GetMaxWorkGroupSizeForX());
  EXPECT_LE(conv_wgs[0].y, gpu_info.GetMaxWorkGroupSizeForY());
  EXPECT_LE(conv_wgs[0].z, gpu_info.GetMaxWorkGroupSizeForZ());
  EXPECT_LE(conv_wgs[0].x * conv_wgs[0].y * conv_wgs[0].z,
            kernel_info.max_work_group_size);
}

}  // namespace
