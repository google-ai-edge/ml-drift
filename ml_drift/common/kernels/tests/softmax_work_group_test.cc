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

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/softmax.h"
#include "ml_drift/common/kernels/softmax1x1.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace {

GpuInfo GetOpenClGpuInfo(const std::string& description, int compute_units) {
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(description, GpuApi::kOpenCl, &gpu_info);
  gpu_info.opencl_info.compute_units_count = compute_units;
  gpu_info.opencl_info.max_work_group_size_x = 256;
  gpu_info.opencl_info.max_work_group_size_y = 256;
  gpu_info.opencl_info.max_work_group_size_z = 256;
  gpu_info.opencl_info.max_work_group_total_size = 256;
  return gpu_info;
}

GpuInfo GetOpenGlGpuInfo(const std::string& description) {
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(description, GpuApi::kOpenGl, &gpu_info);
  gpu_info.opengl_info.max_compute_work_group_size_x = 256;
  gpu_info.opengl_info.max_compute_work_group_size_y = 256;
  gpu_info.opengl_info.max_compute_work_group_size_z = 64;
  gpu_info.opengl_info.max_work_group_invocations = 128;
  return gpu_info;
}

OperationDef GetOperationDef() {
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kFloat32, TensorStorageType::kBuffer, Layout::kHWC});
  op_def.dst_tensors.push_back(
      {DataType::kFloat32, TensorStorageType::kBuffer, Layout::kHWC});
  return op_def;
}

int TotalSize(const int3& wg) { return wg.x * wg.y * wg.z; }

// Shape that selects the work group reduction path of Softmax on a 2-CU GPU.
BHWC GetReductionShape() { return BHWC(1, 16, 16, 256); }

// Shape too small to fill the GPU, so Softmax uses the fallback work group.
BHWC GetSmallReductionShape() { return BHWC(1, 4, 4, 256); }

TEST(SoftmaxWorkGroupTest, MidgardSoftmaxFitsIn64Threads) {
  const GpuInfo gpu_info = GetOpenClGpuInfo("Mali-T830", /*compute_units=*/2);
  ASSERT_TRUE(gpu_info.IsMali());
  ASSERT_TRUE(gpu_info.mali_info.IsMidgard());

  Softmax op = CreateSoftmax(GetOperationDef(), gpu_info, GetReductionShape());
  EXPECT_LE(TotalSize(op.work_group_size_), 64);
}

TEST(SoftmaxWorkGroupTest, NonMidgardSoftmaxKeeps128Threads) {
  const GpuInfo gpu_info = GetOpenClGpuInfo("Mali-G72", /*compute_units=*/2);
  ASSERT_TRUE(gpu_info.IsMali());
  ASSERT_FALSE(gpu_info.mali_info.IsMidgard());

  Softmax op = CreateSoftmax(GetOperationDef(), gpu_info, GetReductionShape());
  EXPECT_EQ(TotalSize(op.work_group_size_), 128);
}

TEST(SoftmaxWorkGroupTest, MidgardSoftmax1x1FitsIn64Threads) {
  const GpuInfo gpu_info = GetOpenClGpuInfo("Mali-T830", /*compute_units=*/2);
  Softmax1x1 op =
      CreateSoftmax1x1(GetOperationDef(), gpu_info, BHWC(1, 1, 1, 1001));
  EXPECT_LE(TotalSize(op.work_group_size_), 64);
}

TEST(SoftmaxWorkGroupTest, MidgardSoftmaxFallbackFitsIn64Threads) {
  const GpuInfo gpu_info = GetOpenClGpuInfo("Mali-T830", /*compute_units=*/2);
  Softmax op =
      CreateSoftmax(GetOperationDef(), gpu_info, GetSmallReductionShape());
  EXPECT_LE(TotalSize(op.work_group_size_), 64);
}

TEST(SoftmaxWorkGroupTest, MidgardOpenGlSoftmaxKeeps128Threads) {
  const GpuInfo gpu_info = GetOpenGlGpuInfo("Mali-T830");
  ASSERT_TRUE(gpu_info.mali_info.IsMidgard());
  Softmax op =
      CreateSoftmax(GetOperationDef(), gpu_info, GetSmallReductionShape());
  EXPECT_EQ(TotalSize(op.work_group_size_), 128);
}

TEST(SoftmaxWorkGroupTest, MidgardOpenGlSoftmax1x1Keeps128Threads) {
  const GpuInfo gpu_info = GetOpenGlGpuInfo("Mali-T830");
  Softmax1x1 op =
      CreateSoftmax1x1(GetOperationDef(), gpu_info, BHWC(1, 1, 1, 1001));
  EXPECT_EQ(TotalSize(op.work_group_size_), 128);
}

}  // namespace
}  // namespace ml_drift
