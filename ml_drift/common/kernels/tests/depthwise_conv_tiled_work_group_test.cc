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

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/depthwise_conv_tiled.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

GpuInfo GetGpuInfo(const std::string& description, GpuApi gpu_api) {
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(description, gpu_api, &gpu_info);
  return gpu_info;
}

// 3x3 depthwise convolution over 64 channels, as in MobileNet.
DepthwiseConvolution2DAttributes Get3x3Attributes() {
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  weights.shape = OHWI(1, 3, 3, 64);
  weights.data.resize(weights.shape.DimensionsProduct(), 0.0f);
  attr.bias.shape = Linear(64);
  attr.bias.data.resize(64, 0.0f);
  return attr;
}

OperationDef GetOperationDef() {
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kFloat16, TensorStorageType::kTexture2D, Layout::kHWC});
  op_def.dst_tensors.push_back(
      {DataType::kFloat16, TensorStorageType::kTexture2D, Layout::kHWC});
  return op_def;
}

class PowerVrOpenClTest : public ::testing::TestWithParam<std::string> {};

TEST_P(PowerVrOpenClTest, ExhaustiveTuningOffersOnlyTheRequiredWorkGroup) {
  const GpuInfo gpu_info = GetGpuInfo(GetParam(), GpuApi::kOpenCl);
  const KernelInfo kernel_info = {/*private_memory_size=*/0,
                                  /*max_work_group_size=*/512};
  std::unique_ptr<GPUOperation> op = CreateDepthwiseConvTiled(
      gpu_info, GetOperationDef(), CalculationsPrecision::kF16,
      Get3x3Attributes());
  ASSERT_NE(op->code_.find("reqd_work_group_size"), std::string::npos);

  std::vector<GPUOperation::DispatchInfo> dispatches;
  op->GetPossibleDispatches(TuningType::kExhaustive, gpu_info, kernel_info,
                            &dispatches);

  ASSERT_EQ(dispatches.size(), 1);
  EXPECT_EQ(dispatches[0].work_group_size, op->work_group_size_);
}

INSTANTIATE_TEST_SUITE_P(DepthwiseConvTiledWorkGroupTest, PowerVrOpenClTest,
                         ::testing::Values("PowerVR GE8320", "PowerVR GE8322",
                                           "PowerVR BXM-8-256"));

TEST(DepthwiseConvTiledWorkGroupTest, OtherGpusDoNotRequireAWorkGroup) {
  const GpuInfo adreno = GetGpuInfo("Adreno (TM) 830", GpuApi::kOpenCl);
  const GpuInfo powervr_opengl = GetGpuInfo("PowerVR GE8320", GpuApi::kOpenGl);

  std::unique_ptr<GPUOperation> adreno_op =
      CreateDepthwiseConvTiled(adreno, GetOperationDef(),
                               CalculationsPrecision::kF16, Get3x3Attributes());
  std::unique_ptr<GPUOperation> powervr_opengl_op = CreateDepthwiseConvTiled(
      powervr_opengl, GetOperationDef(), CalculationsPrecision::kF16,
      Get3x3Attributes());

  EXPECT_EQ(adreno_op->code_.find("reqd_work_group_size"), std::string::npos);
  EXPECT_EQ(powervr_opengl_op->code_.find("reqd_work_group_size"),
            std::string::npos);
}

}  // namespace
}  // namespace ml_drift
