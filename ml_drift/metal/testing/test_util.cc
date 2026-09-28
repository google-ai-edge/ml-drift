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

#include "ml_drift/metal/testing/test_util.h"

#import <Metal/Metal.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/metal/compute_task.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {

std::vector<DataType> MetalExecutionEnvironment::GetSupportedDataTypes() const {
  return {DataType::FLOAT32, DataType::FLOAT16};
}

std::vector<TensorStorageType> MetalExecutionEnvironment::GetSupportedStorages(
    DataType data_type) const {
  std::vector<TensorStorageType> storages = {
      TensorStorageType::BUFFER, TensorStorageType::TEXTURE_2D,
      TensorStorageType::TEXTURE_3D, TensorStorageType::TEXTURE_ARRAY};
  if (GetGpuInfo().SupportsImageBuffer()) {
    storages.push_back(TensorStorageType::IMAGE_BUFFER);
  }
  return storages;
}

absl::Status MetalExecutionEnvironment::ExecuteGpuOperationInternal(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
  @autoreleasepool {
    std::vector<MetalSpatialTensor> src(src_cpu.size());
    for (int i = 0; i < src_cpu.size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          src[i].CreateFromDescriptor(*src_cpu[i], device_.device()));
    }

    std::vector<MetalSpatialTensor> dst(dst_cpu.size());
    for (int i = 0; i < dst_cpu.size(); ++i) {
      ABSL_RETURN_IF_ERROR(
          dst[i].CreateFromDescriptor(*dst_cpu[i], device_.device()));
    }

    ComputeTask gpu_task;
    gpu_task.Init(std::move(operation));
    ABSL_RETURN_IF_ERROR(gpu_task.Compile(&device_));
    for (int i = 0; i < src_cpu.size(); ++i) {
      ABSL_RETURN_IF_ERROR(gpu_task.SetSrcTensor(&src[i], i));
    }
    for (int i = 0; i < dst_cpu.size(); ++i) {
      ABSL_RETURN_IF_ERROR(gpu_task.SetDstTensor(&dst[i], i));
    }
    ABSL_RETURN_IF_ERROR(gpu_task.UpdateParams());

    bool use_icb = false;
    if (use_icb) {
      if (@available(macOS 11.00, iOS 13.0, tvOS 13.0, *)) {
        MTLIndirectCommandBufferDescriptor* icb_desc =
            [[MTLIndirectCommandBufferDescriptor alloc] init];
        icb_desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icb_desc.inheritBuffers = NO;
        icb_desc.inheritPipelineState = NO;
        icb_desc.maxKernelBufferBindCount = 1;

        id<MTLIndirectCommandBuffer> icb =
            [device_.device() newIndirectCommandBufferWithDescriptor:icb_desc
                                                     maxCommandCount:1
                                                             options:0];

        id<MTLIndirectComputeCommand> icb_command =
            [icb indirectComputeCommandAtIndex:0];
        gpu_task.EncodeToICB(icb_command);
        [icb_command setBarrier];

        id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        gpu_task.AddResourcesToEncoder(encoder);
        [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      } else {
        return absl::InternalError(
            "Indirect compute command buffer available since ios 13");
      }
    } else {
      id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      gpu_task.Encode(encoder);
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }

    for (int i = 0; i < dst_cpu.size(); ++i) {
      ABSL_RETURN_IF_ERROR(dst[i].ToDescriptor(dst_cpu[i], device_.device()));
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace ml_drift
