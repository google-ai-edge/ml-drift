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

#include "ml_drift/webgpu/converter.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conversion.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

absl::Status TensorToTensorConverter::Init(const Environment& env,
                                           const TensorDescriptor& src_desc,
                                           const TensorDescriptor& dst_desc) {
  const GpuInfo& gpu_info = env.GetInfo();
  TensorToTensor gpu_op = CreateTensorToTensorOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  ABSL_RETURN_IF_ERROR(
      op_.Init(env, std::make_unique<GPUOperation>(std::move(gpu_op))));
  ABSL_RETURN_IF_ERROR(op_.Compile(env, env.GetComputePipelineCache()));
  return absl::OkStatus();
}

absl::Status TensorToTensorConverter::Convert(
    const wgpu::Device& device, wgpu::ComputePassEncoder compute_encoder,
    SpatialTensor* src, SpatialTensor* dst) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(0, src));
  ABSL_RETURN_IF_ERROR(op_.SetDstTensor(0, dst));
  ABSL_RETURN_IF_ERROR(op_.Update(device));
  op_.UpdateGpuObjectBindings(device);
  return op_.Encode(compute_encoder);
}

absl::Status TensorToBHWCBufferConverter::Init(const Environment& env,
                                               const TensorDescriptor& src_desc,
                                               DataType buffer_type) {
  BufferDescriptor dst_desc;
  dst_desc.element_type = buffer_type;
  dst_desc.element_size = 1;
  dst_desc.memory_type = MemoryType::kGlobal;
  return Init(env, src_desc, dst_desc);
}

absl::Status TensorToBHWCBufferConverter::Init(
    const Environment& env, const TensorDescriptor& src_desc,
    const BufferDescriptor& dst_desc) {
  const GpuInfo& gpu_info = env.GetInfo();
  TensorToBhwcBuffer gpu_op;
  if (SizeOf(dst_desc.element_type) >= 4) {
    gpu_op = CreateTensorToBhwcBufferOp(gpu_info, src_desc, dst_desc);
  } else {
    gpu_op = CreateTensorToBhwcBufferAlignedOp(gpu_info, src_desc, dst_desc);
  }
  gpu_op.RecalculateWorkGroupsSize(gpu_info, src_desc.GetBHWCShape());
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  ABSL_RETURN_IF_ERROR(
      op_.Init(env, std::make_unique<TensorToBhwcBuffer>(std::move(gpu_op))));
  ABSL_RETURN_IF_ERROR(op_.Compile(env, env.GetComputePipelineCache()));
  return absl::OkStatus();
}

absl::Status TensorToBHWCBufferConverter::Convert(
    const wgpu::Device& device, wgpu::ComputePassEncoder compute_encoder,
    SpatialTensor* src, Buffer* dst) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(0, src));
  ABSL_RETURN_IF_ERROR(op_.SetDstBuffer(0, dst));
  ABSL_RETURN_IF_ERROR(op_.Update(device));
  op_.UpdateGpuObjectBindings(device);
  return op_.Encode(compute_encoder);
}

absl::Status BHWCBufferToTensorConverter::Init(
    const Environment& env, DataType buffer_type,
    const TensorDescriptor& dst_desc) {
  BufferDescriptor src_desc;
  src_desc.element_type = buffer_type;
  src_desc.element_size = 1;
  src_desc.memory_type = MemoryType::kGlobal;
  return Init(env, src_desc, dst_desc);
}

absl::Status BHWCBufferToTensorConverter::Init(
    const Environment& env, const BufferDescriptor& src_desc,
    const TensorDescriptor& dst_desc) {
  const GpuInfo& gpu_info = env.GetInfo();
  BhwcBufferToTensor gpu_op;
  if (SizeOf(src_desc.element_type) >= 4) {
    gpu_op = CreateBhwcBufferToTensorOp(gpu_info, src_desc, dst_desc);
  } else {
    gpu_op = CreateBhwcBufferAlignedToTensorOp(gpu_info, src_desc, dst_desc);
  }
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  ABSL_RETURN_IF_ERROR(
      op_.Init(env, std::make_unique<GPUOperation>(std::move(gpu_op))));
  ABSL_RETURN_IF_ERROR(op_.Compile(env, env.GetComputePipelineCache()));
  return absl::OkStatus();
}

absl::Status BHWCBufferToTensorConverter::Convert(
    const wgpu::Device& device, wgpu::ComputePassEncoder compute_encoder,
    Buffer* src, SpatialTensor* dst) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcBuffer(0, src));
  ABSL_RETURN_IF_ERROR(op_.SetDstTensor(0, dst));
  ABSL_RETURN_IF_ERROR(op_.Update(device));
  op_.UpdateGpuObjectBindings(device);
  return op_.Encode(compute_encoder);
}

}  // namespace webgpu
}  // namespace ml_drift
