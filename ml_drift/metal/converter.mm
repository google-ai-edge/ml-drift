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

#include "ml_drift/metal/converter.h"

#include "ml_drift/common/kernels/conversion.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace metal {

absl::Status TensorToBHWCBufferConverter::Init(Environment* env,
                                               const TensorDescriptor& src_desc,
                                               DataType buffer_type) {
  BufferDescriptor dst_desc;
  dst_desc.element_type = buffer_type;
  dst_desc.element_size = 1;
  dst_desc.memory_type = MemoryType::GLOBAL;

  const auto& gpu_info = env->GetInfo();
  TensorToBhwcBuffer gpu_op =
      CreateTensorToBhwcBufferOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, src_desc.GetBHWCShape());
  RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  task_.Init(std::make_unique<TensorToBhwcBuffer>(std::move(gpu_op)));
  return task_.Compile(env);
}

absl::Status TensorToBHWCBufferConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, MetalSpatialTensor* src,
    Buffer* dst) {
  task_.SetSrcTensor(src, 0);
  task_.SetDstBuffer(dst, 0);
  RETURN_IF_ERROR(task_.UpdateParams());
  task_.Encode(encoder);
  return absl::OkStatus();
}

absl::Status TensorToBHWCBufferConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, MetalSpatialTensor* src,
    id<MTLBuffer> dst) {
  Buffer wrapped = CreateBufferShared(dst);
  return Encode(encoder, src, &wrapped);
}

absl::Status BHWCBufferToTensorConverter::Init(
    Environment* env, DataType buffer_type, const TensorDescriptor& dst_desc) {
  BufferDescriptor src_desc;
  src_desc.element_type = buffer_type;
  src_desc.element_size = 1;
  src_desc.memory_type = MemoryType::GLOBAL;

  const auto& gpu_info = env->GetInfo();
  BhwcBufferToTensor gpu_op =
      CreateBhwcBufferToTensorOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  task_.Init(std::make_unique<GPUOperation>(std::move(gpu_op)));
  return task_.Compile(env);
}

absl::Status BHWCBufferToTensorConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, Buffer* src,
    MetalSpatialTensor* dst) {
  task_.SetSrcBuffer(src, 0);
  task_.SetDstTensor(dst, 0);
  RETURN_IF_ERROR(task_.UpdateParams());
  task_.Encode(encoder);
  return absl::OkStatus();
}

absl::Status BHWCBufferToTensorConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> src,
    MetalSpatialTensor* dst) {
  Buffer wrapped = CreateBufferShared(src);
  return Encode(encoder, &wrapped, dst);
}

}  // namespace metal
}  // namespace ml_drift
