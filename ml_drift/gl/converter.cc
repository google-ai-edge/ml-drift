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

#include "ml_drift/gl/converter.h"

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
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_operation.h"
#include "ml_drift/gl/gl_spatial_tensor.h"

namespace ml_drift {
namespace gl {

absl::Status TensorToTensorConverter::Init(const GpuInfo& gpu_info,
                                           const TensorDescriptor& src_desc,
                                           const TensorDescriptor& dst_desc,
                                           ProgramCache* program_cache) {
  TensorToTensor gpu_op = CreateTensorToTensorOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  const int3 wg_size = gpu_op.GetWorkGroupSize();
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  op_.Init(std::make_unique<GPUOperation>(std::move(gpu_op)));
  ABSL_RETURN_IF_ERROR(op_.InitArgs(gpu_info));
  ABSL_RETURN_IF_ERROR(op_.Assemble(gpu_info, program_cache, &wg_size));
  return absl::OkStatus();
}

absl::Status TensorToTensorConverter::Convert(GlSpatialTensor* src,
                                              GlSpatialTensor* dst) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(src, 0));
  ABSL_RETURN_IF_ERROR(op_.SetDstTensor(dst, 0));
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  return op_.AddToQueue();
}

absl::Status TensorToBHWCBufferConverter::Init(const GpuInfo& gpu_info,
                                               const TensorDescriptor& src_desc,
                                               DataType buffer_type,
                                               ProgramCache* program_cache) {
  BufferDescriptor dst_desc;
  dst_desc.element_type = buffer_type;
  dst_desc.element_size = 1;
  dst_desc.memory_type = MemoryType::kGlobal;
  return Init(gpu_info, src_desc, dst_desc, program_cache);
}

absl::Status TensorToBHWCBufferConverter::Init(const GpuInfo& gpu_info,
                                               const TensorDescriptor& src_desc,
                                               const BufferDescriptor& dst_desc,
                                               ProgramCache* program_cache) {
  TensorToBhwcBuffer gpu_op =
      CreateTensorToBhwcBufferOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, src_desc.GetBHWCShape());
  const int3 wg_size = gpu_op.GetWorkGroupSize();
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  op_.Init(std::make_unique<TensorToBhwcBuffer>(std::move(gpu_op)));
  ABSL_RETURN_IF_ERROR(op_.InitArgs(gpu_info));
  ABSL_RETURN_IF_ERROR(op_.Assemble(gpu_info, program_cache, &wg_size));
  return absl::OkStatus();
}

absl::Status TensorToBHWCBufferConverter::Convert(GlSpatialTensor* src,
                                                  GlBuffer* dst) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(src, 0));
  ABSL_RETURN_IF_ERROR(op_.SetDstBuffer(dst, 0));
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  return op_.AddToQueue();
}

absl::Status BHWCBufferToTensorConverter::Init(const GpuInfo& gpu_info,
                                               DataType buffer_type,
                                               const TensorDescriptor& dst_desc,
                                               ProgramCache* program_cache) {
  BufferDescriptor src_desc;
  src_desc.element_type = buffer_type;
  src_desc.element_size = 1;
  src_desc.memory_type = MemoryType::kGlobal;
  return Init(gpu_info, src_desc, dst_desc, program_cache);
}

absl::Status BHWCBufferToTensorConverter::Init(const GpuInfo& gpu_info,
                                               const BufferDescriptor& src_desc,
                                               const TensorDescriptor& dst_desc,
                                               ProgramCache* program_cache) {
  BhwcBufferToTensor gpu_op =
      CreateBhwcBufferToTensorOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  const int3 wg_size = gpu_op.GetWorkGroupSize();
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  op_.Init(std::make_unique<GPUOperation>(std::move(gpu_op)));
  ABSL_RETURN_IF_ERROR(op_.InitArgs(gpu_info));
  ABSL_RETURN_IF_ERROR(op_.Assemble(gpu_info, program_cache, &wg_size));
  return absl::OkStatus();
}

absl::Status BHWCBufferToTensorConverter::Convert(GlBuffer* src,
                                                  GlSpatialTensor* dst) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcBuffer(src, 0));
  ABSL_RETURN_IF_ERROR(op_.SetDstTensor(dst, 0));
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  return op_.AddToQueue();
}

}  // namespace gl
}  // namespace ml_drift
