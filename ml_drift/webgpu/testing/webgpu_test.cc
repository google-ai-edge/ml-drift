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

#include "ml_drift/webgpu/testing/webgpu_test.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/webgpu/compute_task.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"  // IWYU pragma: keep

namespace ml_drift {
namespace webgpu {

absl::Status WebGpuExecutionEnvironment::ExecuteGpuOperationInternal(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
  std::vector<SpatialTensor> src(src_cpu.size());
  for (size_t i = 0; i < src_cpu.size(); ++i) {
    RETURN_IF_ERROR(src[i].CreateFromDescriptor(env_.device(), *src_cpu[i]));
    operation->SetSrc(&src[i], i);
  }

  std::vector<SpatialTensor> dst(dst_cpu.size());
  for (size_t i = 0; i < dst_cpu.size(); ++i) {
    RETURN_IF_ERROR(dst[i].CreateFromDescriptor(env_.device(), *dst_cpu[i]));
    operation->SetDst(&dst[i], i);
  }

  ComputeTask webgpu_op;
  RETURN_IF_ERROR(webgpu_op.Init(env_, std::move(operation)));
  for (size_t i = 0; i < src.size(); ++i) {
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(i, &src[i]));
  }
  for (size_t i = 0; i < dst.size(); ++i) {
    RETURN_IF_ERROR(webgpu_op.SetDstTensor(i, &dst[i]));
  }
  RETURN_IF_ERROR(webgpu_op.Update(env_.device()));
  webgpu_op.UpdateGpuObjectBindings(env_.device());
  RETURN_IF_ERROR(webgpu_op.Compile(env_));

  wgpu::CommandEncoder encoder = env_.device().CreateCommandEncoder();

  wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
  RETURN_IF_ERROR(webgpu_op.Encode(compute_encoder));
  compute_encoder.End();
  wgpu::CommandBuffer cb = encoder.Finish();
  env_.queue().Submit(1, &cb);

  for (size_t i = 0; i < dst_cpu.size(); ++i) {
    RETURN_IF_ERROR(dst[i].ToDescriptor(env_.device(), dst_cpu[i]));
  }
  return absl::OkStatus();
}

}  // namespace webgpu
}  // namespace ml_drift
