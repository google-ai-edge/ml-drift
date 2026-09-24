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

#include "ml_drift/gl/testing/gl_test.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/gl/gl_operation.h"
#include "ml_drift/gl/gl_spatial_tensor.h"

namespace ml_drift {
namespace gl {

absl::Status GlExecutionEnvironment::ExecuteGpuOperationInternal(
    const std::vector<ml_drift::TensorDescriptor*>& src_cpu,
    const std::vector<ml_drift::TensorDescriptor*>& dst_cpu,
    std::unique_ptr<ml_drift::GPUOperation>&& operation) {
  std::vector<GlSpatialTensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(src[i].CreateFromDescriptor(*src_cpu[i]));
  }

  std::vector<GlSpatialTensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(dst[i].CreateFromDescriptor(*dst_cpu[i]));
  }

  GlOperation gl_op;
  gl_op.Init(std::move(operation));
  ABSL_RETURN_IF_ERROR(gl_op.InitArgs(gpu_info_));
  for (int i = 0; i < src_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(gl_op.SetSrcTensor(&src[i], i));
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(gl_op.SetDstTensor(&dst[i], i));
  }
  ABSL_RETURN_IF_ERROR(gl_op.Assemble(gpu_info_));
  ABSL_RETURN_IF_ERROR(gl_op.AddToQueue());
  glFinish();

  for (int i = 0; i < dst_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(dst[i].ToDescriptor(dst_cpu[i]));
  }
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace ml_drift
