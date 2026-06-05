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

#ifndef ML_DRIFT_CL_CL_CONTEXT_H_
#define ML_DRIFT_CL_CL_CONTEXT_H_

#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util_types.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace cl {

// A RAII wrapper around opencl context
class CLContext {
 public:
  CLContext();
  CLContext(cl_context context, bool has_ownership);
  CLContext(cl_context context, bool has_ownership, CLDevice& device);
  // Move only
  CLContext(CLContext&& context);
  CLContext& operator=(CLContext&& context);
  CLContext(const CLContext&) = delete;
  CLContext& operator=(const CLContext&) = delete;

  ~CLContext();

  cl_context context() const { return context_; }

  bool IsFloatTexture2DSupported(int num_channels, DataType data_type,
                                 cl_mem_flags flags = CL_MEM_READ_WRITE) const;

 private:
  void Release();

  cl_context context_ = nullptr;
  bool has_ownership_ = false;
};

struct CLContextOptions {
  // Desired priority for enqueued kernels to be submitted to the device.
  // Kernels enqueued on higher priority contexts take priority over kernels
  // enqueued on lower priority contexts. Low priority contexts are useful for
  // instance to keep UI or rendering operations on other contexts responsive.
  // Defaults to kNormal according to the extension specs.
  PriorityHint priority = PriorityHint::kNormal;

  // Desired performance level for the device on the context.
  // Higher performance implies higher frequencies on the device.
  // Defaults to kHigh according to the extension specs.
  PerformanceHint performance = PerformanceHint::kHigh;
};

absl::Status CreateCLContext(const CLDevice& device, CLContext* result,
                             const CLContextOptions& options = {});
absl::Status CreateCLGLContext(const CLDevice& device,
                               cl_context_properties egl_context,
                               cl_context_properties egl_display,
                               CLContext* result,
                               const CLContextOptions& options = {});

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_CONTEXT_H_
