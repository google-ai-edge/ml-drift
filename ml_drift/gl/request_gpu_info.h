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

#ifndef ML_DRIFT_GL_REQUEST_GPU_INFO_H_
#define ML_DRIFT_GL_REQUEST_GPU_INFO_H_

#include <string>

#include "absl/status/status.h"
#include "ml_drift/common/gpu_info.h"

namespace ml_drift {
namespace gl {

// This method performs multiple GL calls, therefore, egl context needs to be
// created upfront.
absl::Status RequestOpenGlInfo(OpenGlInfo* gl_info);

// This method performs multiple GL calls, therefore, egl context needs to be
// created upfront.
absl::Status RequestGpuInfo(GpuInfo* gpu_info);

AdrenoInfo::OpenGlDriverVersion GeAdrenoDriverVersion(
    const std::string& gl_version);

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_REQUEST_GPU_INFO_H_
