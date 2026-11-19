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

#ifndef ML_DRIFT_GL_GL_ERRORS_H_
#define ML_DRIFT_GL_GL_ERRORS_H_

#include "absl/status/status.h"

namespace ml_drift {
namespace gl {

// @return recent opengl errors and packs them into Status.
absl::Status GetOpenGlErrors();

// @return the error of the last called EGL function in the current thread.
absl::Status GetEglError();

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_ERRORS_H_
