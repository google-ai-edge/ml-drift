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

#ifndef ML_DRIFT_CL_CL_ERRORS_H_
#define ML_DRIFT_CL_CL_ERRORS_H_

#include "absl/status/status.h"
#include "ml_drift/cl/util.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>

namespace ml_drift {
namespace cl {

// @return if error_code is success, then return OK status. Otherwise translates
// error code into a message.
inline absl::Status GetOpenCLError(cl_int error_code) {
  if (error_code == CL_SUCCESS) {
    return absl::OkStatus();
  }
  return absl::InternalError("OpenCL error: " +
                             CLErrorCodeToString(error_code));
}

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_ERRORS_H_
