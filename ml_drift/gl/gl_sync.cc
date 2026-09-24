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

#include "ml_drift/gl/gl_sync.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"

#ifdef __ARM_ACLE
#include <arm_acle.h>
#endif  // __ARM_ACLE

#include "ml_drift/gl/gl_errors.h"

namespace ml_drift {
namespace gl {

absl::Status GlSyncWait() {
  GlSync sync;
  ABSL_RETURN_IF_ERROR(GlSync::NewSync(&sync));
  // Flush sync and loop afterwards without it.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  while (true) {
    switch (status) {
      case GL_TIMEOUT_EXPIRED:
        break;
      case GL_CONDITION_SATISFIED:
      case GL_ALREADY_SIGNALED:
        return absl::OkStatus();
      case GL_WAIT_FAILED:
        return GetOpenGlErrors();
    }
    status = glClientWaitSync(sync.sync(), 0, /* timeout ns = */ 10000000);
  }
  return absl::OkStatus();
}

absl::Status GlActiveSyncWait() {
  GlSync sync;
  ABSL_RETURN_IF_ERROR(GlSync::NewSync(&sync));
  // Since creating a Sync object is itself a GL command it *must* be flushed.
  // Otherwise glGetSynciv may never succeed. Perform a flush with
  // glClientWaitSync call.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  switch (status) {
    case GL_TIMEOUT_EXPIRED:
      break;
    case GL_CONDITION_SATISFIED:
    case GL_ALREADY_SIGNALED:
      return absl::OkStatus();
    case GL_WAIT_FAILED:
      return GetOpenGlErrors();
  }

  // Start active loop.
  GLint result = GL_UNSIGNALED;
  while (true) {
    glGetSynciv(sync.sync(), GL_SYNC_STATUS, sizeof(GLint), nullptr, &result);
    if (result == GL_SIGNALED) {
      return absl::OkStatus();
    }
#ifdef __ARM_ACLE
    // Try to save CPU power by yielding CPU to another thread.
    __yield();
#endif
  }
}

}  // namespace gl
}  // namespace ml_drift
