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

#ifndef ML_DRIFT_GL_GL_SYNC_H_
#define ML_DRIFT_GL_GL_SYNC_H_

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_call.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {

// RAII wrapper for OpenGL GLsync object.
// See https://www.khronos.org/opengl/wiki/Sync_Object for more information.
//
// GlSync is moveable but not copyable.
class GlSync {
 public:
  static absl::Status NewSync(GlSync* gl_sync) {
    GLsync sync;
    ABSL_RETURN_IF_ERROR(ML_DRIFT_CALL_GL(glFenceSync, &sync,
                                          GL_SYNC_GPU_COMMANDS_COMPLETE, 0));
    *gl_sync = GlSync(sync);
    return absl::OkStatus();
  }

  // Creates invalid object.
  GlSync() : GlSync(nullptr) {}

  // Move-only
  GlSync(GlSync&& sync) : sync_(sync.sync_) { sync.sync_ = nullptr; }

  GlSync& operator=(GlSync&& sync) {
    if (this != &sync) {
      Invalidate();
      std::swap(sync_, sync.sync_);
    }
    return *this;
  }

  GlSync(const GlSync&) = delete;
  GlSync& operator=(const GlSync&) = delete;

  ~GlSync() { Invalidate(); }

  const GLsync sync() const { return sync_; }

 private:
  explicit GlSync(GLsync sync) : sync_(sync) {}

  void Invalidate() {
    if (sync_) {
      glDeleteSync(sync_);
      sync_ = nullptr;
    }
  }

  GLsync sync_;
};

// Waits until GPU is done with processing.
absl::Status GlSyncWait();

// Waits until all commands are flushed and then performs active waiting by
// spinning a thread and checking sync status. It leads to shorter wait time
// (up to tens of ms) but consumes more CPU.
absl::Status GlActiveSyncWait();

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_SYNC_H_
