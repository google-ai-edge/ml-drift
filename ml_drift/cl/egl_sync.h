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

#ifndef ML_DRIFT_CL_EGL_SYNC_H_
#define ML_DRIFT_CL_EGL_SYNC_H_

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "absl/status/status.h"

namespace ml_drift {
namespace cl {

// RAII wrapper for EGL sync object.
// EglSync is moveable but not copyable.
class EglSync {
 public:
  // Creates a fence in OpenGL command stream. This sync is enqueued and *not*
  // flushed.
  //
  // Depends on EGL_KHR_fence_sync extension.
  static absl::Status NewFence(EGLDisplay display, EglSync* sync);

  // Creates invalid object.
  EglSync() : EglSync(EGL_NO_DISPLAY, EGL_NO_SYNC_KHR) {}

  EglSync(EGLDisplay display, EGLSyncKHR sync)
      : display_(display), sync_(sync) {}

  // Move-only
  EglSync(EglSync&& sync);
  EglSync& operator=(EglSync&& sync);
  EglSync(const EglSync&) = delete;
  EglSync& operator=(const EglSync&) = delete;

  ~EglSync() { Invalidate(); }

  // Causes GPU to block and wait until this sync has been signaled.
  // This call does not block and returns immediately.
  absl::Status ServerWait();

  // Causes CPU to block and wait until this sync has been signaled.
  absl::Status ClientWait();

  // Returns the EGLDisplay on which this instance was created.
  EGLDisplay display() const { return display_; }

  // Returns the EGLSyncKHR wrapped by this instance.
  EGLSyncKHR sync() const { return sync_; }

  // Returns true if this instance wraps a valid EGLSync object.
  bool is_valid() const { return sync_ != nullptr; }

 private:
  void Invalidate();

  EGLDisplay display_;
  EGLSyncKHR sync_;
};

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_EGL_SYNC_H_
