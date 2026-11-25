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

#include "ml_drift/cl/egl_sync.h"

#include <cstring>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/gl/gl_errors.h"

namespace ml_drift {
namespace cl {

namespace {

bool HasExtension(EGLDisplay display, const char* extension) {
  const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
  return extensions && std::strstr(extensions, extension);
}

absl::Status IsEglFenceSyncSupported(EGLDisplay display) {
  static bool supported = HasExtension(display, "EGL_KHR_fence_sync");
  if (supported) {
    return absl::OkStatus();
  }
  return absl::InternalError("Not supported: EGL_KHR_fence_sync");
}

absl::Status IsEglWaitSyncSupported(EGLDisplay display) {
  static bool supported = HasExtension(display, "EGL_KHR_wait_sync");
  if (supported) {
    return absl::OkStatus();
  }
  return absl::InternalError("Not supported: EGL_KHR_wait_sync");
}

absl::Status CheckEglError(const char* context) {
  const absl::Status status = gl::GetEglError();
  if (status.ok()) {
    return absl::OkStatus();
  }
  return absl::Status(status.code(),
                      std::string(status.message()) + ": " + context);
}

}  // anonymous namespace

ABSL_ATTRIBUTE_NO_SANITIZE_CFI absl::Status EglSync::NewFence(
    EGLDisplay display, EglSync* sync) {
  ABSL_RETURN_IF_ERROR(IsEglFenceSyncSupported(display));
  static auto* egl_create_sync_khr =
      reinterpret_cast<decltype(&eglCreateSyncKHR)>(
          eglGetProcAddress("eglCreateSyncKHR"));
  if (egl_create_sync_khr == nullptr) {
    // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
    return absl::InternalError(
        "Not supported / bad EGL implementation: eglCreateSyncKHR.");
  }
  EGLSyncKHR egl_sync =
      (*egl_create_sync_khr)(display, EGL_SYNC_FENCE_KHR, nullptr);
  ABSL_RETURN_IF_ERROR(CheckEglError("eglCreateSyncKHR"));
  if (egl_sync == EGL_NO_SYNC_KHR) {
    return absl::InternalError("Returned empty KHR EGL sync");
  }
  *sync = EglSync(display, egl_sync);
  return absl::OkStatus();
}

EglSync& EglSync::operator=(EglSync&& sync) {
  if (this != &sync) {
    Invalidate();
    std::swap(sync_, sync.sync_);
    display_ = sync.display_;
  }
  return *this;
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI void EglSync::Invalidate() {
  if (sync_ != EGL_NO_SYNC_KHR) {
    static auto* egl_destroy_sync_khr =
        reinterpret_cast<decltype(&eglDestroySyncKHR)>(
            eglGetProcAddress("eglDestroySyncKHR"));
    // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
    if (IsEglFenceSyncSupported(display_).ok() && egl_destroy_sync_khr) {
      // Note: we're doing nothing when the function pointer is nullptr, or the
      // call returns EGL_FALSE.
      (*egl_destroy_sync_khr)(display_, sync_);
    }
    sync_ = EGL_NO_SYNC_KHR;
  }
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI absl::Status EglSync::ServerWait() {
  ABSL_RETURN_IF_ERROR(IsEglWaitSyncSupported(display_));
  static auto* egl_wait_sync_khr = reinterpret_cast<decltype(&eglWaitSyncKHR)>(
      eglGetProcAddress("eglWaitSyncKHR"));
  if (egl_wait_sync_khr == nullptr) {
    // Needs extension: EGL_KHR_wait_sync
    return absl::InternalError("Not supported: eglWaitSyncKHR.");
  }
  EGLint result = (*egl_wait_sync_khr)(display_, sync_, 0);
  ABSL_RETURN_IF_ERROR(CheckEglError("eglWaitSyncKHR"));
  return result == EGL_TRUE ? absl::OkStatus()
                            : absl::InternalError("eglWaitSync failed");
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI absl::Status EglSync::ClientWait() {
  ABSL_RETURN_IF_ERROR(IsEglFenceSyncSupported(display_));
  static auto* egl_client_wait_sync_khr =
      reinterpret_cast<decltype(&eglClientWaitSyncKHR)>(
          eglGetProcAddress("eglClientWaitSyncKHR"));
  if (egl_client_wait_sync_khr == nullptr) {
    // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
    return absl::InternalError("Not supported: eglClientWaitSyncKHR.");
  }
  // TODO: make it active wait for better performance
  EGLint result = (*egl_client_wait_sync_khr)(
      display_, sync_, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGL_FOREVER_KHR);
  ABSL_RETURN_IF_ERROR(CheckEglError("eglClientWaitSyncKHR"));
  return result == EGL_CONDITION_SATISFIED_KHR
             ? absl::OkStatus()
             : absl::InternalError("eglClientWaitSync failed");
}

}  // namespace cl
}  // namespace ml_drift
