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

#include "ml_drift/gl/egl_environment.h"

#include <memory>

#include "xnnpack.h"  // from @XNNPACK

#ifndef __ANDROID__
#include "absl/debugging/leak_check.h"
#endif  // !__ANDROID__
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/gl/gl_call.h"
#include "ml_drift/gl/request_gpu_info.h"

namespace ml_drift {
namespace gl {
namespace {

// TODO(akulik): detect power management event when all contexts are destroyed
// and OpenGL ES is reinitialized. See eglMakeCurrent

absl::Status InitDisplay(EGLDisplay* egl_display) {
  {
#ifndef __ANDROID__
    // (At least) NVidia desktop EGL implementation leaks a thread-local object
    // from loadEGLExternalPlatform during the eglGetDisplay call. Ignore that
    // leak.
    absl::LeakCheckDisabler disabler;
#endif  // !__ANDROID__
    ABSL_RETURN_IF_ERROR(
        ML_DRIFT_CALL_EGL(eglGetDisplay, egl_display, EGL_DEFAULT_DISPLAY));
  }
  if (*egl_display == EGL_NO_DISPLAY) {
    return absl::UnavailableError("eglGetDisplay returned nullptr");
  }
  bool is_initialized;
  ABSL_RETURN_IF_ERROR(ML_DRIFT_CALL_EGL(eglInitialize, &is_initialized,
                                         *egl_display, nullptr, nullptr));
  if (!is_initialized) {
    return absl::InternalError("No EGL error, but eglInitialize failed");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status EglEnvironment::NewEglEnvironment(
    std::unique_ptr<EglEnvironment>* egl_environment) {
  *egl_environment = std::make_unique<EglEnvironment>();
  ABSL_RETURN_IF_ERROR((*egl_environment)->Init());
  return absl::OkStatus();
}

EglEnvironment::~EglEnvironment() {
  if (dummy_framebuffer_ != GL_INVALID_INDEX) {
    glDeleteFramebuffers(1, &dummy_framebuffer_);
  }
  if (dummy_texture_ != GL_INVALID_INDEX) {
    glDeleteTextures(1, &dummy_texture_);
  }
}

absl::Status EglEnvironment::Init() {
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return absl::InternalError("Failed to initialize XNNPACK.");
  }
  bool is_bound;
  ABSL_RETURN_IF_ERROR(
      ML_DRIFT_CALL_EGL(eglBindAPI, &is_bound, EGL_OPENGL_ES_API));
  if (!is_bound) {
    return absl::InternalError("No EGL error, but eglBindAPI failed");
  }

  // Re-use context and display if it was created on this thread.
  if (eglGetCurrentContext() != EGL_NO_CONTEXT) {
    display_ = eglGetCurrentDisplay();
    context_ =
        EglContext(eglGetCurrentContext(), display_, EGL_NO_CONFIG_KHR, false);
  } else {
    ABSL_RETURN_IF_ERROR(InitDisplay(&display_));

    absl::Status status = InitConfiglessContext();
    if (!status.ok()) {
      status = InitSurfacelessContext();
    }
    if (!status.ok()) {
      status = InitPBufferContext();
    }
    if (!status.ok()) {
      return status;
    }
  }

  if (gpu_info_.vendor == GpuVendor::kUnknown) {
    ABSL_RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
  }
  // TODO(akulik): when do we need ForceSyncTurning?
  ForceSyncTurning();
  return absl::OkStatus();
}

absl::Status EglEnvironment::InitConfiglessContext() {
  ABSL_RETURN_IF_ERROR(
      CreateConfiglessContext(display_, EGL_NO_CONTEXT, &context_));
  return context_.MakeCurrentSurfaceless();
}

absl::Status EglEnvironment::InitSurfacelessContext() {
  ABSL_RETURN_IF_ERROR(
      CreateSurfacelessContext(display_, EGL_NO_CONTEXT, &context_));
  ABSL_RETURN_IF_ERROR(context_.MakeCurrentSurfaceless());

  // PowerVR support EGL_KHR_surfaceless_context, but glFenceSync crashes on
  // PowerVR when it is surface-less.
  ABSL_RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
  if (gpu_info_.IsPowerVR()) {
    return absl::UnavailableError(
        "Surface-less context is not properly supported on powervr.");
  }
  return absl::OkStatus();
}

absl::Status EglEnvironment::InitPBufferContext() {
  ABSL_RETURN_IF_ERROR(
      CreatePBufferContext(display_, EGL_NO_CONTEXT, &context_));
  ABSL_RETURN_IF_ERROR(CreatePbufferRGBSurface(context_.config(), display_, 1,
                                               1, &surface_read_));
  ABSL_RETURN_IF_ERROR(CreatePbufferRGBSurface(context_.config(), display_, 1,
                                               1, &surface_draw_));
  return context_.MakeCurrent(surface_read_.surface(), surface_draw_.surface());
}

void EglEnvironment::ForceSyncTurning() {
  glGenFramebuffers(1, &dummy_framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, dummy_framebuffer_);

  glGenTextures(1, &dummy_texture_);
  glBindTexture(GL_TEXTURE_2D, dummy_texture_);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         dummy_texture_, 0);

  GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, draw_buffers);

  glViewport(0, 0, 4, 4);
  glClear(GL_COLOR_BUFFER_BIT);
}

}  // namespace gl
}  // namespace ml_drift
