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

#include "ml_drift/gl/request_gpu_info.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/ascii.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/gl/gl_errors.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {
AdrenoInfo::OpenGlDriverVersion GeAdrenoDriverVersion(
    const std::string& gl_version) {
  size_t pos = gl_version.find("V@");
  if (pos == std::string::npos) {
    return {0, 0, 0};
  }
  pos += 2;
  if (pos >= gl_version.length()) {
    return {0, 0, 0};
  }
  std::array<int, 3> versions{0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    bool has_next = false;
    for (; pos < gl_version.length(); ++pos) {
      if (gl_version[pos] == '.') {
        has_next = true;
        pos += 1;
        break;
      } else if (absl::ascii_isdigit(gl_version[pos])) {
        versions[i] = (versions[i] * 10 + gl_version[pos] - '0');
      } else {
        break;
      }
    }
    if (!has_next) {
      break;
    }
  }
  return {versions[0], versions[1], versions[2]};
}

absl::Status RequestOpenGlInfo(OpenGlInfo* gl_info) {
  const GLubyte* renderer_name = glGetString(GL_RENDERER);
  if (renderer_name) {
    gl_info->renderer_name = reinterpret_cast<const char*>(renderer_name);
  }

  const GLubyte* vendor_name = glGetString(GL_VENDOR);
  if (vendor_name) {
    gl_info->vendor_name = reinterpret_cast<const char*>(vendor_name);
  }

  const GLubyte* version_name = glGetString(GL_VERSION);
  if (version_name) {
    gl_info->version = reinterpret_cast<const char*>(version_name);
  }

  glGetIntegerv(GL_MAJOR_VERSION, &gl_info->major_version);
  glGetIntegerv(GL_MINOR_VERSION, &gl_info->minor_version);

  return absl::OkStatus();
}

absl::Status RequestGpuInfo(GpuInfo* gpu_info) {
  GpuInfo info;
  ABSL_RETURN_IF_ERROR(RequestOpenGlInfo(&info.opengl_info));

  GetGpuInfoFromDeviceDescription(info.opengl_info.renderer_name,
                                  GpuApi::kOpenGl, &info);

  if (info.IsAdreno()) {
    info.adreno_info.opengl_driver_version =
        GeAdrenoDriverVersion(info.opengl_info.version);
  }

  GLint extensions_count;
  glGetIntegerv(GL_NUM_EXTENSIONS, &extensions_count);
  info.opengl_info.extensions.resize(extensions_count);
  for (int i = 0; i < extensions_count; ++i) {
    info.opengl_info.extensions[i] = std::string(
        reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i)));
  }
  if (info.IsPowerVR() && info.SupportsExtension("GL_KHR_shader_subgroup")) {
    info.opengl_info.extensions.push_back("ucl_wave_memory");
  }
  // The maximum number of active shader storage blocks that may be accessed by
  // a compute shader.
  glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
                &info.opengl_info.max_ssbo_bindings);
  // The maximum supported number of image variables in compute shaders.
  glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS,
                &info.opengl_info.max_image_bindings);
  // The maximum supported texture image units that can be used to access
  // texture maps from the compute shader.
  glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS,
                &info.opengl_info.max_texture_bindings);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0,
                  &info.opengl_info.max_compute_work_group_size_x);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1,
                  &info.opengl_info.max_compute_work_group_size_y);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2,
                  &info.opengl_info.max_compute_work_group_size_z);
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                &info.opengl_info.max_work_group_invocations);
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &info.opengl_info.max_texture_size);
  glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS,
                &info.opengl_info.max_array_texture_layers);
  glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS,
                &info.opengl_info.max_fragment_uniform_vec4_count);
  glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE,
                &info.opengl_info.max_renderbuffer_size);
  glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,
                &info.opengl_info.max_shader_storage_block_size);
  {  // remove GL_KHR_shader_subgroup if compute stage not supported.
    auto it =
        std::find(info.opengl_info.extensions.begin(),
                  info.opengl_info.extensions.end(), "GL_KHR_shader_subgroup");
    if (it != info.opengl_info.extensions.end()) {
      GLint supported_stages;
      glGetIntegerv(GL_SUBGROUP_SUPPORTED_STAGES_KHR, &supported_stages);
      if (!(supported_stages & GL_COMPUTE_SHADER_BIT)) {
        info.opengl_info.extensions.erase(it);
      }
    }
  }
  if (info.SupportsExtension("GL_KHR_shader_subgroup")) {
    auto& subgroup_info = info.opengl_info.subgroup_info;
    glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroup_info.size);
    GLint supported_ops;
    glGetIntegerv(GL_SUBGROUP_SUPPORTED_FEATURES_KHR, &supported_ops);
    subgroup_info.basic = supported_ops & GL_SUBGROUP_FEATURE_BASIC_BIT_KHR;
    subgroup_info.vote = supported_ops & GL_SUBGROUP_FEATURE_VOTE_BIT_KHR;
    subgroup_info.arithmetic =
        supported_ops & GL_SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR;
    subgroup_info.ballot = supported_ops & GL_SUBGROUP_FEATURE_BALLOT_BIT_KHR;
    subgroup_info.shuffle = supported_ops & GL_SUBGROUP_FEATURE_SHUFFLE_BIT_KHR;
    subgroup_info.shuffle_relative =
        supported_ops & GL_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR;
    subgroup_info.clustered =
        supported_ops & GL_SUBGROUP_FEATURE_CLUSTERED_BIT_KHR;
    subgroup_info.quad = supported_ops & GL_SUBGROUP_FEATURE_QUAD_BIT_KHR;
  }
  GLint max_viewport_dims[2];
  glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_viewport_dims);
  info.opengl_info.max_viewport_width = max_viewport_dims[0];
  info.opengl_info.max_viewport_height = max_viewport_dims[1];
  GLint max_color_atttachments;
  glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &max_color_atttachments);
  GLint max_draw_buffers;
  glGetIntegerv(GL_MAX_DRAW_BUFFERS, &max_draw_buffers);
  info.opengl_info.max_color_atttachments =
      std::min(max_color_atttachments, max_draw_buffers);
  ABSL_RETURN_IF_ERROR(GetOpenGlErrors());
  *gpu_info = info;
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace ml_drift
