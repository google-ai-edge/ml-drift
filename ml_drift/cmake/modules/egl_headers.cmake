# Copyright 2026 The ML Drift Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cmake/modules/egl_headers.cmake
include(FetchContent)

FetchContent_Declare(
  egl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/EGL-Registry.git
  GIT_TAG db3425b8246136faccb5e2782b5694960bd6edf1
)

FetchContent_MakeAvailable(egl_headers)

message(STATUS "egl_headers_SOURCE_DIR: ${egl_headers_SOURCE_DIR}")

if(NOT TARGET egl_headers)
  add_library(egl_headers INTERFACE)
  target_include_directories(egl_headers SYSTEM INTERFACE
    "${egl_headers_SOURCE_DIR}/api"
  )
  # Alias for compatibility
  add_library(egl_headers::EGL ALIAS egl_headers)
endif()
