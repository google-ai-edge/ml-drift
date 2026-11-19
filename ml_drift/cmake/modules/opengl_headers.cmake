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

# cmake/modules/opengl_headers.cmake
include(FetchContent)

FetchContent_Declare(
  opengl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenGL-Registry.git
  GIT_TAG 1cdd228e34966dd6b95bd203e9f84faba0f371a1
)

FetchContent_MakeAvailable(opengl_headers)

message(STATUS "opengl_headers_SOURCE_DIR: ${opengl_headers_SOURCE_DIR}")

if(NOT TARGET opengl_headers)
  add_library(opengl_headers INTERFACE)
  target_include_directories(opengl_headers SYSTEM INTERFACE
    "${opengl_headers_SOURCE_DIR}/api"
  )
  # Alias for compatibility
  add_library(opengl_headers::OpenGL ALIAS opengl_headers)
endif()
