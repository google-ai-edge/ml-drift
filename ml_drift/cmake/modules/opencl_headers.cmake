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

# cmake/modules/Findopencl_headers.cmake
include(FetchContent)

FetchContent_Declare(
  opencl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
  GIT_TAG v2024.05.08
)

FetchContent_MakeAvailable(opencl_headers)

message(STATUS "opencl_headers_SOURCE_DIR: ${opencl_headers_SOURCE_DIR}")

if(NOT TARGET opencl_headers)
  add_library(opencl_headers INTERFACE)
  target_include_directories(opencl_headers SYSTEM INTERFACE
    "${opencl_headers_SOURCE_DIR}"
  )
  # Alias for compatibility
  add_library(opencl_headers::OpenCL ALIAS opencl_headers)
endif()
