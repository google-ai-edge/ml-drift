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

include(FetchContent)

FetchContent_Declare(
  abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20260817.0
)

# Set options to avoid building tests if GTest is not found or is a different version
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "")
set(ABSL_USE_SYSTEM_GTEST ON CACHE BOOL "Use system-provided GoogleTest")

# TODO(dlho): conditionalize these
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

FetchContent_GetProperties(abseil)
if(NOT abseil_POPULATED)
  FetchContent_Populate(abseil)
  add_subdirectory(${abseil_SOURCE_DIR} ${abseil_BINARY_DIR})
endif()
