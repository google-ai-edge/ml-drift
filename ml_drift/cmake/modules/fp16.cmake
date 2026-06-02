# Copyright 2025 The ML Drift Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cmake/modules/Findfp16.cmake
include(FetchContent)

FetchContent_Declare(
  FP16
  GIT_REPOSITORY https://github.com/Maratyszcza/FP16.git
  GIT_TAG 98b0a46bce017382a6351a19577ec43a715b6835
)

# Disable tests for FP16 project
set(FP16_BUILD_TESTS OFF)
set(BUILD_TESTING OFF)

FetchContent_MakeAvailable(FP16)

# Restore BUILD_TESTING to its original value if needed
# This depends on whether the main project enables tests. Default is ON.
set(BUILD_TESTING ON)

message(STATUS "FP16_SOURCE_DIR: ${fp16_SOURCE_DIR}")

if(NOT TARGET FP16)
  add_library(FP16 INTERFACE)
  target_include_directories(FP16 SYSTEM INTERFACE
    "${fp16_SOURCE_DIR}/include"
  )
  # Create an alias for compatibility
  add_library(FP16::FP16 ALIAS FP16)
endif()
