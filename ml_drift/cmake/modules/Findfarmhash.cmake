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
include(FindPackageHandleStandardArgs)

FetchContent_Declare(
  farmhash
  GIT_REPOSITORY https://github.com/google/farmhash.git
  GIT_TAG 0d859a811870d10f53a594927d0d0b97573ad06d
)

FetchContent_GetProperties(farmhash)
if(NOT farmhash_POPULATED)
  FetchContent_Populate(farmhash)
  add_library(farmhash STATIC
    ${farmhash_SOURCE_DIR}/src/farmhash.h
    ${farmhash_SOURCE_DIR}/src/farmhash.cc
  )
  target_include_directories(farmhash PUBLIC ${farmhash_SOURCE_DIR}/src)
  set(FARMHASH_FOUND TRUE)
else()
  if(NOT TARGET farmhash)
      message(FATAL_ERROR "farmhash was populated but target not found")
  endif()
  set(FARMHASH_FOUND TRUE)
endif()

find_package_handle_standard_args(farmhash DEFAULT_MSG FARMHASH_FOUND)

# Alias for compatibility if projects expect farmhash::farmhash
if(TARGET farmhash AND NOT TARGET farmhash::farmhash)
  add_library(farmhash::farmhash ALIAS farmhash)
endif()
