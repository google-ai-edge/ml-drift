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
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Set variables to a non-empty value to satisfy find_package_handle_standard_args
# The actual targets are available globally through FetchContent_MakeAvailable
set(GTEST_INCLUDE_DIR ${googletest_SOURCE_DIR}/googletest/include)
set(GMOCK_INCLUDE_DIR ${googletest_SOURCE_DIR}/googlemock/include)
set(GTEST_LIBRARY gtest)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GMOCK_LIBRARY gmock)
set(GMOCK_MAIN_LIBRARY gmock_main)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTest DEFAULT_MSG
  GTEST_LIBRARY GTEST_INCLUDE_DIR
  GMOCK_LIBRARY GMOCK_INCLUDE_DIR
  GTEST_MAIN_LIBRARY GMOCK_MAIN_LIBRARY)

set(GTest_FOUND TRUE)
