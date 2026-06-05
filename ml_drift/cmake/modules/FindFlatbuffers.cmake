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

find_program(NINJA_EXECUTABLE ninja)

if(NOT TARGET flatbuffers)
  include(FetchContent)
  FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v24.3.25
  )
  set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Disable flatbuffers tests")

  # Build TARGET library
  FetchContent_MakeAvailable(flatbuffers)

  # Build HOST flatc
  set(HOST_TOOLS_DIR ${CMAKE_BINARY_DIR}/host-tools)
  set(HOST_FLATC_EXE ${HOST_TOOLS_DIR}/bin/flatc)

  include(ExternalProject)
  ExternalProject_Add(
    flatc_host_tool
    SOURCE_DIR ${flatbuffers_SOURCE_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/_build_flatc_host
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${HOST_TOOLS_DIR}
      -DFLATBUFFERS_BUILD_TESTS=OFF
      -DFLATBUFFERS_BUILD_SHAREDLIB=OFF
      -DCMAKE_MAKE_PROGRAM=${NINJA_EXECUTABLE}
      # Force external build to use the host compilers
      -UCMAKE_TOOLCHAIN_FILE
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    UPDATE_COMMAND ""
    TEST_COMMAND ""
  )

  # Set the variable for mld_generate_flatbuffers_cpp function to use
  set(FLATBUFFERS_FLATC_EXECUTABLE ${HOST_FLATC_EXE}
      CACHE FILEPATH "Path to host flatc built by external project")

endif()
