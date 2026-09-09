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

# cmake/modules/dawn.cmake
include(FindPackageHandleStandardArgs)

if(EMSCRIPTEN)
  if(NOT TARGET dawn)
    add_library(dawn INTERFACE)
    target_compile_options(dawn INTERFACE "-sUSE_WEBGPU=1")
    target_link_options(dawn INTERFACE "-sUSE_WEBGPU=1")
  endif()
  set(DAWN_FOUND TRUE)
else()
  # Determine expected Chromium version from build_libdawn.sh
  set(_EXPECTED_CHROMIUM_VERSION "147.0.7714.2")
  find_file(_BUILD_LIBDAWN_SCRIPT
    NAMES build_libdawn.sh
    PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dawn"
      "${CMAKE_SOURCE_DIR}/third_party/dawn"
      "${CMAKE_CURRENT_SOURCE_DIR}/release/dawn_scripts"
      "${CMAKE_SOURCE_DIR}/release/dawn_scripts"
      "${ML_DRIFT_PARENT_DIR}/third_party/dawn"
      "${ML_DRIFT_PARENT_DIR}/release/dawn_scripts"
    NO_DEFAULT_PATH
  )
  if(_BUILD_LIBDAWN_SCRIPT)
    file(READ "${_BUILD_LIBDAWN_SCRIPT}" _BUILD_LIBDAWN_CONTENT)
    if(_BUILD_LIBDAWN_CONTENT MATCHES "CHROMIUM_VERSION=([0-9.]+)")
      set(_EXPECTED_CHROMIUM_VERSION "${CMAKE_MATCH_1}")
    endif()
  endif()

  # Mimic cached_dawn.sh from Kokoro / Bazel workflow:
  # Check for cached Dawn tarball at KOKORO_GFILE_DIR and extract to /tmp/dawn if needed
  set(_DAWN_TARBALL "$ENV{KOKORO_GFILE_DIR}/webgpu-dawn-binaries.tar.gz")
  set(_CACHED_CHROMIUM_VERSION "")
  if(EXISTS "/tmp/dawn/webgpu-dawn-binaries/chromium_version.txt")
    file(READ "/tmp/dawn/webgpu-dawn-binaries/chromium_version.txt" _CACHED_CHROMIUM_VERSION)
    string(STRIP "${_CACHED_CHROMIUM_VERSION}" _CACHED_CHROMIUM_VERSION)
  endif()

  # If not already extracted or version does not match, try extracting from the cached tarball
  if(NOT _CACHED_CHROMIUM_VERSION STREQUAL _EXPECTED_CHROMIUM_VERSION OR
     NOT EXISTS "/tmp/dawn/webgpu-dawn-binaries/out/latest/include/webgpu/webgpu_cpp.h")
    if(DEFINED ENV{KOKORO_GFILE_DIR} AND EXISTS "${_DAWN_TARBALL}")
      message(STATUS "Found cached Dawn tarball at ${_DAWN_TARBALL}. Extracting to /tmp/dawn...")
      file(MAKE_DIRECTORY "/tmp/dawn")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar -xf "${_DAWN_TARBALL}"
        WORKING_DIRECTORY "/tmp/dawn"
        RESULT_VARIABLE _TAR_RESULT
      )
      if(EXISTS "/tmp/dawn/webgpu-dawn-binaries/chromium_version.txt")
        file(READ "/tmp/dawn/webgpu-dawn-binaries/chromium_version.txt" _CACHED_CHROMIUM_VERSION)
        string(STRIP "${_CACHED_CHROMIUM_VERSION}" _CACHED_CHROMIUM_VERSION)
      endif()
    endif()
  endif()

  # Check if cached chromium version matches current dawn version
  if(_CACHED_CHROMIUM_VERSION STREQUAL _EXPECTED_CHROMIUM_VERSION AND
     EXISTS "/tmp/dawn/webgpu-dawn-binaries/out/latest/include/webgpu/webgpu_cpp.h")
    message(STATUS "Cached chromium version (${_CACHED_CHROMIUM_VERSION}) matches current dawn version. Using cached dawn binary.")
  else()
    if(NOT _CACHED_CHROMIUM_VERSION)
      message(STATUS "Cached dawn version not found. Falling back to building/fetching dawn from jspanchu repository.")
    else()
      message(STATUS "Cached dawn version (${_CACHED_CHROMIUM_VERSION}) does not match current (${_EXPECTED_CHROMIUM_VERSION}). Rebuilding.")
    endif()

    # Fallback: run build_libdawn.sh to build dawn binaries from jspanchu repository (matching bazel)
    if(_BUILD_LIBDAWN_SCRIPT)
      file(REMOVE_RECURSE "/tmp/dawn")
      message(STATUS "Executing ${_BUILD_LIBDAWN_SCRIPT}...")
      execute_process(
        COMMAND bash "${_BUILD_LIBDAWN_SCRIPT}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE _BUILD_RESULT
      )
    endif()

    # Fallback: fetch jspanchu/webgpu-dawn-binaries via FetchContent if out/latest is still missing
    if(NOT EXISTS "/tmp/dawn/webgpu-dawn-binaries/out/latest/include/webgpu/webgpu_cpp.h")
      include(FetchContent)
      FetchContent_Declare(
        webgpu_dawn_binaries
        GIT_REPOSITORY https://github.com/jspanchu/webgpu-dawn-binaries.git
        GIT_TAG main
        GIT_SHALLOW TRUE
      )
      FetchContent_GetProperties(webgpu_dawn_binaries)
      if(NOT webgpu_dawn_binaries_POPULATED)
        FetchContent_Populate(webgpu_dawn_binaries)
      endif()
    endif()
  endif()

  set(DAWN_SEARCH_PATHS
    ${DAWN_DIR}
    ${DAWN_ROOT}
    ${Dawn_DIR}
    ${Dawn_ROOT}
    $ENV{DAWN_DIR}
    $ENV{DAWN_ROOT}
    "/tmp/dawn/webgpu-dawn-binaries/out/latest"
    "/tmp/dawn/webgpu-dawn-binaries"
    "${webgpu_dawn_binaries_SOURCE_DIR}/out/latest"
    "${webgpu_dawn_binaries_SOURCE_DIR}"
    "/tmp/dawn/out/latest"
    "/tmp/dawn"
    "${ML_DRIFT_PARENT_DIR}/third_party/dawn"
    "${CMAKE_SOURCE_DIR}/third_party/dawn"
    "${CMAKE_SOURCE_DIR}/../third_party/dawn"
    "${CMAKE_PREFIX_PATH}"
    "/usr/local"
    "/usr"
  )

  find_path(DAWN_INCLUDE_DIR
    NAMES
      webgpu/webgpu_cpp.h
      webgpu/webgpu.h
      dawn/webgpu.h
    PATHS ${DAWN_SEARCH_PATHS}
    PATH_SUFFIXES
      include
      third_party/dawn/include
      out/latest/include
      webgpu-dawn-binaries/out/latest/include
  )

  find_library(DAWN_LIBRARY
    NAMES
      dawn
      libdawn
      webgpu_dawn
    PATHS ${DAWN_SEARCH_PATHS}
    PATH_SUFFIXES
      lib
      lib64
      out/latest/lib
      webgpu-dawn-binaries/out/latest/lib
  )

  if(DAWN_INCLUDE_DIR AND DAWN_LIBRARY)
    set(DAWN_FOUND TRUE)
    if(NOT TARGET dawn)
      add_library(dawn UNKNOWN IMPORTED)
      set_target_properties(dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${DAWN_INCLUDE_DIR}"
      )
    endif()
  elseif(DAWN_INCLUDE_DIR)
    set(DAWN_FOUND TRUE)
    if(NOT TARGET dawn)
      add_library(dawn INTERFACE IMPORTED)
      set_target_properties(dawn PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${DAWN_INCLUDE_DIR}"
      )
    endif()
  elseif(EXISTS "${webgpu_dawn_binaries_SOURCE_DIR}/out/latest/include")
    set(DAWN_FOUND TRUE)
    if(NOT TARGET dawn)
      add_library(dawn INTERFACE)
      target_include_directories(dawn INTERFACE
        "${webgpu_dawn_binaries_SOURCE_DIR}/out/latest/include"
      )
    endif()
  endif()
endif()

if(NOT TARGET Dawn::dawn AND TARGET dawn)
  add_library(Dawn::dawn ALIAS dawn)
endif()

if(NOT TARGET webgpu_dawn AND TARGET dawn)
  add_library(webgpu_dawn ALIAS dawn)
endif()

find_package_handle_standard_args(dawn DEFAULT_MSG DAWN_FOUND)
