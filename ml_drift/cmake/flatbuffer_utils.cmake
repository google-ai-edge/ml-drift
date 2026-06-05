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

include(CMakeParseArguments)

function(mld_generate_flatbuffers_cpp TARGET_NAME FBS_FILE)
  set(OPTIONS)
  set(ONE_VALUE_KEYWORDS OUTPUT_DIR INCLUDE_DIR)
  set(MULTI_VALUE_KEYWORDS INCLUDE_DIRS DEPENDS)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_KEYWORDS}" "${MULTI_VALUE_KEYWORDS}" ${ARGN})

  if(NOT ARG_OUTPUT_DIR)
    set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  get_filename_component(FBS_NAME "${FBS_FILE}" NAME_WE)
  set(GENERATED_HEADER "${ARG_OUTPUT_DIR}/${FBS_NAME}_generated.h")

  set(FLATC_COMMAND $<TARGET_FILE:flatc>)

  # This variable is set by the ExternalProject logic and points to the
  # host-runnable flatc
  set(FLATC_COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE})
  if(NOT FLATC_COMMAND)
    message(FATAL_ERROR "FLATBUFFERS_FLATC_EXECUTABLE is not set! "
      "The host-build logic (ExternalProject_Add) is missing.")
  endif()

  set(FLATC_ARGS --cpp --scoped-enums -o "${ARG_OUTPUT_DIR}")

  if(ARG_INCLUDE_DIRS)
    foreach(DIR ${ARG_INCLUDE_DIRS})
      list(APPEND FLATC_ARGS -I "${DIR}")
    endforeach()
  endif()

  list(APPEND FLATC_ARGS "${FBS_FILE}")

  add_custom_command(
    OUTPUT "${GENERATED_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
    COMMAND ${FLATC_COMMAND} ${FLATC_ARGS}
    DEPENDS "${FBS_FILE}" ${ARG_DEPENDS}
    COMMENT "Generating C++ headers from ${FBS_FILE}"
  )

  set(GEN_TARGET_NAME "${TARGET_NAME}_gen")
  add_custom_target(${GEN_TARGET_NAME}
    DEPENDS "${GENERATED_HEADER}"
  )

  add_library(${TARGET_NAME} INTERFACE)
  target_include_directories(${TARGET_NAME} INTERFACE "${ARG_OUTPUT_DIR}")
  if (ARG_INCLUDE_DIR)
    target_include_directories(${TARGET_NAME} INTERFACE "${ARG_INCLUDE_DIR}")
  endif()
  add_dependencies(${TARGET_NAME} ${GEN_TARGET_NAME})

endfunction()
