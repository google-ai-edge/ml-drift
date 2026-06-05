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

cmake_minimum_required(VERSION 3.16)

project(MLDrift CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO(dlho): actually conditionalize this
option(MLD_BUILD_SAMPLES "Build sample applications" OFF)

# --- External Dependencies ---

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH}
)

set(ML_DRIFT_PARENT_DIR ${CMAKE_SOURCE_DIR}/..)

# Prevent Abseil from using a system gtest or building its own unexpectedly
set(ABSL_USE_SYSTEM_GTEST OFF CACHE BOOL "Disable Abseil using system GTest")
set(ABSL_BUILD_TESTING OFF CACHE BOOL "Disable Abseil's tests")
set(BUILD_TESTING OFF CACHE BOOL "Disable tests by default")

# --- Find Packages ---
find_package(GTest REQUIRED)
find_package(absl REQUIRED)
find_package(fp16 REQUIRED)
find_package(opencl_headers REQUIRED)
find_package(Flatbuffers REQUIRED)
find_package(XNNPACK REQUIRED)
find_package(farmhash REQUIRED)
find_package(re2 REQUIRED)

# --- Includes ---
include(tensorflow)
include(cmake/flatbuffer_utils.cmake)
include(GoogleTest)

# --- Add Subdirectories ---
add_subdirectory(common)
add_subdirectory(cl)
add_subdirectory(samples/stable_diffusion)

# --- ML Drift Library ---

# Usage: -D ML_DRIFT_API=...
set(ML_DRIFT_API "cl" CACHE STRING "Select the GPU backend API for ML Drift. Options: cl, mtl, wgpu")

# Input validation
set(VALID_APIS "cl")
list(FIND VALID_APIS "${ML_DRIFT_API}" IS_VALID)
if(IS_VALID EQUAL -1)
    message(FATAL_ERROR "Invalid ML_DRIFT_API: ${ML_DRIFT_API}. Must be one of: ${VALID_APIS}")
endif()

add_library(ml_drift INTERFACE)
target_include_directories(ml_drift INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ml_drift INTERFACE
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_util
    mld_common_model
    mld_common_model_hints
    mld_common_precision
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_task_tensor_desc
)

if(ML_DRIFT_API STREQUAL "cl")
    message(STATUS "ML Drift: Building with OpenCL (cl) backend")
    target_compile_definitions(ml_drift INTERFACE ML_DRIFT_API_CL)
    target_link_libraries(ml_drift INTERFACE
        mld_cl_cl_api_util
        mld_cl_cl_command_buffer
        mld_cl_environment
        mld_cl_inference_context
        mld_cl_tensor
    )
# elseif(ML_DRIFT_API STREQUAL "mtl")
#     message(STATUS "ML Drift: Building with Metal (mtl) backend")
#     target_compile_definitions(ml_drift INTERFACE ML_DRIFT_API_MTL)
#     target_link_libraries(ml_drift INTERFACE
#         mld_metal
#     )
# elseif(ML_DRIFT_API STREQUAL "wgpu")
#     message(STATUS "ML Drift: Building with WebGPU (wgpu) backend")
#     target_compile_definitions(ml_drift INTERFACE ML_DRIFT_API_WGPU)
#     target_link_libraries(ml_drift INTERFACE
#         mld_wgpu
#     )
else()
    message(STATUS "ML Drift: No GPU backend specified.")
endif()

