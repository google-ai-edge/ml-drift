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

add_library(mld_cl_opencl_wrapper
    opencl_wrapper.cc
    opencl_wrapper.h
)
target_link_libraries(mld_cl_opencl_wrapper
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    opencl_headers
  PRIVATE
    absl::strings
)
target_include_directories(mld_cl_opencl_wrapper
    PUBLIC
        $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)

# --- Platform-Specific Linker Options ---

# Handle Android-specific linker flags (-ldl, -lm)
if(ANDROID)
    target_link_libraries(mld_cl_opencl_wrapper
      PRIVATE
        dl
        m
    )
else()
    target_link_libraries(mld_cl_opencl_wrapper
      PRIVATE
        ${CMAKE_DL_LIBS}
    )
    target_link_options(mld_cl_opencl_wrapper PRIVATE "-rdynamic")
endif()

add_library(mld_cl_environment environment.cc environment.h)
target_include_directories(mld_cl_environment PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_compile_definitions(mld_cl_environment PUBLIC ML_DRIFT_ENABLE_OPENCL)
target_link_libraries(mld_cl_environment
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_cl_command_queue
    mld_cl_cl_context
    mld_cl_cl_device
    mld_cl_cl_program_cache
    mld_cl_util_types
    mld_common_gpu_info
    mld_common_precision
    mld_common_task_tensor_desc
  PRIVATE
    XNNPACK
)

add_library(mld_cl_cl_command_queue cl_command_queue.cc cl_command_queue.h)
target_include_directories(mld_cl_cl_command_queue PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_command_queue
  PUBLIC
    absl::status
    mld_cl_cl_context
    mld_cl_cl_event
    mld_cl_cl_kernel
  PRIVATE
    mld_cl_util
)

add_library(mld_cl_cl_context cl_context.cc cl_context.h)
target_include_directories(mld_cl_cl_context PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_context
  PUBLIC
    absl::status
    mld_cl_cl_device
  PRIVATE
    mld_cl_util
)

add_library(mld_cl_cl_device cl_device.cc cl_device.h)
target_include_directories(mld_cl_cl_device PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_device
  PUBLIC
    absl::status
    mld_cl_opencl_wrapper
    mld_cl_util
    mld_common_gpu_info
)

# --- Flatbuffer generation for compiled_program_cache.fbs ---
mld_generate_flatbuffers_cpp(mld_cl_compiled_program_cache_cc_fbs
  "${CMAKE_CURRENT_SOURCE_DIR}/compiled_program_cache.fbs"
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ml_drift/cl"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS flatc_host_tool
)
# --- End Flatbuffer generation ---

add_library(mld_cl_cl_program_cache program_cache.cc program_cache.h)
target_include_directories(mld_cl_cl_program_cache PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(mld_cl_cl_program_cache
  PUBLIC
    absl::flat_hash_map
    mld_cl_cl_context
    mld_cl_cl_program
  PRIVATE
    farmhash
    flatbuffers
    mld_cl_compiled_program_cache_cc_fbs
)

add_library(mld_cl_util_types INTERFACE
  util_types.h
)
target_include_directories(mld_cl_util_types INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)

# --- Flatbuffer generation for serialization.fbs ---
mld_generate_flatbuffers_cpp(mld_cl_serialization_cc_fbs
  "${CMAKE_CURRENT_SOURCE_DIR}/serialization.fbs"
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ml_drift/cl"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/gpu_model.fbs"
    flatc_host_tool
)
# --- End Flatbuffer generation ---

add_library(mld_cl_inference_context inference_context.cc inference_context.h)
target_include_directories(mld_cl_inference_context PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(mld_cl_inference_context
  PUBLIC
    flatbuffers
    mld_cl_buffer
    mld_cl_cl_command_buffer
    mld_cl_cl_operation
    mld_cl_command_buffer
    mld_cl_default_qcom_command_buffer
    mld_cl_environment
    mld_cl_memory_manager
    mld_cl_serialization_cc_fbs
    mld_common_gpu_model
    mld_common_gpu_model_cc_fbs
    mld_common_model
  PRIVATE
    mld_common_merge_nodes
    mld_common_operations
)



add_library(mld_cl_buffer buffer.cc buffer.h)
target_include_directories(mld_cl_buffer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_buffer
  PUBLIC
    absl::span
    mld_cl_cl_command_queue
    mld_common_task_buffer_desc
)

add_library(mld_cl_memory_manager memory_manager.cc memory_manager.h)
target_include_directories(mld_cl_memory_manager PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_cl_memory_manager
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_buffer
    mld_cl_tensor
    mld_common_gpu_model
  PRIVATE
    mld_common_memory_management
)

add_library(mld_cl_cl_command_buffer cl_command_buffer.cc cl_command_buffer.h)
target_include_directories(mld_cl_cl_command_buffer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_command_buffer
  PUBLIC
    absl::status
    mld_cl_cl_command_queue
    mld_cl_cl_operation
    mld_cl_command_buffer
)

add_library(mld_cl_command_buffer command_buffer.h)
target_include_directories(mld_cl_command_buffer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_command_buffer
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_cl_command_queue
    mld_cl_cl_event
    mld_cl_cl_operation
)

add_library(mld_cl_cl_event cl_event.cc cl_event.h)
target_include_directories(mld_cl_cl_event PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_event
  PUBLIC
    mld_cl_opencl_wrapper
)

add_library(mld_cl_cl_kernel cl_kernel.cc cl_kernel.h)
target_include_directories(mld_cl_cl_kernel PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_kernel
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_cl_cl_device
    mld_cl_opencl_wrapper
    mld_cl_util
    mld_common_util
)

add_library(mld_cl_cl_program cl_program.cc cl_program.h)
target_include_directories(mld_cl_cl_program PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_program
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_cl_cl_device
    mld_cl_cl_kernel
    mld_cl_opencl_wrapper
)

add_library(mld_cl_util util.cc util.h)
target_include_directories(mld_cl_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_util
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_cl_cl_device
    mld_cl_cl_image_format
    mld_cl_opencl_wrapper
    mld_common_data_type
    mld_common_gpu_info
    mld_common_types
)

add_library(mld_cl_cl_image_format cl_image_format.cc cl_image_format.h)
target_include_directories(mld_cl_cl_image_format PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_image_format
  PUBLIC
    mld_cl_opencl_wrapper
    mld_common_data_type
)

add_library(mld_cl_tensor tensor.cc tensor.h)
target_include_directories(mld_cl_tensor PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_tensor
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_buffer
    mld_cl_cl_command_queue
    mld_cl_cl_context
    mld_cl_opencl_wrapper
    mld_cl_util
    mld_common_task_tensor_desc
    mld_common_types
)

add_library(mld_cl_cl_arguments cl_arguments.cc cl_arguments.h)
target_include_directories(mld_cl_cl_arguments PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_arguments
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_cl_kernel
    mld_cl_opencl_wrapper
    mld_cl_qcom_thin_filter
    mld_cl_tensor
    mld_common_task_arguments
    mld_common_task_gpu_object_desc
    mld_common_util
)

add_library(mld_cl_qcom_thin_filter qcom_thin_filter.cc qcom_thin_filter.h
  ../common/task/qcom_thin_filter_desc.cc)
target_include_directories(mld_cl_qcom_thin_filter PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_qcom_thin_filter
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_buffer
    mld_cl_cl_context
    mld_common_task_gpu_object_desc
)

add_library(mld_cl_cl_operation cl_operation.cc cl_operation.h)
target_include_directories(mld_cl_cl_operation PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_cl_operation
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_cl_arguments
    mld_cl_cl_context
    mld_cl_cl_device
    mld_cl_cl_kernel
    mld_cl_cl_program_cache
    mld_cl_tensor
    mld_common_gpu_info
    mld_common_task_gpu_operation
  PRIVATE
    mld_common_task_util
)

add_library(cl_test INTERFACE)
target_include_directories(cl_test INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(cl_test
  INTERFACE
    gmock
    gtest
    mld_cl_default_util
    mld_cl_environment
    mld_cl_opencl_wrapper
)

add_library(gpu_object INTERFACE)
target_include_directories(gpu_object INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(gpu_object
  INTERFACE
    absl::status
    absl::status_macros
    absl::statusor
    mld_cl_opencl_wrapper
    mld_common_task_gpu_object_desc
)

add_executable(buffer_test buffer_test.cc)
target_link_libraries(buffer_test
  PRIVATE
    absl::span
    cl_test
    gtest_main
    mld_cl_buffer
    mld_common_types
)

add_executable(cl_arguments_test cl_arguments_test.cc)
target_link_libraries(cl_arguments_test
  PRIVATE
    absl::status
    absl::status_macros
    absl::statusor
    cl_test
    gtest_main
    mld_cl_cl_arguments
    mld_common_task_arguments
)

add_executable(inference_context_test inference_context_test.cc)
target_link_libraries(inference_context_test
  PRIVATE
    absl::log
    absl::status
    absl::status_macros
    absl::statusor
    cl_test
    gtest_main
    mld_cl_cl_arguments
    mld_cl_inference_context
    mld_cl_tensor
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_precision
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_tensor
)

if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(buffer_test)
    gtest_discover_tests(cl_arguments_test)
    gtest_discover_tests(inference_context_test)
endif()

add_subdirectory(default)
add_subdirectory(testing)
