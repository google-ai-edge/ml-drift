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

add_library(mld_gl_portable INTERFACE
  portable_egl.h
  portable_gl31.h
)
target_include_directories(mld_gl_portable INTERFACE
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_gl_portable
  INTERFACE
    EGL
    egl_headers
    GLESv2
    opengl_headers
)
if(ANDROID)
  target_link_libraries(mld_gl_portable
    INTERFACE
      GLESv3
  )
endif()

add_library(mld_gl_gpu_object INTERFACE gpu_object.h)
target_include_directories(mld_gl_gpu_object INTERFACE
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_gl_gpu_object
  INTERFACE
    absl::status
    mld_common_access_type
    mld_common_data_type
    mld_common_task_gpu_object_desc
    mld_gl_portable
)

add_library(mld_gl_gl_errors gl_errors.cc gl_errors.h)
target_include_directories(mld_gl_gl_errors PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_errors
  PUBLIC
    absl::status
    absl::strings
    mld_gl_portable
)

add_library(mld_gl_gl_call INTERFACE gl_call.h)
target_include_directories(mld_gl_gl_call INTERFACE
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_gl_gl_call
  INTERFACE
    absl::status
    absl::status_macros
    mld_gl_gl_errors
)

add_library(mld_gl_request_gpu_info request_gpu_info.cc request_gpu_info.h)
target_include_directories(mld_gl_request_gpu_info PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_request_gpu_info
  PUBLIC
    absl::status
    absl::status_macros
    absl::strings
    mld_common_gpu_info
    mld_gl_gl_errors
    mld_gl_portable
)

add_library(mld_gl_egl_context egl_context.cc egl_context.h)
target_include_directories(mld_gl_egl_context PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_egl_context
  PUBLIC
    absl::status
    absl::status_macros
    mld_gl_gl_call
    mld_gl_gl_errors
    mld_gl_portable
)

add_library(mld_gl_egl_surface egl_surface.cc egl_surface.h)
target_include_directories(mld_gl_egl_surface PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_egl_surface
  PUBLIC
    absl::status
    absl::status_macros
    mld_gl_gl_call
    mld_gl_gl_errors
    mld_gl_portable
)

add_library(mld_gl_egl_environment egl_environment.cc egl_environment.h)
target_include_directories(mld_gl_egl_environment PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_egl_environment
  PUBLIC
    absl::memory
    absl::status
    absl::status_macros
    mld_gl_egl_context
    mld_gl_egl_surface
    mld_gl_gl_call
    mld_gl_portable
    mld_gl_request_gpu_info
    XNNPACK
)
if(NOT ANDROID)
  target_link_libraries(mld_gl_egl_environment
    PUBLIC
      absl::leak_check
  )
endif()

add_library(mld_gl_gl_buffer gl_buffer.cc gl_buffer.h)
target_include_directories(mld_gl_gl_buffer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_buffer
  PUBLIC
    absl::status
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_gl_gpu_object
)

add_library(mld_gl_gl_texture_helper gl_texture_helper.cc gl_texture_helper.h)
target_include_directories(mld_gl_gl_texture_helper PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_texture_helper
  PUBLIC
    mld_common_data_type
    mld_gl_portable
)

add_library(mld_gl_gl_spatial_tensor gl_spatial_tensor.cc gl_spatial_tensor.h)
target_include_directories(mld_gl_gl_spatial_tensor PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_spatial_tensor
  PUBLIC
    absl::status
    absl::status_macros
    mld_common_access_type
    mld_common_data_type
    mld_common_shape
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_tensor
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_common_types
    mld_common_util
    mld_gl_gl_texture_helper
    mld_gl_gpu_object
    mld_gl_portable
)

add_library(mld_gl_gl_arguments gl_arguments.cc gl_arguments.h)
target_include_directories(mld_gl_gl_arguments PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_arguments
  PUBLIC
    absl::flat_hash_map
    absl::status
    absl::status_macros
    absl::strings
    mld_common_access_type
    mld_common_data_type
    mld_common_gpu_info
    mld_common_task_arguments
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_tensor_desc
    mld_common_types
    mld_common_util
    mld_gl_gl_buffer
    mld_gl_gl_spatial_tensor
    mld_gl_gl_texture_helper
    mld_gl_gpu_object
    mld_gl_portable
)

add_library(mld_gl_gl_shader gl_shader.cc gl_shader.h)
target_include_directories(mld_gl_gl_shader PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_shader
  PUBLIC
    absl::status
    absl::status_macros
    mld_gl_gl_call
    mld_gl_gl_errors
    mld_gl_portable
)

add_library(mld_gl_gl_operation gl_operation.cc gl_operation.h)
target_include_directories(mld_gl_gl_operation PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_operation
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    absl::time
    farmhash
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_task_gpu_operation
    mld_common_task_tuning_type
    mld_common_types
    mld_gl_gl_arguments
    mld_gl_gl_buffer
    mld_gl_gl_shader
    mld_gl_gl_spatial_tensor
    mld_gl_portable
)

add_library(mld_gl_gl_sync gl_sync.cc gl_sync.h)
target_include_directories(mld_gl_gl_sync PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_gl_sync
  PUBLIC
    absl::status
    absl::status_macros
    mld_gl_gl_buffer
    mld_gl_gl_call
    mld_gl_gl_errors
    mld_gl_portable
)

add_library(mld_gl_converter converter.cc converter.h)
target_include_directories(mld_gl_converter PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_converter
  PUBLIC
    absl::status
    absl::status_macros
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernels_conversion
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_types
    mld_gl_gl_buffer
    mld_gl_gl_operation
    mld_gl_gl_spatial_tensor
)

add_library(mld_gl_memory_manager memory_manager.cc memory_manager.h)
target_include_directories(mld_gl_memory_manager PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_memory_manager
  PUBLIC
    absl::flat_hash_map
    absl::status
    absl::status_macros
    mld_common_gpu_model
    mld_common_memory_management
    mld_common_model
    mld_common_task_tensor_desc
    mld_common_types
    mld_gl_gl_buffer
    mld_gl_gl_spatial_tensor
)

# --- Flatbuffer generation for gl_inference_context.fbs ---
mld_generate_flatbuffers_cpp(mld_gl_gl_inference_context_cc_fbs
  "${CMAKE_CURRENT_SOURCE_DIR}/gl_inference_context.fbs"
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ml_drift/gl"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/gpu_model.fbs"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/task/serialization_base.fbs"
    flatc_host_tool
)
# --- End Flatbuffer generation ---

add_library(mld_gl_gl_inference_context gl_inference_context.cc gl_inference_context.h)
target_include_directories(mld_gl_gl_inference_context PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(mld_gl_gl_inference_context
  PUBLIC
    absl::flat_hash_map
    absl::span
    absl::status
    absl::status_macros
    absl::time
    flatbuffers
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_gpu_model_cc_fbs
    mld_common_gpu_model_util
    mld_common_ir_model
    mld_common_ir_model_util
    mld_common_model
    mld_common_shape
    mld_common_task_profiling_info
    mld_common_task_serialization_base
    mld_common_task_serialization_base_cc_fbs
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_common_types
    mld_gl_gl_inference_context_cc_fbs
    mld_gl_gl_operation
    mld_gl_gl_spatial_tensor
    mld_gl_memory_manager
    mld_gl_portable
    mld_gl_request_gpu_info
)
add_dependencies(mld_gl_gl_inference_context mld_gl_gl_inference_context_cc_fbs_gen)

add_executable(converter_test converter_test.cc)
target_include_directories(converter_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(converter_test
  PRIVATE
    absl::status
    absl::status_matchers
    gmock
    gtest_main
    mld_common_data_type
    mld_common_shape
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_gl_converter
    mld_gl_gl_buffer
    mld_gl_gl_spatial_tensor
    mld_gl_testing_gl_test
)

add_executable(gl_spatial_tensor_test gl_spatial_tensor_test.cc)
target_include_directories(gl_spatial_tensor_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(gl_spatial_tensor_test
  PRIVATE
    absl::check
    absl::log
    absl::status_matchers
    gmock
    gtest_main
    mld_common_data_type
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_gl_gl_spatial_tensor
    mld_gl_testing_gl_test
)

add_executable(request_gpu_info_test request_gpu_info_test.cc)
target_include_directories(request_gpu_info_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(request_gpu_info_test
  PRIVATE
    absl::status_matchers
    gmock
    gtest_main
    mld_common_gpu_info
    mld_gl_request_gpu_info
)

add_executable(mld_gl_converter_test ALIAS converter_test)
add_executable(mld_gl_gl_spatial_tensor_test ALIAS gl_spatial_tensor_test)
add_executable(mld_gl_request_gpu_info_test ALIAS request_gpu_info_test)

if(NOT CMAKE_CROSSCOMPILING)
  gtest_discover_tests(converter_test)
  gtest_discover_tests(gl_spatial_tensor_test)
  gtest_discover_tests(request_gpu_info_test)
endif()

add_subdirectory(testing)
