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

add_library(mld_webgpu_webgpu_headers INTERFACE webgpu_headers.h)
target_include_directories(mld_webgpu_webgpu_headers INTERFACE
    $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_webgpu_webgpu_headers
  INTERFACE
    dawn
)

add_library(mld_webgpu_gpu_object INTERFACE gpu_object.h)
target_include_directories(mld_webgpu_gpu_object INTERFACE
    $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_webgpu_gpu_object
  INTERFACE
    absl::status
    mld_common_task_gpu_object_desc
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_instance instance.cc instance.h)
target_include_directories(mld_webgpu_instance PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_instance
  PUBLIC
    absl::core_headers
    absl::flat_hash_map
    absl::leak_check
    absl::status
    absl::strings
    absl::synchronization
    absl::time
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_preprocessor preprocessor.cc preprocessor.h)
target_include_directories(mld_webgpu_preprocessor PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_preprocessor
  PUBLIC
    absl::status
    absl::status_macros
    absl::strings
    mld_common_gpu_info
)

add_library(mld_webgpu_webgpu_api_util webgpu_api_util.cc webgpu_api_util.h)
target_include_directories(mld_webgpu_webgpu_api_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_webgpu_api_util
  PUBLIC
    absl::leak_check
    absl::log
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    absl::synchronization
    absl::time
    mld_common_data_type
    mld_common_executor
    mld_common_util
    mld_webgpu_instance
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_buffer buffer.cc buffer.h)
target_include_directories(mld_webgpu_buffer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_buffer
  PUBLIC
    absl::span
    absl::status
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_util
    mld_webgpu_gpu_object
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_compute_pipeline_cache INTERFACE compute_pipeline_cache.h)
target_include_directories(mld_webgpu_compute_pipeline_cache INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_compute_pipeline_cache
  INTERFACE
    absl::flat_hash_map
    absl::status
    absl::status_macros
    absl::statusor
    farmhash
    mld_common_executor
    mld_common_gpu_info
    mld_webgpu_preprocessor
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_environment environment.cc environment.h)
target_include_directories(mld_webgpu_environment PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
if(NOT EMSCRIPTEN)
  target_compile_definitions(mld_webgpu_environment PUBLIC
    DAWN_ENABLE_BACKEND_VULKAN
    ML_DRIFT_DAWN_NATIVE
  )
endif()
target_link_libraries(mld_webgpu_environment
  PUBLIC
    absl::core_headers
    absl::log
    absl::memory
    absl::no_destructor
    absl::status
    absl::status_macros
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_task_tensor_desc
    mld_webgpu_buffer
    mld_webgpu_compute_pipeline_cache
    mld_webgpu_instance
    mld_webgpu_webgpu_headers
    XNNPACK
)

add_library(mld_webgpu_execution_environment INTERFACE execution_environment.h)
target_include_directories(mld_webgpu_execution_environment INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
if(NOT EMSCRIPTEN)
  target_compile_definitions(mld_webgpu_execution_environment INTERFACE
    DAWN_ENABLE_BACKEND_VULKAN
    ML_DRIFT_DAWN_NATIVE
  )
endif()
target_link_libraries(mld_webgpu_execution_environment
  INTERFACE
    mld_webgpu_environment
)

add_library(mld_webgpu_spatial_tensor spatial_tensor.cc spatial_tensor.h)
target_include_directories(mld_webgpu_spatial_tensor PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_spatial_tensor
  PUBLIC
    absl::log
    absl::span
    absl::status
    absl::status_macros
    absl::strings
    absl::time
    mld_common_data_type
    mld_common_gpu_info
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_tensor
    mld_common_task_tensor_desc
    mld_common_util
    mld_webgpu_environment
    mld_webgpu_gpu_object
    mld_webgpu_instance
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_arguments arguments.cc arguments.h)
target_include_directories(mld_webgpu_arguments PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_arguments
  PUBLIC
    absl::span
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
    mld_webgpu_buffer
    mld_webgpu_gpu_object
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_compute_task compute_task.cc compute_task.h)
target_include_directories(mld_webgpu_compute_task PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_compute_task
  PUBLIC
    absl::int128
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
    mld_webgpu_arguments
    mld_webgpu_buffer
    mld_webgpu_compute_pipeline_cache
    mld_webgpu_environment
    mld_webgpu_preprocessor
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_converter converter.cc converter.h)
target_include_directories(mld_webgpu_converter PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_converter
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
    mld_webgpu_buffer
    mld_webgpu_compute_task
    mld_webgpu_environment
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_metrics_collector
  metrics_collector.cc
  metrics_collector.h
  metrics_collector_internal.h
)
target_include_directories(mld_webgpu_metrics_collector PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_metrics_collector
  PUBLIC
    absl::status
    absl::status_macros
    absl::time
    mld_webgpu_buffer
    mld_webgpu_environment
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_memory_manager memory_manager.cc memory_manager.h)
target_include_directories(mld_webgpu_memory_manager PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_memory_manager
  PUBLIC
    absl::algorithm_container
    absl::flat_hash_map
    absl::hash
    absl::node_hash_map
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_gpu_model
    mld_common_memory_management
    mld_common_model
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_types
    mld_webgpu_buffer
    mld_webgpu_environment
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_headers
)

# --- Flatbuffer generation for serialization.fbs ---
mld_generate_flatbuffers_cpp(mld_webgpu_serialization_cc_fbs
  "${CMAKE_CURRENT_SOURCE_DIR}/serialization.fbs"
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ml_drift/webgpu"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/gpu_model.fbs"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/task/serialization_base.fbs"
    flatc_host_tool
)
# --- End Flatbuffer generation ---

add_library(mld_webgpu_inference_context inference_context.cc inference_context.h)
target_include_directories(mld_webgpu_inference_context PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(mld_webgpu_inference_context
  PUBLIC
    absl::algorithm_container
    absl::flat_hash_map
    absl::flat_hash_set
    absl::span
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    absl::time
    flatbuffers
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_cc_fbs
    mld_common_ir_model
    mld_common_ir_model_util
    mld_common_model
    mld_common_model_hints
    mld_common_shape
    mld_common_task_profiling_info
    mld_common_task_serialization_base
    mld_common_task_serialization_base_cc_fbs
    mld_common_task_tensor_desc
    mld_common_task_tuning_type
    mld_common_tensor
    mld_common_types
    mld_webgpu_compute_task
    mld_webgpu_environment
    mld_webgpu_memory_manager
    mld_webgpu_metrics_collector
    mld_webgpu_serialization_cc_fbs
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)
add_dependencies(mld_webgpu_inference_context mld_webgpu_serialization_cc_fbs_gen)

add_library(mld_webgpu_webgpu_weights_manager webgpu_weights_manager.cc webgpu_weights_manager.h)
target_include_directories(mld_webgpu_webgpu_weights_manager PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_webgpu_weights_manager
  PUBLIC
    absl::flat_hash_map
    absl::status
    absl::status_macros
    absl::statusor
    absl::time
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_task_gpu_operation
    mld_common_task_gpu_tensor
    mld_common_task_tuning_type
    mld_webgpu_compute_task
    mld_webgpu_environment
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_library(mld_webgpu_registry INTERFACE registry.h)
target_include_directories(mld_webgpu_registry INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_webgpu_registry
  INTERFACE
    absl::status
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_gpu_model_util
    mld_common_kernels_reduce_parser
    mld_common_model
    mld_common_selectors_operation_selector
    mld_common_task_gpu_operation
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/api.cc")
  add_library(mld_webgpu_api api.cc api.h)
  target_include_directories(mld_webgpu_api PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
  if(NOT EMSCRIPTEN)
    target_compile_definitions(mld_webgpu_api PUBLIC
      DAWN_ENABLE_BACKEND_VULKAN
    )
  endif()
  target_link_libraries(mld_webgpu_api
    PUBLIC
      absl::status
      absl::status_macros
      absl::statusor
      absl::strings
      mld_common_access_type
      mld_common_api_common
      mld_common_data_type
      mld_common_gpu_model
      mld_common_gpu_model_util
      mld_common_model
      mld_common_operations
      mld_common_precision
      mld_common_shape
      mld_common_task_tensor_desc
      mld_common_tensor
      mld_webgpu_environment
      mld_webgpu_inference_context
      mld_webgpu_metrics_collector
      mld_webgpu_registry
      mld_webgpu_spatial_tensor
      mld_webgpu_webgpu_headers
  )
endif()

add_executable(webgpu_weights_manager_test webgpu_weights_manager_test.cc)
target_include_directories(webgpu_weights_manager_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(webgpu_weights_manager_test
  PRIVATE
    absl::flat_hash_map
    absl::status_matchers
    absl::time
    gmock
    gtest_main
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_precision
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_task_weights_layout
    mld_common_tensor
    mld_webgpu_execution_environment
    mld_webgpu_inference_context
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
    mld_webgpu_webgpu_weights_manager
)

add_executable(converter_test converter_test.cc)
target_include_directories(converter_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(converter_test
  PRIVATE
    gmock
    gtest_main
    mld_common_data_type
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_common_types
    mld_common_util
    mld_webgpu_buffer
    mld_webgpu_converter
    mld_webgpu_spatial_tensor
    mld_webgpu_testing_webgpu_test
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)

add_executable(inference_context_test inference_context_test.cc)
target_include_directories(inference_context_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(inference_context_test
  PRIVATE
    absl::check
    absl::status_macros
    absl::statusor
    gmock
    gtest_main
    mld_common_data_type
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_gpu_model_util
    mld_common_model
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_webgpu_environment
    mld_webgpu_inference_context
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_headers
    XNNPACK
)

add_executable(preprocessor_test preprocessor_test.cc)
target_include_directories(preprocessor_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(preprocessor_test
  PRIVATE
    gmock
    gtest_main
    mld_common_gpu_info
    mld_webgpu_preprocessor
)

add_executable(spatial_tensor_test spatial_tensor_test.cc)
target_include_directories(spatial_tensor_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(spatial_tensor_test
  PRIVATE
    absl::check
    absl::log
    gmock
    gtest_main
    mld_common_data_type
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_webgpu_environment
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_headers
)

add_executable(mld_webgpu_webgpu_weights_manager_test ALIAS webgpu_weights_manager_test)
add_executable(mld_webgpu_converter_test ALIAS converter_test)
add_executable(mld_webgpu_inference_context_test ALIAS inference_context_test)
add_executable(mld_webgpu_preprocessor_test ALIAS preprocessor_test)
add_executable(mld_webgpu_spatial_tensor_test ALIAS spatial_tensor_test)

if(NOT CMAKE_CROSSCOMPILING)
  gtest_discover_tests(webgpu_weights_manager_test)
  gtest_discover_tests(converter_test)
  gtest_discover_tests(inference_context_test)
  gtest_discover_tests(preprocessor_test)
  gtest_discover_tests(spatial_tensor_test)
endif()

add_subdirectory(testing)
