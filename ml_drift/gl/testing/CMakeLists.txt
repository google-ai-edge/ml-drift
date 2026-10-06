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

add_library(mld_gl_testing_perf_util
  perf_util.cc
  perf_util.h
)
target_include_directories(mld_gl_testing_perf_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_testing_perf_util
  PUBLIC
    absl::status
    absl::status_macros
    absl::time
    mld_common_data_type
    mld_common_flops_util
    mld_common_gpu_info
    mld_common_kernels_conv_constants
    mld_common_kernels_conv_generic
    mld_common_kernels_conv_wave_memory
    mld_common_kernels_depthwise_conv_tiled
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_gl_egl_environment
    mld_gl_gl_operation
    mld_gl_gl_spatial_tensor
    mld_gl_portable
    mld_gl_request_gpu_info
    XNNPACK
)

add_library(mld_gl_testing_gl_test
  gl_test.cc
  gl_test.h
)
target_include_directories(mld_gl_testing_gl_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_gl_testing_gl_test
  PUBLIC
    absl::status
    absl::status_macros
    absl::status_matchers
    gmock
    gtest
    mld_common_data_type
    mld_common_gpu_info
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_testing_util
    mld_gl_egl_environment
    mld_gl_gl_operation
    mld_gl_gl_spatial_tensor
    mld_gl_portable
    mld_gl_request_gpu_info
)
if(NOT ANDROID)
  target_link_libraries(mld_gl_testing_gl_test
    PUBLIC
      ${CMAKE_DL_LIBS}
  )
endif()

add_executable(operation_perf_test
  operation_perf_test.cc
)
target_include_directories(operation_perf_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(operation_perf_test
  PRIVATE
    absl::status_matchers
    gmock
    gtest_main
    mld_common_precision
    mld_common_shape
    mld_gl_testing_perf_util
)

add_executable(gpu_weights_conversion_test
  gpu_weights_conversion_test.cc
)
target_include_directories(gpu_weights_conversion_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(gpu_weights_conversion_test
  PRIVATE
    absl::check
    absl::flat_hash_map
    absl::status
    absl::status_macros
    absl::status_matchers
    absl::statusor
    absl::strings
    gmock
    gtest_main
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_gpu_model_util
    mld_common_gpu_weights_conversion_test_util
    mld_common_model
    mld_common_precision
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_task_testing_util
    mld_common_task_weights_conversion
    mld_common_task_weights_layout
    mld_common_tensor
    mld_common_util
    mld_gl_egl_environment
    mld_gl_gl_inference_context
    mld_gl_gl_spatial_tensor
    mld_gl_portable
    mld_gl_testing_gl_test
)

add_executable(mld_gl_operation_perf_test ALIAS operation_perf_test)
add_executable(mld_gl_gpu_weights_conversion_test ALIAS gpu_weights_conversion_test)

if(NOT CMAKE_CROSSCOMPILING)
  gtest_discover_tests(operation_perf_test)
  gtest_discover_tests(gpu_weights_conversion_test)
endif()
