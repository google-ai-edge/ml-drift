# Copyright 2025 The ML Drift Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(mld_common_kernels_tests_add_test_util add_test_util.cc add_test_util.h)
target_include_directories(mld_common_kernels_tests_add_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_add_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_add
  mld_common_shape
  mld_common_status
  mld_common_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(add_test_opencl opencl_kernel_test.cc add_test.cc)
target_link_libraries(add_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_add_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_accumulate_input_channels_test_util accumulate_input_channels_test_util.cc accumulate_input_channels_test_util.h)
target_include_directories(mld_common_kernels_tests_accumulate_input_channels_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_accumulate_input_channels_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_accumulate_input_channels
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_library(mld_common_kernels_tests_bitcast_test_util bitcast_test_util.cc bitcast_test_util.h)
target_include_directories(mld_common_kernels_tests_bitcast_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_bitcast_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_bitcast
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(bitcast_test_opencl opencl_kernel_test.cc bitcast_test.cc)
target_link_libraries(bitcast_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_bitcast_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_buffer_desc_test_util buffer_desc_test_util.cc buffer_desc_test_util.h)
target_include_directories(mld_common_kernels_tests_buffer_desc_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_buffer_desc_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_task_buffer_desc
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(buffer_desc_test_opencl opencl_kernel_test.cc buffer_desc_test.cc)
target_link_libraries(buffer_desc_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_buffer_desc_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_cast_test_util cast_test_util.cc cast_test_util.h)
target_include_directories(mld_common_kernels_tests_cast_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_cast_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_cast
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(cast_test_opencl opencl_kernel_test.cc cast_test.cc)
target_link_libraries(cast_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_cast_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_concat_test_util concat_test_util.cc concat_test_util.h)
target_include_directories(mld_common_kernels_tests_concat_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_concat_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_concat_xy
  mld_common_kernels_concat_z
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(concat_test_opencl opencl_kernel_test.cc concat_test.cc)
target_link_libraries(concat_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_concat_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_conv_constants_test_util conv_constants_test_util.cc conv_constants_test_util.h)
target_include_directories(mld_common_kernels_tests_conv_constants_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_conv_constants_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  mld_common_data_type
  mld_common_kernels_conv_constants
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
)

add_executable(conv_constants_test_opencl opencl_kernel_test.cc conv_constants_test.cc)
target_link_libraries(conv_constants_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_conv_constants_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_conv_generic_test_util conv_generic_test_util.cc conv_generic_test_util.h)
target_include_directories(mld_common_kernels_tests_conv_generic_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_conv_generic_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  absl::log
  absl::span
  mld_common_data_type
  mld_common_kernels_conv_generic
  mld_common_kernels_conv_weights_converter
  mld_common_kernels_fully_connected
  mld_common_kernels_quantize_and_dequantize
  mld_common_kernels_winograd
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_winograd_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
  mld_common_task_weights_layout
)

add_executable(conv_generic_test_opencl opencl_kernel_test.cc conv_generic_test.cc)
target_link_libraries(conv_generic_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_conv_generic_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_conv_transpose_test_util conv_transpose_test_util.cc conv_transpose_test_util.h)
target_include_directories(mld_common_kernels_tests_conv_transpose_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_conv_transpose_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  mld_common_data_type
  mld_common_kernels_convolution_transposed
  mld_common_kernels_convolution_transposed_2x2
  mld_common_kernels_convolution_transposed_3x3
  mld_common_kernels_convolution_transposed_4x4
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
)

add_executable(conv_transpose_test_opencl opencl_kernel_test.cc conv_transpose_test.cc)
target_link_libraries(conv_transpose_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_conv_transpose_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_conv_transpose_thin_test_util conv_transpose_thin_test_util.cc conv_transpose_thin_test_util.h)
target_include_directories(mld_common_kernels_tests_conv_transpose_thin_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_conv_transpose_thin_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  mld_common_data_type
  mld_common_kernels_convolution_transposed_3x3_thin
  mld_common_kernels_convolution_transposed_thin
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
)

add_executable(conv_transpose_thin_test_opencl opencl_kernel_test.cc conv_transpose_thin_test.cc)
target_link_libraries(conv_transpose_thin_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_conv_transpose_thin_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_conv_wave_matrix_mali_test_util conv_wave_matrix_mali_test_util.cc conv_wave_matrix_mali_test_util.h)
target_include_directories(mld_common_kernels_tests_conv_wave_matrix_mali_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_conv_wave_matrix_mali_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  absl::status
  absl::span
  mld_common_data_type
  mld_common_kernels_conv_wave_matrix_mali
  mld_common_kernels_quantize_and_dequantize
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
  mld_common_task_weights_layout
)

add_executable(conv_wave_matrix_mali_test_opencl opencl_kernel_test.cc conv_wave_matrix_mali_test.cc)
target_link_libraries(conv_wave_matrix_mali_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_conv_wave_matrix_mali_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_conv_wave_matrix_test_util conv_wave_matrix_test_util.cc conv_wave_matrix_test_util.h)
target_include_directories(mld_common_kernels_tests_conv_wave_matrix_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_conv_wave_matrix_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  absl::span
  mld_common_data_type
  mld_common_kernels_conv_wave_matrix
  mld_common_kernels_fully_connected
  mld_common_kernels_quantize_and_dequantize
  mld_common_kernels_winograd
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_winograd_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
  mld_common_task_weights_layout
)

add_executable(conv_wave_matrix_test_opencl opencl_kernel_test.cc conv_wave_matrix_test.cc)
target_link_libraries(conv_wave_matrix_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_conv_wave_matrix_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_conv_wave_memory_test_util conv_wave_memory_test_util.cc conv_wave_memory_test_util.h)
target_include_directories(mld_common_kernels_tests_conv_wave_memory_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_conv_wave_memory_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  absl::log
  absl::span
  mld_common_data_type
  mld_common_kernels_conv_wave_memory
  mld_common_kernels_quantize_and_dequantize
  mld_common_kernels_winograd
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_winograd_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
  mld_common_task_weights_layout
)

add_executable(conv_wave_memory_test_opencl opencl_kernel_test.cc conv_wave_memory_test.cc)
target_link_libraries(conv_wave_memory_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_conv_wave_memory_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)



add_library(mld_common_kernels_tests_cumsum_test_util cumsum_test_util.cc cumsum_test_util.h)
target_include_directories(mld_common_kernels_tests_cumsum_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_cumsum_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_cumsum
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(cumsum_test_opencl opencl_kernel_test.cc cumsum_test.cc)
target_link_libraries(cumsum_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_cumsum_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_depthwise_conv_test_util depthwise_conv_test_util.cc depthwise_conv_test_util.h)
target_include_directories(mld_common_kernels_tests_depthwise_conv_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_depthwise_conv_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_depthwise_conv
  mld_common_kernels_depthwise_conv_3x3
  mld_common_kernels_depthwise_conv_tiled
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(depthwise_conv_test_opencl opencl_kernel_test.cc depthwise_conv_test.cc)
target_link_libraries(depthwise_conv_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_depthwise_conv_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_depthwise_conv_wave_memory_test_util depthwise_conv_wave_memory_test_util.cc depthwise_conv_wave_memory_test_util.h)
target_include_directories(mld_common_kernels_tests_depthwise_conv_wave_memory_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_depthwise_conv_wave_memory_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_depthwise_conv_wave_memory
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(depthwise_conv_wave_memory_test_opencl opencl_kernel_test.cc depthwise_conv_wave_memory_test.cc)
target_link_libraries(depthwise_conv_wave_memory_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_depthwise_conv_wave_memory_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_dot_general_test_util dot_general_test_util.cc dot_general_test_util.h)
target_include_directories(mld_common_kernels_tests_dot_general_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_dot_general_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_dot_general
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(dot_general_test_opencl opencl_kernel_test.cc dot_general_test.cc)
target_link_libraries(dot_general_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_dot_general_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_dynamic_update_slice_test_util dynamic_update_slice_test_util.cc dynamic_update_slice_test_util.h)
target_include_directories(mld_common_kernels_tests_dynamic_update_slice_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_dynamic_update_slice_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_dynamic_update_slice
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(dynamic_update_slice_test_opencl opencl_kernel_test.cc dynamic_update_slice_test.cc)
target_link_libraries(dynamic_update_slice_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_dynamic_update_slice_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_elementwise_test_util elementwise_test_util.cc elementwise_test_util.h)
target_include_directories(mld_common_kernels_tests_elementwise_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_elementwise_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_elementwise
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(elementwise_test_opencl opencl_kernel_test.cc elementwise_test.cc)
target_link_libraries(elementwise_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_elementwise_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_embedding_lookup_test_util embedding_lookup_test_util.cc embedding_lookup_test_util.h)
target_include_directories(mld_common_kernels_tests_embedding_lookup_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_embedding_lookup_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  absl::status
  mld_common_data_type
  mld_common_kernels_embedding_lookup
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(embedding_lookup_test_opencl opencl_kernel_test.cc embedding_lookup_test.cc)
target_link_libraries(embedding_lookup_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_embedding_lookup_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_fully_connected_test_util fully_connected_test_util.cc fully_connected_test_util.h)
target_include_directories(mld_common_kernels_tests_fully_connected_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_fully_connected_test_util PUBLIC
  gtest
  gmock
  XNNPACK
  mld_common_data_type
  mld_common_kernels_fully_connected
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(fully_connected_test_opencl opencl_kernel_test.cc fully_connected_test.cc)
target_link_libraries(fully_connected_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_fully_connected_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_gather_test_util gather_test_util.cc gather_test_util.h)
target_include_directories(mld_common_kernels_tests_gather_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_gather_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_gather
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(gather_test_opencl opencl_kernel_test.cc gather_test.cc)
target_link_libraries(gather_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_gather_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_executable(group_norm_test_opencl opencl_kernel_test.cc group_norm_test.cc)
target_link_libraries(group_norm_test_opencl PUBLIC
  mld_common_task_util
  absl::status
  absl::strings
  mld_common_data_type
  mld_common_gpu_model
  mld_common_gpu_model_util
  mld_common_model
  mld_common_model_hints
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_tensor
  mld_common_tensor_matcher
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_executable(layer_norm_test_opencl opencl_kernel_test.cc layer_norm_test.cc)
target_link_libraries(layer_norm_test_opencl PUBLIC
  mld_common_kernels_conv_apple_mpp
  mld_common_selectors_convolution_selector
  mld_common_kernels_mean_stddev_normalization
  mld_common_task_util
  absl::status
  absl::strings
  mld_common_data_type
  mld_common_gpu_model
  mld_common_gpu_model_util
  mld_common_model
  mld_common_model_hints
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_tensor
  mld_common_tensor_matcher
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_lstm_test_util lstm_test_util.cc lstm_test_util.h)
target_include_directories(mld_common_kernels_tests_lstm_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_lstm_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_lstm
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(lstm_test_opencl opencl_kernel_test.cc lstm_test.cc)
target_link_libraries(lstm_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_lstm_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_max_unpooling_test_util max_unpooling_test_util.cc max_unpooling_test_util.h)
target_include_directories(mld_common_kernels_tests_max_unpooling_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_max_unpooling_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_max_unpooling
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(max_unpooling_test_opencl opencl_kernel_test.cc max_unpooling_test.cc)
target_link_libraries(max_unpooling_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_max_unpooling_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_mean_stddev_normalization_test_util mean_stddev_normalization_test_util.cc mean_stddev_normalization_test_util.h)
target_include_directories(mld_common_kernels_tests_mean_stddev_normalization_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_mean_stddev_normalization_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_mean_stddev_normalization
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(mean_stddev_normalization_test_opencl opencl_kernel_test.cc mean_stddev_normalization_test.cc)
target_link_libraries(mean_stddev_normalization_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_mean_stddev_normalization_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_one_hot_test_util one_hot_test_util.cc one_hot_test_util.h)
target_include_directories(mld_common_kernels_tests_one_hot_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_one_hot_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_one_hot
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(one_hot_test_opencl opencl_kernel_test.cc one_hot_test.cc)
target_link_libraries(one_hot_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_one_hot_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_padding_test_util padding_test_util.cc padding_test_util.h)
target_include_directories(mld_common_kernels_tests_padding_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_padding_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_padding
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(padding_test_opencl opencl_kernel_test.cc padding_test.cc)
target_link_libraries(padding_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_padding_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_pooling_test_util pooling_test_util.cc pooling_test_util.h)
target_include_directories(mld_common_kernels_tests_pooling_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_pooling_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_pooling
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(pooling_test_opencl opencl_kernel_test.cc pooling_test.cc)
target_link_libraries(pooling_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_pooling_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_executable(positional_embedding_test_opencl opencl_kernel_test.cc positional_embedding_test.cc)
target_link_libraries(positional_embedding_test_opencl PUBLIC
  mld_common_task_util
  absl::status
  mld_common_data_type
  mld_common_kernels_positional_embedding
  mld_common_precision
  mld_common_shape
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_prelu_test_util prelu_test_util.cc prelu_test_util.h)
target_include_directories(mld_common_kernels_tests_prelu_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_prelu_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_prelu
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(prelu_test_opencl opencl_kernel_test.cc prelu_test.cc)
target_link_libraries(prelu_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_prelu_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)



add_library(mld_common_kernels_tests_random_philox_test_util random_philox_test_util.cc random_philox_test_util.h)
target_include_directories(mld_common_kernels_tests_random_philox_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_random_philox_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_random_philox
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(random_philox_test_opencl opencl_kernel_test.cc random_philox_test.cc)
target_link_libraries(random_philox_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_random_philox_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_reduce_test_util reduce_test_util.cc reduce_test_util.h)
target_include_directories(mld_common_kernels_tests_reduce_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_reduce_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_reduce
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(reduce_test_opencl opencl_kernel_test.cc reduce_test.cc)
target_link_libraries(reduce_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_reduce_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_relu_test_util relu_test_util.cc relu_test_util.h)
target_include_directories(mld_common_kernels_tests_relu_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_relu_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_relu
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(relu_test_opencl opencl_kernel_test.cc relu_test.cc)
target_link_libraries(relu_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_relu_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_resampler_test_util resampler_test_util.cc resampler_test_util.h)
target_include_directories(mld_common_kernels_tests_resampler_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_resampler_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_resampler
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(resampler_test_opencl opencl_kernel_test.cc resampler_test.cc)
target_link_libraries(resampler_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_resampler_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_reshape_test_util reshape_test_util.cc reshape_test_util.h)
target_include_directories(mld_common_kernels_tests_reshape_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_reshape_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_reshape
  mld_common_kernels_reshapex4
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(reshape_test_opencl opencl_kernel_test.cc reshape_test.cc)
target_link_libraries(reshape_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_reshape_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_resize_test_util resize_test_util.cc resize_test_util.h)
target_include_directories(mld_common_kernels_tests_resize_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_resize_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_resize
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(resize_test_opencl opencl_kernel_test.cc resize_test.cc)
target_link_libraries(resize_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_resize_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_rope_test_util rope_test_util.cc rope_test_util.h)
target_include_directories(mld_common_kernels_tests_rope_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_rope_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_rope
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(rope_test_opencl opencl_kernel_test.cc rope_test.cc)
target_link_libraries(rope_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_rope_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_executable(sdpa_test_opencl opencl_kernel_test.cc sdpa_test.cc)
target_link_libraries(sdpa_test_opencl PUBLIC
  mld_common_task_util
  absl::strings
  mld_common_data_type
  mld_common_gpu_model
  mld_common_gpu_model_util
  mld_common_model
  mld_common_model_hints
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_tensor_matcher
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_select_v2_test_util select_v2_test_util.cc select_v2_test_util.h)
target_include_directories(mld_common_kernels_tests_select_v2_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_select_v2_test_util PUBLIC
  gtest
  gmock
  absl::status
  mld_common_data_type
  mld_common_kernels_select_v2
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(select_v2_test_opencl opencl_kernel_test.cc select_v2_test.cc)
target_link_libraries(select_v2_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_select_v2_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_softmax_test_util softmax_test_util.cc softmax_test_util.h)
target_include_directories(mld_common_kernels_tests_softmax_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_softmax_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_gpu_info
  mld_common_kernels_softmax
  mld_common_kernels_softmax1x1
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(softmax_test_opencl opencl_kernel_test.cc softmax_test.cc)
target_link_libraries(softmax_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_softmax_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_space_to_depth_test_util space_to_depth_test_util.cc space_to_depth_test_util.h)
target_include_directories(mld_common_kernels_tests_space_to_depth_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_space_to_depth_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_space_to_depth
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(space_to_depth_test_opencl opencl_kernel_test.cc space_to_depth_test.cc)
target_link_libraries(space_to_depth_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_space_to_depth_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_split_test_util split_test_util.cc split_test_util.h)
target_include_directories(mld_common_kernels_tests_split_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_split_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_split
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(split_test_opencl opencl_kernel_test.cc split_test.cc)
target_link_libraries(split_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_split_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_strided_slice_test_util strided_slice_test_util.cc strided_slice_test_util.h)
target_include_directories(mld_common_kernels_tests_strided_slice_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_strided_slice_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_strided_slice
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(strided_slice_test_opencl opencl_kernel_test.cc strided_slice_test.cc)
target_link_libraries(strided_slice_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_strided_slice_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_tile_test_util tile_test_util.cc tile_test_util.h)
target_include_directories(mld_common_kernels_tests_tile_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_tile_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_tile
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(tile_test_opencl opencl_kernel_test.cc tile_test.cc)
target_link_libraries(tile_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_tile_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_top_k_test_util top_k_test_util.cc top_k_test_util.h)
target_include_directories(mld_common_kernels_tests_top_k_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_top_k_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_top_k
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_types
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_util
)

add_executable(top_k_test_opencl opencl_kernel_test.cc top_k_test.cc)
target_link_libraries(top_k_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_top_k_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_transpose_test_util transpose_test_util.cc transpose_test_util.h)
target_include_directories(mld_common_kernels_tests_transpose_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_transpose_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_transpose
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(transpose_test_opencl opencl_kernel_test.cc transpose_test.cc)
target_link_libraries(transpose_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_transpose_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

add_library(mld_common_kernels_tests_winograd_test_util winograd_test_util.cc winograd_test_util.h)
target_include_directories(mld_common_kernels_tests_winograd_test_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tests_winograd_test_util PUBLIC
  gtest
  gmock
  mld_common_data_type
  mld_common_kernels_winograd
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_util
  mld_common_winograd_util
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
  mld_common_task_weights_conversion
  mld_common_task_weights_layout
)

add_executable(winograd_test_opencl opencl_kernel_test.cc winograd_test.cc)
target_link_libraries(winograd_test_opencl PUBLIC
  mld_common_task_util
  mld_common_kernels_tests_winograd_test_util
  mld_common_data_type
  mld_common_status_matchers
  mld_common_task_tensor_desc
  mld_common_task_testing_util
  absl::strings
  gtest
  gmock
  benchmark
  opencl_headers
  mld_cl_environment
  mld_cl_opencl_wrapper
  mld_cl_cl_command_queue
  mld_cl_cl_context
  mld_cl_cl_program_cache
  mld_cl_util_types
  mld_common_operations
  mld_common_util
  mld_cl_cl_event
  mld_cl_cl_kernel
  mld_cl_cl_program
  mld_cl_default_util
  mld_cl_testing_cl_test
)

