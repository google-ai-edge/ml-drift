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

add_library(mld_cl_testing_perf_util
    perf_util.cc
    perf_util.h
)
target_include_directories(mld_cl_testing_perf_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_testing_perf_util
    PUBLIC
        absl::status
        absl::status_macros
        absl::statusor
        absl::time
        flatbuffers
        mld_cl_cl_operation
        mld_cl_environment
        mld_cl_tensor
        mld_common_data_type
        mld_common_flops_util
        mld_common_gpu_info
        mld_common_gpu_model_external_weights_test_util
        mld_common_kernels_conv_constants
        mld_common_kernels_conv_generic
        mld_common_kernels_conv_wave_matrix_mali
        mld_common_kernels_conv_wave_memory
        mld_common_kernels_depthwise_conv
        mld_common_kernels_depthwise_conv_3x3
        mld_common_kernels_depthwise_conv_tiled
        mld_common_kernels_depthwise_conv_wave_memory
        mld_common_kernels_fully_connected
        mld_common_kernels_quantize_and_dequantize
        mld_common_kernels_softmax
        mld_common_kernels_special_conv_softmax_conv
        mld_common_operations
        mld_common_precision
        mld_common_selectors_convolution_selector
        mld_common_shape
        mld_common_task_serialization_base_cc_fbs
        mld_common_task_weights_conversion
        mld_common_task_weights_layout
        mld_common_tensor
        mld_common_types
        mld_common_util
        mld_common_winograd_util
        XNNPACK
)

add_executable(operation_perf_test
    operation_perf_test.cc
)
target_include_directories(operation_perf_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(operation_perf_test PRIVATE
    gtest_main
    absl::status_matchers
    absl::time
    mld_cl_opencl_wrapper
    mld_cl_testing_perf_util
    mld_common_precision
    mld_common_shape
    mld_cl_default_util
)

add_library(mld_cl_testing_cl_test cl_test.cc cl_test.h)
target_include_directories(mld_cl_testing_cl_test PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_testing_cl_test PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  gmock
  gtest
  mld_cl_cl_arguments
  mld_cl_cl_operation
  mld_cl_default_util
  mld_cl_environment
  mld_cl_tensor
  mld_common_data_type
  mld_common_gpu_info
  mld_common_shape
  mld_common_task_testing_util
  mld_common_tensor
  mld_common_types
)

add_executable(gpu_model_external_weights_test gpu_model_external_weights_test.cc)
target_link_libraries(gpu_model_external_weights_test PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    gtest_main
    mld_cl_default_util
    mld_cl_testing_cl_test
    mld_common_gpu_model_external_weights_test_util
)
