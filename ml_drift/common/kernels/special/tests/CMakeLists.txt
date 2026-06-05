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

add_executable(concat_conv_test_opencl ../../tests/opencl_kernel_test.cc concat_conv_test.cc)
target_link_libraries(concat_conv_test_opencl PUBLIC
  gtest
  gmock
  absl::flat_hash_map
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_gpu_model
  mld_common_gpu_model_builder
  mld_common_model
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_concat_conv
  mld_common_kernels_special_thin_local_memory_fuser
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(conv2x2_max_pool2x2_test_opencl ../../tests/opencl_kernel_test.cc conv2x2_max_pool2x2_test.cc)
target_link_libraries(conv2x2_max_pool2x2_test_opencl PUBLIC
  gtest
  gmock
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_conv2x2_max_pool2x2
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(conv_softmax_conv_test_opencl ../../tests/opencl_kernel_test.cc conv_softmax_conv_test.cc)
target_link_libraries(conv_softmax_conv_test_opencl PUBLIC
  gtest
  gmock
  XNNPACK
  absl::strings
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_tensor
  mld_common_util
  mld_common_kernels_special_conv_softmax_conv
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(dw7x7_conv2to6_concat_conv8to8_test_opencl ../../tests/opencl_kernel_test.cc dw7x7_conv2to6_concat_conv8to8_test.cc)
target_link_libraries(dw7x7_conv2to6_concat_conv8to8_test_opencl PUBLIC
  gtest
  gmock
  absl::strings
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_dw7x7_conv2to6_concat_conv8to8
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(dw3x3_conv16to16_conv16to32_add_conv32to16_test_opencl ../../tests/opencl_kernel_test.cc dw3x3_conv16to16_conv16to32_add_conv32to16_test.cc)
target_link_libraries(dw3x3_conv16to16_conv16to32_add_conv32to16_test_opencl PUBLIC
  gtest
  gmock
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_dw3x3_conv16to16_conv16to32_add_conv32to16
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8_test_opencl ../../tests/opencl_kernel_test.cc dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8_test.cc)
target_link_libraries(dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8_test_opencl PUBLIC
  gtest
  gmock
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32_test_opencl ../../tests/opencl_kernel_test.cc dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32_test.cc)
target_link_libraries(dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32_test_opencl PUBLIC
  gtest
  gmock
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_operations
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(thin_local_memory_fuser_test_opencl ../../tests/opencl_kernel_test.cc thin_local_memory_fuser_test.cc)
target_link_libraries(thin_local_memory_fuser_test_opencl PUBLIC
  gtest
  gmock
  absl::flat_hash_map
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_gpu_model
  mld_common_gpu_model_builder
  mld_common_model
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_thin_local_memory_fuser
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)

add_executable(thin_pointwise_fuser_test_opencl ../../tests/opencl_kernel_test.cc thin_pointwise_fuser_test.cc)
target_link_libraries(thin_pointwise_fuser_test_opencl PUBLIC
  gtest
  gmock
  absl::flat_hash_map
  mld_cl_testing_cl_test
  mld_common_data_type
  mld_common_gpu_model
  mld_common_gpu_model_builder
  mld_common_model
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_status
  mld_common_tensor
  mld_common_kernels_special_thin_pointwise_fuser
  mld_common_task_gpu_operation
  mld_common_task_tensor_desc
  mld_common_task_testing_ref_ops
  mld_common_task_testing_util
)
