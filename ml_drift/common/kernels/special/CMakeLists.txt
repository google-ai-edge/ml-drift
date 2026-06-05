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

add_library(mld_common_kernels_special_concat_conv concat_conv.cc concat_conv.h)
target_include_directories(mld_common_kernels_special_concat_conv PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_special_concat_conv PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_flops_util
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_kernel_info
    mld_common_model
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_types
    mld_common_util
    mld_common_task_arguments
    mld_common_task_buffer_desc
    mld_common_task_compiler_options
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_tuning_type
    mld_common_task_work_group_picking
)

add_library(mld_common_kernels_special_conv_pointwise conv_pointwise.cc conv_pointwise.h)
target_include_directories(mld_common_kernels_special_conv_pointwise PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_special_conv_pointwise PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_status
    mld_common_types
    mld_common_util
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
)

add_library(mld_common_kernels_special_conv_softmax_conv conv_softmax_conv.cc conv_softmax_conv.h)
target_include_directories(mld_common_kernels_special_conv_softmax_conv PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_special_conv_softmax_conv PUBLIC
    absl::strings
    absl::span
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_precision
    mld_common_shape
    mld_common_tensor
    mld_common_types
    mld_common_util
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_weights_conversion
    mld_common_task_weights_layout
)

add_library(mld_common_kernels_special_dw7x7_conv2to6_concat_conv8to8 dw7x7_conv2to6_concat_conv8to8.cc dw7x7_conv2to6_concat_conv8to8.h)
target_include_directories(mld_common_kernels_special_dw7x7_conv2to6_concat_conv8to8 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_special_dw7x7_conv2to6_concat_conv8to8 PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_flops_util
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_types
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_qcom_thin_filter_desc
    mld_common_task_tensor_desc
)

add_library(mld_common_kernels_special_thin_local_memory_fuser thin_local_memory_fuser.cc thin_local_memory_fuser.h)
target_include_directories(mld_common_kernels_special_thin_local_memory_fuser PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_special_thin_local_memory_fuser PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_flops_util
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_kernel_info
    mld_common_model
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_types
    mld_common_util
    mld_common_kernels_resize
    mld_common_task_arguments
    mld_common_task_buffer_desc
    mld_common_task_compiler_options
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_tuning_type
)

add_library(mld_common_kernels_special_thin_pointwise_fuser thin_pointwise_fuser.cc thin_pointwise_fuser.h)
target_include_directories(mld_common_kernels_special_thin_pointwise_fuser PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_special_thin_pointwise_fuser PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_flops_util
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_types
    mld_common_util
    mld_common_kernels_elementwise
    mld_common_kernels_prelu
    mld_common_kernels_relu
    mld_common_task_arguments
    mld_common_task_buffer_desc
    mld_common_task_compiler_options
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_util
)
