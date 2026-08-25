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

add_library(mld_common_selectors_default_selector default/default_selector.cc default_selector.h)
target_include_directories(mld_common_selectors_default_selector PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_selectors_default_selector PUBLIC
    mld_common_gpu_info
    mld_common_model
    mld_common_model_hints
    mld_common_status
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
)

add_library(mld_common_selectors_operation_selector operation_selector.cc operation_selector.h)
target_include_directories(mld_common_selectors_operation_selector PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_selectors_operation_selector PUBLIC
    mld_common_selectors_default_selector
    mld_common_selectors_simple_selectors
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_model_hints
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_kernels_elementwise
    mld_common_kernels_fully_connected
    mld_common_kernels_mean_stddev_normalization
    mld_common_kernels_reduce
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_weights_layout
)

add_library(mld_common_selectors_special_selector special_selector.cc special_selector.h)
target_include_directories(mld_common_selectors_special_selector PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_selectors_special_selector PUBLIC
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_model_hints
    mld_common_status
    mld_common_kernels_mean_stddev_normalization_parser
    mld_common_kernels_mish
    mld_common_kernels_reduce_parser
    mld_common_kernels_special_concat_conv
    mld_common_kernels_special_conv_pointwise
    mld_common_kernels_special_dw7x7_conv2to6_concat_conv8to8
    mld_common_kernels_special_thin_local_memory_fuser
    mld_common_kernels_special_thin_pointwise_fuser
)

add_library(mld_common_selectors_convolution_selector convolution_selector.cc convolution_selector.h)
target_include_directories(mld_common_selectors_convolution_selector PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_selectors_convolution_selector PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_model_hints
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_tensor
    mld_common_kernels_conv_apple_mpp
    mld_common_kernels_conv_constants
    mld_common_kernels_conv_generic
    mld_common_kernels_conv_wave_matrix
    mld_common_kernels_conv_wave_matrix_mali
    mld_common_kernels_conv_wave_memory
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_weights_layout
)

add_library(mld_common_selectors_convolution_transposed_selector convolution_transposed_selector.cc convolution_transposed_selector.h)
target_include_directories(mld_common_selectors_convolution_transposed_selector PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_selectors_convolution_transposed_selector PUBLIC
    mld_common_gpu_info
    mld_common_operations
    mld_common_precision
    mld_common_kernels_convolution_transposed
    mld_common_kernels_convolution_transposed_2x2
    mld_common_kernels_convolution_transposed_3x3
    mld_common_kernels_convolution_transposed_3x3_thin
    mld_common_kernels_convolution_transposed_4x4
    mld_common_kernels_convolution_transposed_thin
    mld_common_task_gpu_operation
    mld_common_task_weights_layout
)

add_library(mld_common_selectors_dw_convolution_selector dw_convolution_selector.cc dw_convolution_selector.h)
target_include_directories(mld_common_selectors_dw_convolution_selector PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_selectors_dw_convolution_selector PUBLIC
    mld_common_gpu_info
    mld_common_operations
    mld_common_precision
    mld_common_kernels_depthwise_conv
    mld_common_kernels_depthwise_conv_3x3
    mld_common_kernels_depthwise_conv_tiled
    mld_common_kernels_depthwise_conv_wave_memory
    mld_common_task_gpu_operation
)

add_library(mld_common_selectors_simple_selectors simple_selectors.cc simple_selectors.h)
target_include_directories(mld_common_selectors_simple_selectors PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_selectors_simple_selectors PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_model_hints
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_util
    mld_common_kernels_add
    mld_common_kernels_concat_xy
    mld_common_kernels_concat_z
    mld_common_kernels_conv_weights_converter
    mld_common_kernels_cumsum
    mld_common_kernels_dynamic_update_slice
    mld_common_kernels_embedding_lookup
    mld_common_kernels_gather
    mld_common_kernels_lstm
    mld_common_kernels_max_unpooling
    mld_common_kernels_one_hot
    mld_common_kernels_padding
    mld_common_kernels_pooling
    mld_common_kernels_prelu
    mld_common_kernels_quantize_and_dequantize
    mld_common_kernels_reduce
    mld_common_kernels_relu
    mld_common_kernels_resampler
    mld_common_kernels_reshape
    mld_common_kernels_reshapex4
    mld_common_kernels_resize
    mld_common_kernels_reverse
    mld_common_kernels_scatter_nd
    mld_common_kernels_select_v2
    mld_common_kernels_softmax
    mld_common_kernels_softmax1x1
    mld_common_kernels_space_to_depth
    mld_common_kernels_split
    mld_common_kernels_tile
    mld_common_kernels_winograd
    mld_common_task_gpu_operation
    mld_common_task_weights_layout
)
