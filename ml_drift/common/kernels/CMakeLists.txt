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

add_library(mld_common_kernels_add add.cc add.h)
target_include_directories(mld_common_kernels_add PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_add
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    flatbuffers
    mld_common_data_type
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_object_desc
    mld_common_tensor
    mld_common_util
)

add_library(mld_common_kernels_elementwise elementwise.cc)
target_include_directories(mld_common_kernels_elementwise PUBLIC
    $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_common_kernels_elementwise
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_accumulate_input_channels accumulate_input_channels.cc accumulate_input_channels.h)
target_include_directories(mld_common_kernels_accumulate_input_channels PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_accumulate_input_channels
  PUBLIC
    absl::log
    absl::strings
    mld_common_data_type
    mld_common_shape
    mld_common_task_buffer_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
)

add_library(mld_common_kernels_bitcast bitcast.cc bitcast.h)
target_include_directories(mld_common_kernels_bitcast PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_bitcast
  PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_cast cast.cc cast.h)
target_include_directories(mld_common_kernels_cast PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_cast
  PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_concat_xy concat_xy.cc concat_xy.h)
target_include_directories(mld_common_kernels_concat_xy PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_concat_xy
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_concat_z concat_z.cc concat_z.h)
target_include_directories(mld_common_kernels_concat_z PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_concat_z
  PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_util
)

add_library(mld_common_kernels_conv_weights_converter conv_weights_converter.cc conv_weights_converter.h)
target_include_directories(mld_common_kernels_conv_weights_converter PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conv_weights_converter
  PUBLIC
    absl::flat_hash_map
    absl::log
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_shape
    mld_common_task_arguments
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_weights_conversion
    mld_common_task_weights_layout
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_conv_constants conv_constants.cc conv_constants.h)
target_include_directories(mld_common_kernels_conv_constants PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conv_constants
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    flatbuffers
    mld_common_data_type
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_object_desc
    mld_common_tensor
    mld_common_util
)

add_library(mld_common_kernels_conv_generic conv_generic.cc conv_generic.h)
target_include_directories(mld_common_kernels_conv_generic PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conv_generic
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    flatbuffers
    mld_common_data_type
    mld_common_kernels_fully_connected_util
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_object_desc
    mld_common_tensor
    mld_common_util
)


add_library(mld_common_kernels_cumsum cumsum.cc cumsum.h)
target_include_directories(mld_common_kernels_cumsum PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_cumsum
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_types
)

add_library(mld_common_kernels_embedding_lookup embedding_lookup.cc embedding_lookup.h)
target_include_directories(mld_common_kernels_embedding_lookup PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_embedding_lookup
  PUBLIC
    absl::span
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernels_fully_connected
    mld_common_operations
    mld_common_shape
    mld_common_task_buffer_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_weights_conversion
    mld_common_task_weights_layout
    mld_common_tensor
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_fully_connected fully_connected.cc fully_connected.h)
target_include_directories(mld_common_kernels_fully_connected PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_fully_connected
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_kernels_fully_connected_util
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_task_arguments
    mld_common_task_util
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_weights_conversion
    mld_common_task_weights_layout
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_fully_connected_util fully_connected_util.cc fully_connected_util.h)
target_include_directories(mld_common_kernels_fully_connected_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_fully_connected_util
  PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_precision
    mld_common_shape
    mld_common_task_buffer_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
)

add_library(mld_common_kernels_mean_stddev_normalization mean_stddev_normalization.cc mean_stddev_normalization.h)
target_include_directories(mld_common_kernels_mean_stddev_normalization PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_mean_stddev_normalization
  PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_mean_stddev_normalization_parser mean_stddev_normalization_parser.cc mean_stddev_normalization_parser.h)
target_include_directories(mld_common_kernels_mean_stddev_normalization_parser PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_mean_stddev_normalization_parser
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_kernels_mean_stddev_normalization
    mld_common_model
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_tensor
)

add_library(mld_common_kernels_mish mish.cc mish.h)
target_include_directories(mld_common_kernels_mish PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_mish
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_kernels_elementwise
    mld_common_model
    mld_common_operations
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_positional_embedding positional_embedding.cc positional_embedding.h)
target_include_directories(mld_common_kernels_positional_embedding PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_positional_embedding
  PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_quantize_and_dequantize quantize_and_dequantize.cc quantize_and_dequantize.h)
target_include_directories(mld_common_kernels_quantize_and_dequantize PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_quantize_and_dequantize
  PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_reduce reduce.cc reduce.h)
target_include_directories(mld_common_kernels_reduce PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_reduce
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_operations
    mld_common_shape
    mld_common_task_arguments
    mld_common_task_gpu_operation
    mld_common_task_util
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_conversion conversion.cc conversion.h)
target_include_directories(mld_common_kernels_conversion PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conversion
  PUBLIC
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_shape
    mld_common_task_buffer_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_tuning_type
    mld_common_task_work_group_picking
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_reduce_parser reduce_parser.cc reduce_parser.h)
target_include_directories(mld_common_kernels_reduce_parser PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_reduce_parser
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_gpu_info
    mld_common_gpu_model_builder
    mld_common_kernels_reduce
    mld_common_model
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_reverse reverse.cc reverse.h)
target_include_directories(mld_common_kernels_reverse PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_reverse
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_rope rope.cc rope.h)
target_include_directories(mld_common_kernels_rope PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_rope
  PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_select_v2 select_v2.cc select_v2.h)
target_include_directories(mld_common_kernels_select_v2 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_select_v2
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_softmax softmax.cc softmax.h)
target_include_directories(mld_common_kernels_softmax PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_softmax
  PUBLIC
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_shape
    mld_common_task_arguments
    mld_common_task_buffer_desc
    mld_common_task_gpu_operation
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_space_to_depth space_to_depth.cc space_to_depth.h)
target_include_directories(mld_common_kernels_space_to_depth PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_space_to_depth
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_strided_slice strided_slice.cc strided_slice.h)
target_include_directories(mld_common_kernels_strided_slice PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_strided_slice
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_operations
    mld_common_shape
    mld_common_task_arguments
    mld_common_task_gpu_operation
    mld_common_types
)

add_library(mld_common_kernels_top_k top_k.cc top_k.h)
target_include_directories(mld_common_kernels_top_k PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_top_k
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_task_arguments
    mld_common_task_gpu_operation
    mld_common_types
    mld_common_util
)

add_library(mld_common_kernels_transpose transpose.cc transpose.h)
target_include_directories(mld_common_kernels_transpose PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_transpose
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_max_unpooling max_unpooling.cc max_unpooling.h)
target_include_directories(mld_common_kernels_max_unpooling PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_max_unpooling
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)


add_library(mld_common_kernels_winograd winograd.cc winograd.h)
target_include_directories(mld_common_kernels_winograd PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_winograd
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_kernel_info
    mld_common_operations
    mld_common_shape
    mld_common_task_arguments
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_weights_conversion
    mld_common_task_weights_layout
    mld_common_tensor
    mld_common_types
    mld_common_util
    mld_common_winograd_util
)

add_library(mld_common_kernels_convolution_transposed_thin convolution_transposed_thin.cc convolution_transposed_thin.h)
target_include_directories(mld_common_kernels_convolution_transposed_thin PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_convolution_transposed_thin
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_convolution_transposed_3x3_thin convolution_transposed_3x3_thin.cc convolution_transposed_3x3_thin.h)
target_include_directories(mld_common_kernels_convolution_transposed_3x3_thin PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_convolution_transposed_3x3_thin
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_conv_wave_memory conv_wave_memory.cc conv_wave_memory.h)
target_include_directories(mld_common_kernels_conv_wave_memory PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conv_wave_memory
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_conv_wave_matrix conv_wave_matrix.cc conv_wave_matrix.h)
target_include_directories(mld_common_kernels_conv_wave_matrix PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conv_wave_matrix
  PUBLIC
    mld_common_flops_util
    mld_common_gpu_info
    mld_common_kernels_fully_connected_util
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_conv_wave_matrix_mali conv_wave_matrix_mali.cc conv_wave_matrix_mali.h)
target_include_directories(mld_common_kernels_conv_wave_matrix_mali PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conv_wave_matrix_mali
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_gather gather.cc gather.h)
target_include_directories(mld_common_kernels_gather PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_gather
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_lstm lstm.cc lstm.h)
target_include_directories(mld_common_kernels_lstm PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_lstm
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_dynamic_update_slice dynamic_update_slice.cc dynamic_update_slice.h)
target_include_directories(mld_common_kernels_dynamic_update_slice PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_dynamic_update_slice
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_depthwise_conv depthwise_conv.cc depthwise_conv.h)
target_include_directories(mld_common_kernels_depthwise_conv PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_depthwise_conv
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_depthwise_conv_3x3 depthwise_conv_3x3.cc depthwise_conv_3x3.h)
target_include_directories(mld_common_kernels_depthwise_conv_3x3 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_depthwise_conv_3x3
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_depthwise_conv_tiled depthwise_conv_tiled.cc depthwise_conv_tiled.h)
target_include_directories(mld_common_kernels_depthwise_conv_tiled PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_depthwise_conv_tiled
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_convolution_transposed convolution_transposed.cc convolution_transposed.h)
target_include_directories(mld_common_kernels_convolution_transposed PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_convolution_transposed
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_weights_conversion
)

add_library(mld_common_kernels_convolution_transposed_util convolution_transposed_util.cc convolution_transposed_util.h)
target_include_directories(mld_common_kernels_convolution_transposed_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_convolution_transposed_util
  PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_task_buffer_desc
    mld_common_task_gpu_object_desc
    mld_common_task_wave_memory_util
)

add_library(mld_common_kernels_convolution_transposed_2x2 convolution_transposed_2x2.cc convolution_transposed_2x2.h)
target_include_directories(mld_common_kernels_convolution_transposed_2x2 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_convolution_transposed_2x2
  PUBLIC
    mld_common_kernels_convolution_transposed_util
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_convolution_transposed_3x3 convolution_transposed_3x3.cc convolution_transposed_3x3.h)
target_include_directories(mld_common_kernels_convolution_transposed_3x3 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_convolution_transposed_3x3
  PUBLIC
    mld_common_kernels_convolution_transposed_util
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_convolution_transposed_4x4 convolution_transposed_4x4.cc convolution_transposed_4x4.h)
target_include_directories(mld_common_kernels_convolution_transposed_4x4 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_convolution_transposed_4x4
  PUBLIC
    mld_common_kernels_convolution_transposed_util
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_depthwise_conv_wave_memory depthwise_conv_wave_memory.cc depthwise_conv_wave_memory.h)
target_include_directories(mld_common_kernels_depthwise_conv_wave_memory PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_depthwise_conv_wave_memory
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_dot_general dot_general.cc dot_general.h)
target_include_directories(mld_common_kernels_dot_general PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_dot_general
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_one_hot one_hot.cc one_hot.h)
target_include_directories(mld_common_kernels_one_hot PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_one_hot
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_padding padding.cc padding.h)
target_include_directories(mld_common_kernels_padding PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_padding
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_prelu prelu.cc prelu.h)
target_include_directories(mld_common_kernels_prelu PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_prelu
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_random_philox random_philox.cc random_philox.h)
target_include_directories(mld_common_kernels_random_philox PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_random_philox
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_pooling pooling.cc pooling.h)
target_include_directories(mld_common_kernels_pooling PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_pooling
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_reshape reshape.cc reshape.h)
target_include_directories(mld_common_kernels_reshape PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_reshape
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_reshapex4 reshapex4.cc reshapex4.h)
target_include_directories(mld_common_kernels_reshapex4 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_reshapex4
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_resize resize.cc resize.h)
target_include_directories(mld_common_kernels_resize PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_resize
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_relu relu.cc relu.h)
target_include_directories(mld_common_kernels_relu PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_relu
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_softmax1x1 softmax1x1.cc softmax1x1.h)
target_include_directories(mld_common_kernels_softmax1x1 PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_softmax1x1
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_split split.cc split.h)
target_include_directories(mld_common_kernels_split PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_split
  PUBLIC
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_resampler resampler.cc resampler.h)
target_include_directories(mld_common_kernels_resampler PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_resampler
  PUBLIC
    mld_common_gpu_info
    mld_common_shape
    mld_common_task_gpu_operation
)

add_library(mld_common_kernels_tile tile.cc tile.h)
target_include_directories(mld_common_kernels_tile PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_tile
  PUBLIC
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
)

add_library(mld_common_kernels_conv_apple_mpp conv_apple_mpp.cc conv_apple_mpp.h)
target_include_directories(mld_common_kernels_conv_apple_mpp PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_kernels_conv_apple_mpp
  PUBLIC
    mld_common_kernels_fully_connected_util
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
)

# Every test executable under tests/ is built on opencl_kernel_test.cc and links
# the mld_cl_* targets, which only exist when the OpenCL backend is enabled.
# Skip the directory entirely for other backends (e.g. wgpu).
if(ML_DRIFT_API STREQUAL "cl")
  add_subdirectory(tests)
endif()
add_subdirectory(special)
