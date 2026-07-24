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

## --- Header-Only (INTERFACE) Libraries ---

add_library(mld_common_status INTERFACE
    status.h
)
target_include_directories(mld_common_status INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_status INTERFACE absl::status absl::statusor)

add_library(mld_common_access_type INTERFACE
    access_type.h
)
target_include_directories(mld_common_access_type INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)

add_library(mld_common_executor INTERFACE
    executor.h
)
target_include_directories(mld_common_executor INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)

add_library(mld_common_kernel_info INTERFACE
    kernel_info.h
)
target_include_directories(mld_common_kernel_info INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)

add_library(mld_common_model_hints INTERFACE
    model_hints.h
)
target_include_directories(mld_common_model_hints INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)

add_library(mld_common_types INTERFACE
    types.h
)
target_include_directories(mld_common_types INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_types INTERFACE fp16)


## --- Libraries with Source Files ---

add_library(mld_common_data_type data_type.cc data_type.h)
target_include_directories(mld_common_data_type PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_data_type PRIVATE absl::strings)

add_library(mld_common_flops_util flops_util.cc flops_util.h)
target_include_directories(mld_common_flops_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_flops_util PUBLIC mld_common_shape)

add_library(mld_common_shape shape.cc shape.h)
target_include_directories(mld_common_shape PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_shape PRIVATE absl::strings)

add_library(mld_common_precision precision.cc precision.h)
target_include_directories(mld_common_precision PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_precision PUBLIC mld_common_data_type)

add_library(mld_common_tensor tensor.h)
target_include_directories(mld_common_tensor PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_tensor PUBLIC
    mld_common_data_type
    mld_common_shape
    mld_common_types
    absl::span
)

add_library(mld_common_operations operations.cc operations.h)
target_include_directories(mld_common_operations PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_operations PUBLIC
    mld_common_data_type
    mld_common_shape
    mld_common_status
    mld_common_tensor
PRIVATE
    absl::flat_hash_map
    absl::log
    XNNPACK
)

add_library(mld_common_util util.cc util.h)
target_include_directories(mld_common_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_util PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_precision
    mld_common_shape
    mld_common_types
)

add_library(mld_common_gpu_info gpu_info.cc gpu_info.h)
target_include_directories(mld_common_gpu_info PUBLIC
    $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
    $<TARGET_PROPERTY:FP16,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(mld_common_gpu_info PUBLIC
    mld_common_data_type
    absl::flat_hash_set
    mld_common_types
PRIVATE
    absl::strings
    FP16
)

add_library(mld_common_winograd_util winograd_util.cc winograd_util.h)
target_include_directories(mld_common_winograd_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_winograd_util PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_operations
    mld_common_shape
    mld_common_tensor
PRIVATE
    mld_common_util
    XNNPACK
)

# --- Flatbuffer generation for gpu_model.fbs ---
mld_generate_flatbuffers_cpp(mld_common_gpu_model_cc_fbs
  "${CMAKE_CURRENT_SOURCE_DIR}/gpu_model.fbs"
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ml_drift/common"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/task/serialization_base.fbs"
    flatc_host_tool
)
# --- End Flatbuffer generation ---

add_library(mld_common_gpu_model gpu_model.cc gpu_model.h)
target_include_directories(mld_common_gpu_model PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${CMAKE_CURRENT_BINARY_DIR}"
  $<TARGET_PROPERTY:mld_common_task_serialization_base_cc_fbs,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(mld_common_gpu_model PUBLIC
    mld_common_gpu_model_cc_fbs
    mld_common_model
    mld_common_model_hints
    mld_common_precision
    mld_common_status
    mld_common_task_gpu_operation
    mld_common_task_gpu_tensor
    mld_common_task_tensor_desc
    absl::flat_hash_map
    absl::flat_hash_set
    flatbuffers
PRIVATE
    mld_common_operations
    mld_common_task_serialization_base
    mld_common_task_serialization_base_cc_fbs
)

add_library(mld_common_gpu_model_builder gpu_model_builder.cc gpu_model_builder.h)
target_include_directories(mld_common_gpu_model_builder PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_common_gpu_model_builder PUBLIC
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_model
    mld_common_model_hints
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_status
    mld_common_tensor
    absl::flat_hash_map
    mld_common_kernels_reduce
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_task_weights_layout
PRIVATE
    mld_common_flops_util
    mld_common_merge_nodes
    mld_common_util
    mld_common_winograd_util
    absl::log
    absl::strings
    mld_common_kernels_accumulate_input_channels
    mld_common_kernels_bitcast
    mld_common_kernels_cast
    mld_common_kernels_conv_weights_converter
    mld_common_kernels_cumsum
    mld_common_kernels_elementwise
    mld_common_kernels_embedding_lookup
    mld_common_kernels_fully_connected
    mld_common_kernels_mean_stddev_normalization
    mld_common_kernels_positional_embedding
    mld_common_kernels_quantize_and_dequantize
    mld_common_kernels_rope
    mld_common_kernels_select_v2
    mld_common_kernels_softmax
    mld_common_kernels_space_to_depth
    mld_common_kernels_strided_slice
    mld_common_kernels_top_k
    mld_common_kernels_transpose
    mld_common_kernels_winograd
    mld_common_kernels_special_conv_softmax_conv
    mld_common_selectors_convolution_selector
    mld_common_selectors_convolution_transposed_selector
    mld_common_selectors_dw_convolution_selector
    mld_common_selectors_simple_selectors
    mld_common_task_buffer_desc
    mld_common_task_weights_conversion
)

add_library(mld_common_gpu_model_util gpu_model_util.cc gpu_model_util.h)
target_include_directories(mld_common_gpu_model_util PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_common_gpu_model_util PUBLIC
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_status
    mld_common_task_gpu_operation
    absl::flat_hash_map
PRIVATE
    mld_common_data_type
    mld_common_model_hints
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_tensor
    absl::flat_hash_set
    absl::strings
    mld_common_selectors_operation_selector
    mld_common_selectors_special_selector
    mld_common_task_gpu_tensor
    mld_common_task_tensor_desc
)

add_library(mld_common_gpu_model_external_weights_test_util gpu_model_external_weights_test_util.cc gpu_model_external_weights_test_util.h)
target_include_directories(mld_common_gpu_model_external_weights_test_util PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_common_gpu_model_external_weights_test_util PUBLIC
    mld_common_data_type
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_gpu_model_util
    mld_common_model
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_status
    mld_common_tensor
    gtest
    XNNPACK
    absl::span
    mld_common_task_tensor_desc
    mld_common_task_testing_ref_ops
    mld_common_task_testing_util
    mld_common_task_weights_conversion
    mld_common_task_weights_layout
)

add_library(mld_common_model model.cc model.h)
target_include_directories(mld_common_model PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_common_model PUBLIC
    mld_common_status
    mld_common_tensor
PRIVATE
    absl::log
)

add_library(mld_common_merge_nodes merge_nodes.cc merge_nodes.h)
target_include_directories(mld_common_merge_nodes PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_common_merge_nodes PUBLIC
    mld_common_gpu_model
)

add_library(mld_common_tensor_matcher INTERFACE)
target_include_directories(mld_common_tensor_matcher INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_tensor_matcher INTERFACE gtest mld_common_tensor)


add_library(mld_common_memory_management memory_management.cc memory_management.h)
target_include_directories(mld_common_memory_management PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
)
target_link_libraries(mld_common_memory_management PUBLIC
    mld_common_status
    mld_common_memory_management_impl
PRIVATE
    mld_common_gpu_model
    mld_common_gpu_info
)

add_subdirectory(memory_management)
add_subdirectory(task)
add_subdirectory(default)
add_subdirectory(kernels)
add_subdirectory(selectors)

