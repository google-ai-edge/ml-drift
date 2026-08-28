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

# --- Flatbuffer generation for serialization_base.fbs ---
mld_generate_flatbuffers_cpp(mld_common_task_serialization_base_cc_fbs
  "${CMAKE_CURRENT_SOURCE_DIR}/serialization_base.fbs"
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ml_drift/common/task"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS flatc_host_tool
)
# --- End Flatbuffer generation ---

add_library(mld_common_task_serialization_base serialization_base.cc serialization_base.h)
target_include_directories(mld_common_task_serialization_base PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${FLATBUFFERS_GEN_INCLUDE_DIR}"
)
target_link_libraries(mld_common_task_serialization_base PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  flatbuffers
  mld_common_data_type
  mld_common_shape
  mld_common_task_gpu_operation
  mld_common_task_serialization_base_cc_fbs
  mld_common_task_tensor_desc
  mld_common_types
PRIVATE
  mld_common_access_type
  mld_common_task_arguments
  mld_common_task_buffer_desc
  mld_common_task_gpu_object_desc
)
add_dependencies(mld_common_task_serialization_base mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_profiling_info profiling_info.cc profiling_info.h)
target_include_directories(mld_common_task_profiling_info PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_profiling_info PUBLIC
  mld_common_shape
  absl::time
)

add_library(mld_common_task_gpu_object_desc INTERFACE
  gpu_object_desc.h
)
target_include_directories(mld_common_task_gpu_object_desc INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_gpu_object_desc INTERFACE
  flatbuffers
  mld_common_task_serialization_base_cc_fbs
)
add_dependencies(mld_common_task_gpu_object_desc mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_qcom_thin_filter_desc qcom_thin_filter_desc.cc qcom_thin_filter_desc.h)
target_include_directories(mld_common_task_qcom_thin_filter_desc PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_qcom_thin_filter_desc PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_gpu_info
  mld_common_task_gpu_object_desc
PRIVATE
  absl::strings
)

add_library(mld_common_task_tensor_desc tensor_desc.cc tensor_desc.h)
target_include_directories(mld_common_task_tensor_desc PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${FLATBUFFERS_GEN_INCLUDE_DIR}"
)
target_link_libraries(mld_common_task_tensor_desc PUBLIC
  absl::span
  absl::status
  absl::status_macros
  absl::statusor
  flatbuffers
  mld_common_data_type
  mld_common_shape
  mld_common_task_gpu_object_desc
  mld_common_task_serialization_base_cc_fbs
  mld_common_tensor
  mld_common_types
  mld_common_util
PRIVATE
  absl::strings
  mld_common_task_util
)
add_dependencies(mld_common_task_tensor_desc mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_gpu_tensor INTERFACE
    gpu_tensor.h
)
target_include_directories(mld_common_task_gpu_tensor INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_gpu_tensor INTERFACE
    mld_common_task_tensor_desc
    mld_common_shape
)

add_library(mld_common_task_buffer_desc buffer_desc.cc buffer_desc.h)
target_include_directories(mld_common_task_buffer_desc PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  "${FLATBUFFERS_GEN_INCLUDE_DIR}"
)
target_link_libraries(mld_common_task_buffer_desc PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_data_type
  mld_common_gpu_info
  mld_common_task_gpu_object_desc
PRIVATE
  absl::strings
  mld_common_task_util
)
add_dependencies(mld_common_task_buffer_desc mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_arguments arguments.cc arguments.h)
target_include_directories(mld_common_task_arguments PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_arguments PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_access_type
  mld_common_data_type
  mld_common_gpu_info
  mld_common_task_gpu_object_desc
  mld_common_types
PRIVATE
  absl::flat_hash_set
  absl::str_format
  absl::strings
  flatbuffers
  mld_common_task_buffer_desc
  mld_common_task_serialization_base_cc_fbs
  mld_common_task_util
)
add_dependencies(mld_common_task_arguments mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_compiler_options compiler_options.cc compiler_options.h)
target_include_directories(mld_common_task_compiler_options PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)

add_library(mld_common_task_tuning_type INTERFACE)
target_include_directories(mld_common_task_tuning_type INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)

add_library(mld_common_task_work_group_picking work_group_picking.cc work_group_picking.h)
target_include_directories(mld_common_task_work_group_picking PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_work_group_picking PUBLIC
  mld_common_task_tuning_type
  mld_common_gpu_info
  mld_common_kernel_info
  mld_common_types
PRIVATE
  mld_common_util
)

add_library(mld_common_task_gpu_operation
  gpu_operation.cc
  gpu_operation.h
)
target_include_directories(mld_common_task_gpu_operation PUBLIC
  $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>
  $<TARGET_PROPERTY:FP16,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(mld_common_task_gpu_operation PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  flatbuffers
  mld_common_data_type
  mld_common_gpu_info
  mld_common_kernel_info
  mld_common_precision
  mld_common_shape
  mld_common_task_arguments
  mld_common_task_buffer_desc
  mld_common_task_compiler_options
  mld_common_task_gpu_tensor
  mld_common_task_serialization_base_cc_fbs
  mld_common_task_tensor_desc
  mld_common_task_tuning_type
  mld_common_types
PRIVATE
  absl::strings
  FP16
  mld_common_access_type
  mld_common_task_gpu_object_desc
  mld_common_task_util
  mld_common_task_wave_matrix_util
  mld_common_task_wave_memory_util
  mld_common_task_work_group_picking
  mld_common_util
)
add_dependencies(mld_common_task_gpu_operation mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_weights_layout
  weights_layout.cc
  weights_layout.h
)
target_include_directories(mld_common_task_weights_layout PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_weights_layout PUBLIC
  mld_common_data_type
  mld_common_shape
PRIVATE
  absl::status
  absl::status_macros
  absl::statusor
  absl::strings
  flatbuffers
  mld_common_task_serialization_base_cc_fbs
  mld_common_tensor
  mld_common_types
  mld_common_util
  XNNPACK
)
add_dependencies(mld_common_task_weights_layout mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_wave_matrix_util wave_matrix_util.cc wave_matrix_util.h)
target_include_directories(mld_common_task_wave_matrix_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_wave_matrix_util PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_gpu_info
PRIVATE
  absl::flat_hash_map
  absl::strings
  mld_common_data_type
  mld_common_task_util
)

add_library(mld_common_task_wave_memory_util wave_memory_util.cc wave_memory_util.h)
target_include_directories(mld_common_task_wave_memory_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_wave_memory_util PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_data_type
  mld_common_gpu_info
  mld_common_shape
  mld_common_task_arguments
  mld_common_task_buffer_desc
  mld_common_task_compiler_options
  mld_common_task_weights_layout
PRIVATE
  absl::strings
  mld_common_task_gpu_object_desc
  mld_common_task_util
  mld_common_task_weights_conversion
  mld_common_util
)

add_library(mld_common_task_weights_conversion
  weights_conversion.cc
  weights_conversion.h
)
target_include_directories(mld_common_task_weights_conversion PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_weights_conversion PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_data_type
  mld_common_shape
  mld_common_task_tensor_desc
  mld_common_task_weights_layout
  mld_common_tensor
  mld_common_types
PRIVATE
  absl::check
  absl::log
  absl::strings
  flatbuffers
  mld_common_task_serialization_base_cc_fbs
  mld_common_util
  XNNPACK
)
add_dependencies(mld_common_task_weights_conversion mld_common_task_serialization_base_cc_fbs_gen)

add_library(mld_common_task_util util.cc util.h)
target_include_directories(mld_common_task_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_compile_definitions(mld_common_task_util PUBLIC ML_DRIFT_ENABLE_VULKAN)
target_link_libraries(mld_common_task_util PUBLIC
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_data_type
  mld_common_gpu_info
  mld_common_precision
  mld_common_task_gpu_object_desc
  mld_common_types
PRIVATE
  absl::flat_hash_map
  absl::strings
  mld_common_util
)

add_library(mld_common_task_testing_util testing_util.cc testing_util.h)
target_include_directories(mld_common_task_testing_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_testing_util PUBLIC
  absl::flat_hash_map
  absl::log
  absl::random_bit_gen_ref
  absl::random_random
  absl::status
  absl::status_macros
  absl::statusor
  absl::strings
  gmock
  gtest
  mld_common_data_type
  mld_common_gpu_info
  mld_common_gpu_model
  mld_common_operations
  mld_common_precision
  mld_common_shape
  mld_common_task_gpu_operation
  mld_common_task_profiling_info
  mld_common_task_tensor_desc
  mld_common_tensor
  mld_common_util
)

add_library(mld_common_task_testing_ref_ops testing_ref_ops.cc testing_ref_ops.h)
target_include_directories(mld_common_task_testing_ref_ops PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_task_testing_ref_ops PUBLIC
  absl::check
  absl::log
  absl::status
  absl::status_macros
  absl::statusor
  mld_common_data_type
  mld_common_operations
  mld_common_shape
  mld_common_task_gpu_operation
  mld_common_task_testing_util
  mld_common_tensor
  mld_common_util
  mld_common_winograd_util
  XNNPACK
)
