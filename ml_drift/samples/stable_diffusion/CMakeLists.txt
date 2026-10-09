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

add_library(mld_samples_stable_diffusion_autoencoder_kl_builder
    autoencoder_kl_builder.cc
    autoencoder_kl_builder.h
)
target_include_directories(mld_samples_stable_diffusion_autoencoder_kl_builder PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_autoencoder_kl_builder
  PUBLIC
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_operations
    mld_common_shape
    mld_common_types
    mld_samples_stable_diffusion_util
)

add_library(mld_samples_stable_diffusion_bpe_tokenizer
    bpe_tokenizer.cc
    bpe_tokenizer.h
)
target_include_directories(mld_samples_stable_diffusion_bpe_tokenizer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_bpe_tokenizer
  PRIVATE
    absl::log
    absl::strings
    re2
)

add_library(mld_samples_stable_diffusion_text_guidance_builder
    text_guidance_builder.cc
    text_guidance_builder.h
)
target_include_directories(mld_samples_stable_diffusion_text_guidance_builder PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_text_guidance_builder
  PUBLIC
    absl::log
    absl::status
    absl::status_macros
    absl::statusor
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_kernels_reduce
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_tensor
    mld_common_types
    mld_samples_stable_diffusion_model_data_loader
    mld_samples_stable_diffusion_util
    tflite_schema
)

add_library(mld_samples_stable_diffusion_unet_builder
    unet_builder.cc
    unet_builder.h
)
target_include_directories(mld_samples_stable_diffusion_unet_builder PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_unet_builder
  PUBLIC
    absl::log
    absl::span
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_operations
    mld_common_shape
    mld_common_types
    mld_samples_stable_diffusion_util
)

add_library(mld_samples_stable_diffusion_util
    util.cc
    util.h
)
target_include_directories(mld_samples_stable_diffusion_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_util
  PUBLIC
    absl::span
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_common_data_type
    mld_common_gpu_info
    mld_common_operations
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_common_types
    XNNPACK
)

add_library(mld_samples_stable_diffusion_model_data_loader
    model_data_loader.cc
    model_data_loader.h
)
target_include_directories(mld_samples_stable_diffusion_model_data_loader PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_model_data_loader
  PUBLIC
    absl::flat_hash_map
    absl::log
    absl::memory
    absl::span
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    flatbuffers
    mld_common_types
    mld_samples_stable_diffusion_util
    tflite_schema
)


if(ML_DRIFT_API STREQUAL "cl")
add_library(mld_samples_stable_diffusion_diffuser
    diffuser.cc
    diffuser.h
)
target_include_directories(mld_samples_stable_diffusion_diffuser PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_diffuser
  PUBLIC
    absl::log
    absl::memory
    absl::status
    absl::status_macros
    absl::statusor
    absl::strings
    mld_cl_cl_command_queue
    mld_cl_cl_operation
    mld_cl_environment
    mld_cl_inference_context
    mld_cl_opencl_wrapper
    mld_cl_tensor
    mld_common_data_type
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_kernels_cast
    mld_common_kernels_random_philox
    mld_common_model
    mld_common_model_hints
    mld_common_precision
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_common_types
    mld_samples_stable_diffusion_autoencoder_kl_builder
    mld_samples_stable_diffusion_bpe_tokenizer
    mld_samples_stable_diffusion_ops_diffusion_step
    mld_samples_stable_diffusion_ops_temb_generation
    mld_samples_stable_diffusion_text_guidance_builder
    mld_samples_stable_diffusion_unet_builder
    mld_samples_stable_diffusion_util
)

add_executable(mld_samples_stable_diffusion_sd_gpu_opencl
    sd_gpu_opencl.cc
)
add_executable(sd_gpu_opencl ALIAS mld_samples_stable_diffusion_sd_gpu_opencl)
target_include_directories(mld_samples_stable_diffusion_sd_gpu_opencl PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_sd_gpu_opencl
  PUBLIC
    absl::status
    absl::status_macros
    absl::strings
    mld_cl_opencl_wrapper
    mld_samples_stable_diffusion_diffuser
    mld_samples_stable_diffusion_util
)
target_link_options(mld_samples_stable_diffusion_sd_gpu_opencl PRIVATE
    "-Wl,--whole-archive"
    "$<TARGET_FILE:mld_samples_stable_diffusion_ops_diffusion_step>"
    "$<TARGET_FILE:mld_samples_stable_diffusion_ops_temb_generation>"
    "-Wl,--no-whole-archive"
)
endif()

# The WebGPU sample executable links real Dawn symbols. The cached Dawn tarball
# only ships a host (Linux x86_64) libdawn, so it cannot be linked for a
# cross-compiled build.
if(ML_DRIFT_API STREQUAL "wgpu" AND NOT CMAKE_CROSSCOMPILING)
add_executable(mld_samples_stable_diffusion_sd_gpu_webgpu
    sd_gpu_webgpu.cc
)
add_executable(sd_gpu_webgpu ALIAS mld_samples_stable_diffusion_sd_gpu_webgpu)
target_include_directories(mld_samples_stable_diffusion_sd_gpu_webgpu PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_sd_gpu_webgpu
  PUBLIC
    absl::status
    absl::status_macros
    absl::time
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_kernels_cast
    mld_common_kernels_random_philox
    mld_common_model
    mld_common_model_hints
    mld_common_precision
    mld_common_shape
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
    mld_common_tensor
    mld_samples_stable_diffusion_autoencoder_kl_builder
    mld_samples_stable_diffusion_bpe_tokenizer
    mld_samples_stable_diffusion_ops_diffusion_step
    mld_samples_stable_diffusion_ops_temb_generation
    mld_samples_stable_diffusion_text_guidance_builder
    mld_samples_stable_diffusion_unet_builder
    mld_samples_stable_diffusion_util
    mld_webgpu_compute_task
    mld_webgpu_environment
    mld_webgpu_inference_context
    mld_webgpu_spatial_tensor
    mld_webgpu_webgpu_api_util
    mld_webgpu_webgpu_headers
)
target_link_options(mld_samples_stable_diffusion_sd_gpu_webgpu PRIVATE
    "-Wl,--whole-archive"
    "$<TARGET_FILE:mld_samples_stable_diffusion_ops_diffusion_step>"
    "$<TARGET_FILE:mld_samples_stable_diffusion_ops_temb_generation>"
    "-Wl,--no-whole-archive"
)
endif()

add_subdirectory(ops)
