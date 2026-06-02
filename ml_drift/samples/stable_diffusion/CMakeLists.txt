add_library(mld_samples_stable_diffusion_autoencoder_kl_builder
    autoencoder_kl_builder.cc
    autoencoder_kl_builder.h
)
target_include_directories(mld_samples_stable_diffusion_autoencoder_kl_builder PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_autoencoder_kl_builder PUBLIC
    mld_samples_stable_diffusion_util
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_types
)

add_library(mld_samples_stable_diffusion_bpe_tokenizer
    bpe_tokenizer.cc
    bpe_tokenizer.h
)
target_include_directories(mld_samples_stable_diffusion_bpe_tokenizer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_bpe_tokenizer PRIVATE
    absl::log
    absl::strings
    re2
)

add_library(mld_samples_stable_diffusion_text_guidance_builder
    text_guidance_builder.cc
    text_guidance_builder.h
)
target_include_directories(mld_samples_stable_diffusion_text_guidance_builder PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_text_guidance_builder PUBLIC
    mld_samples_stable_diffusion_model_data_loader
    mld_samples_stable_diffusion_util
    absl::log
    mld_common_data_type
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_types
    mld_common_kernels_reduce
    mld_common_task_gpu_operation
    tflite_schema
)

add_library(mld_samples_stable_diffusion_unet_builder
    unet_builder.cc
    unet_builder.h
)
target_include_directories(mld_samples_stable_diffusion_unet_builder PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_unet_builder PUBLIC
    mld_samples_stable_diffusion_util
    absl::log
    absl::strings
    absl::span
    mld_common_gpu_info
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_model
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_types
)

add_library(mld_samples_stable_diffusion_util
    util.cc
    util.h
)
target_include_directories(mld_samples_stable_diffusion_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_util PUBLIC
    XNNPACK
    absl::strings
    absl::span
    mld_common_data_type
    mld_common_gpu_info
    mld_common_operations
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_types
    mld_common_task_gpu_operation
    mld_common_task_tensor_desc
)

add_library(mld_samples_stable_diffusion_model_data_loader
    model_data_loader.cc
    model_data_loader.h
)
target_include_directories(mld_samples_stable_diffusion_model_data_loader PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_model_data_loader PUBLIC
    mld_samples_stable_diffusion_util
    absl::flat_hash_map
    absl::log
    absl::memory
    absl::strings
    absl::span
    flatbuffers
    mld_common_status
    mld_common_types
    tflite_schema
)

add_executable(sd_gpu_opencl
    sd_gpu_opencl.cc
)
target_link_libraries(sd_gpu_opencl PUBLIC
    mld_samples_stable_diffusion_autoencoder_kl_builder
    mld_samples_stable_diffusion_bpe_tokenizer
    mld_samples_stable_diffusion_text_guidance_builder
    mld_samples_stable_diffusion_unet_builder
    mld_samples_stable_diffusion_util
    mld_cl_cl_command_queue
    mld_cl_cl_operation
    mld_cl_environment
    mld_cl_inference_context
    mld_cl_opencl_wrapper
    mld_cl_tensor
    mld_common_data_type
    mld_common_gpu_model
    mld_common_gpu_model_builder
    mld_common_model_hints
    mld_common_operations
    mld_common_precision
    mld_common_shape
    mld_common_status
    mld_common_tensor
    mld_common_types
    mld_common_kernels_elementwise
    mld_common_kernels_random_philox
    mld_common_task_gpu_operation
    mld_common_task_profiling_info
    mld_common_task_tensor_desc
    mld_cl_default_util
)
