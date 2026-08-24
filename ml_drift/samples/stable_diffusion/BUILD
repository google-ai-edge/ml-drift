# Copyright 2023 The ML Drift Authors.
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

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test", "objc_library")

package(licenses = ["notice"])

cc_library(
    name = "libimagegenerator_gpu_so",
    srcs = [":libimagegenerator_gpu.so"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    visibility = ["//third_party/mediapipe/tasks/benchmark/cc/vision/image_generator/diffuser:__pkg__"],
)

# build -c opt --config android_arm64 --copt -Os --linkopt -s --strip always
cc_binary(
    name = "libimagegenerator_gpu.so",
    linkopts = [
        "-Wl,-soname=libimagegenerator_gpu.so",
        "-fvisibility=hidden",
    ],
    linkshared = 1,
    visibility = [
        "//java/com/google/android/apps/xeno:__pkg__",
    ],
    deps = [":diffuser_gpu_lib"],
)

cc_library(
    name = "diffuser_gpu_lib",
    srcs = ["diffuser_gpu.cc"],
    hdrs = ["diffuser_gpu.h"],
    deps = [
        ":diffuser",
        "//ml_drift/cl:util_types",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_library(
    name = "autoencoder_kl_builder",
    srcs = ["autoencoder_kl_builder.cc"],
    hdrs = ["autoencoder_kl_builder.h"],
    visibility = ["//ml_drift/samples/stable_diffusion_web:__subpackages__"],
    deps = [
        ":util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "bpe_tokenizer",
    srcs = ["bpe_tokenizer.cc"],
    hdrs = ["bpe_tokenizer.h"],
    visibility = [
        "//hardware/gchips/sw_experiences/generative_imaging:__subpackages__",
        "//ml_drift/samples/stable_diffusion:__subpackages__",
        "//ml_drift/samples/stable_diffusion_web:__subpackages__",
        "//third_party/odml/infra/genai/inference/calculators:__subpackages__",
    ],
    deps = [
        "@com_github_google_re2//:re2",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "sd_gpu_metal_bin",
    deps = [
        ":sd_gpu_metal",
        "//third_party/apple_frameworks:Metal",
        "//third_party/apple_frameworks:QuartzCore",
    ],
)

objc_library(
    name = "sd_gpu_metal",
    srcs = ["sd_gpu_metal.mm"],
    deps = [
        ":autoencoder_kl_builder",
        ":bpe_tokenizer",
        ":text_guidance_builder",
        ":unet_builder",
        ":util",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:mean_stddev_normalization",
        "//ml_drift/common/kernels:random_philox",
        "//ml_drift/common/selectors:convolution_selector",
        "//ml_drift/common/selectors:simple_selectors",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/transformations:model_transformations",
        "//ml_drift/metal:inference_context",
        "//ml_drift/metal:metal_device",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_binary(
    name = "sd_gpu_webgpu",
    srcs = ["sd_gpu_webgpu.cc"],
    deps = [
        ":autoencoder_kl_builder",
        ":bpe_tokenizer",
        ":text_guidance_builder",
        ":unet_builder",
        ":util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:random_philox",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/webgpu:compute_task",
        "//ml_drift/webgpu:environment",
        "//ml_drift/webgpu:inference_context",
        "//ml_drift/webgpu:spatial_tensor",
        "//ml_drift/webgpu:webgpu_api_util",
        "//ml_drift/webgpu:webgpu_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "sd_gpu_opencl",
    srcs = ["sd_gpu_opencl.cc"],
    deps = [
        ":autoencoder_kl_builder",
        ":bpe_tokenizer",
        ":text_guidance_builder",
        ":unet_builder",
        ":util",
        "//ml_drift/cl:cl_command_queue",
        "//ml_drift/cl:cl_operation",
        "//ml_drift/cl:environment",
        "//ml_drift/cl:inference_context",
        "//ml_drift/cl:opencl_wrapper",
        "//ml_drift/cl:tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:random_philox",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "text_guidance_builder",
    srcs = ["text_guidance_builder.cc"],
    hdrs = ["text_guidance_builder.h"],
    visibility = ["//ml_drift/samples/stable_diffusion_web:__subpackages__"],
    deps = [
        ":model_data_loader",
        ":util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:reduce",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "unet_builder",
    srcs = ["unet_builder.cc"],
    hdrs = ["unet_builder.h"],
    visibility = ["//ml_drift/samples/stable_diffusion_web:__subpackages__"],
    deps = [
        ":util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    visibility = [
        "//ml_drift/samples/stable_diffusion_web:__subpackages__",
        "//research/aimatter/diffusion/klgan:__pkg__",
    ],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "diffuser",
    srcs = ["diffuser.cc"],
    hdrs = ["diffuser.h"],
    deps = [
        ":autoencoder_kl_builder",
        ":bpe_tokenizer",
        ":text_guidance_builder",
        ":unet_builder",
        ":util",
        "//ml_drift/cl:cl_command_queue",
        "//ml_drift/cl:cl_operation",
        "//ml_drift/cl:environment",
        "//ml_drift/cl:inference_context",
        "//ml_drift/cl:opencl_wrapper",
        "//ml_drift/cl:tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:random_philox",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "model_data_loader",
    srcs = ["model_data_loader.cc"],
    hdrs = ["model_data_loader.h"],
    visibility = [
        "//research/aimatter/diffusion/klgan:__subpackages__",
        "//wireless/android/camera/diffusion/inference:__subpackages__",
        "@//ml_drift_delegate/tflite:__subpackages__",
    ],
    deps = [
        ":util",
        "//ml_drift/common:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
