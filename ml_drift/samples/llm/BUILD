# Copyright 2025 The ML Drift Authors.
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
load("//devtools/python/bazel:pytype.bzl", "pytype_strict_binary")

package(licenses = ["notice"])

# Extracts weights from a HuggingFace model (safetensors).
# bazel run //ml_drift/samples/llm:extract_weights_hf -- --model_path=/tmp/model.safetensors --output_dir=/tmp/extracted/
pytype_strict_binary(
    name = "extract_weights_hf",
    srcs = ["extract_weights_hf.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch",
        "//third_party/py/torch:pytorch",
    ],
)

cc_library(
    name = "llm_tensor_loader",
    hdrs = ["llm_tensor_loader.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "llm_file_tensor_loader",
    srcs = ["llm_file_tensor_loader.cc"],
    hdrs = ["llm_file_tensor_loader.h"],
    deps = [
        ":llm_tensor_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "llm_file_tensor_loader_test",
    srcs = ["llm_file_tensor_loader_test.cc"],
    deps = [
        ":llm_file_tensor_loader",
        ":llm_tensor_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "llm_dummy_tensor_loader",
    srcs = ["llm_dummy_tensor_loader.cc"],
    hdrs = ["llm_dummy_tensor_loader.h"],
    deps = [
        ":llm_tensor_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "llm_dummy_tensor_loader_test",
    srcs = ["llm_dummy_tensor_loader_test.cc"],
    deps = [
        ":llm_dummy_tensor_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "llm_config",
    hdrs = ["llm_config.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "llm_weights_loader",
    srcs = ["llm_weights_loader.cc"],
    hdrs = ["llm_weights_loader.h"],
    deps = [
        ":llm_tensor_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "llm_weights_loader_test",
    srcs = ["llm_weights_loader_test.cc"],
    deps = [
        ":llm_dummy_tensor_loader",
        ":llm_weights_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gemma3_model_builder",
    srcs = ["gemma3_model_builder.cc"],
    hdrs = ["gemma3_model_builder.h"],
    deps = [
        ":llm_config",
        ":llm_tensor_loader",
        ":llm_weights_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "gemma3_model_builder_test",
    srcs = ["gemma3_model_builder_test.cc"],
    deps = [
        ":gemma3_model_builder",
        ":llm_config",
        ":llm_dummy_tensor_loader",
        ":llm_tensor_loader",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gemma4_model_builder",
    srcs = ["gemma4_model_builder.cc"],
    hdrs = ["gemma4_model_builder.h"],
    deps = [
        ":llm_config",
        ":llm_tensor_loader",
        ":llm_weights_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "gemma4_model_builder_test",
    srcs = ["gemma4_model_builder_test.cc"],
    deps = [
        ":gemma4_model_builder",
        ":llm_config",
        ":llm_dummy_tensor_loader",
        ":llm_tensor_loader",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "qwen3_model_builder",
    srcs = ["qwen3_model_builder.cc"],
    hdrs = ["qwen3_model_builder.h"],
    deps = [
        ":llm_config",
        ":llm_tensor_loader",
        ":llm_weights_loader",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "qwen3_model_builder_test",
    srcs = ["qwen3_model_builder_test.cc"],
    deps = [
        ":llm_config",
        ":llm_dummy_tensor_loader",
        ":llm_tensor_loader",
        ":qwen3_model_builder",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "llm_runner",
    srcs = ["llm_runner.cc"],
    hdrs = ["llm_runner.h"],
    deps = [
        ":llm_config",
        "//third_party/gloop/base:init_google",
        "//third_party/gloop/base:log_file_flags",
        "//third_party/sentencepiece/src:sentencepiece_processor",
        "//third_party/tokenizers_cpp:huggingface_tokenizer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

# bazel run -c opt //ml_drift/samples/llm:llm_runner_webgpu -- --prompt="Hello world." --model=gemma3:1b --weights_path=/tmp/extracted/
cc_binary(
    name = "llm_runner_webgpu",
    srcs = ["llm_webgpu.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":gemma3_model_builder",
        ":gemma4_model_builder",
        ":llm_config",
        ":llm_file_tensor_loader",
        ":llm_runner",
        ":llm_tensor_loader",
        ":qwen3_model_builder",
        "//devtools/rust:rust_okay_here",  # copybara:strip
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/webgpu:compute_task",
        "//ml_drift/webgpu:environment",
        "//ml_drift/webgpu:inference_context",
        "//ml_drift/webgpu:instance",
        "//ml_drift/webgpu:spatial_tensor",
        "//ml_drift/webgpu:webgpu_api_util",
        "//ml_drift/webgpu:webgpu_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

# bazel run -c opt //ml_drift/samples/llm:llm_runner_opencl -- --prompt="Hello world." --model=gemma3:1b --weights_path=/tmp/extracted/
cc_binary(
    name = "llm_runner_opencl",
    srcs = ["llm_opencl.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":gemma3_model_builder",
        ":gemma4_model_builder",
        ":llm_config",
        ":llm_file_tensor_loader",
        ":llm_runner",
        ":llm_tensor_loader",
        ":qwen3_model_builder",
        "//base",
        "//devtools/rust:rust_okay_here",  # copybara:strip
        "//ml_drift/cl:cl_command_queue",
        "//ml_drift/cl:cl_event",
        "//ml_drift/cl:cl_operation",
        "//ml_drift/cl:environment",
        "//ml_drift/cl:inference_context",
        "//ml_drift/cl:opencl_wrapper",
        "//ml_drift/cl:tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//third_party/gloop/base:init_google",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + select({
        "//conditions:default": [],
    }),
)

objc_library(
    name = "llm_runner_metal_lib",
    srcs = ["llm_metal.cc"],
    copts = [
        "-x",
        "objective-c++",
    ],
    deps = [
        ":gemma3_model_builder",
        ":gemma4_model_builder",
        ":llm_config",
        ":llm_file_tensor_loader",
        ":llm_runner",
        ":llm_tensor_loader",
        ":qwen3_model_builder",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/metal:common",
        "//ml_drift/metal:environment",
        "//ml_drift/metal:inference_context",
        "//ml_drift/metal:metal_spatial_tensor",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

# bazel run -c opt //ml_drift/samples/llm:llm_runner_metal -- --prompt="Hello world." --model=gemma3:1b --weights_path=/tmp/extracted/
cc_binary(
    name = "llm_runner_metal",
    deps = [
        ":llm_runner",
        ":llm_runner_metal_lib",
        "//devtools/rust:rust_okay_here",  # copybara:strip
        "//third_party/apple_frameworks:Metal",
    ],
)

alias(
    name = "llm_performance_profiling",
    actual = select({
        "//tools/cc_target_os:apple": ":llm_performance_profiling_metal",
        "//conditions:default": ":llm_performance_profiling_webgpu",
    }),
)

# bazel run -c opt //ml_drift/samples/llm:llm_performance_profiling_webgpu -- --model=gemma4:12b
cc_binary(
    name = "llm_performance_profiling_webgpu",
    srcs = ["llm_performance_profiling.cc"],
    copts = ["-DML_DRIFT_LLM_PROFILE_WEBGPU=1"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":gemma3_model_builder",
        ":gemma4_model_builder",
        ":llm_config",
        ":llm_file_tensor_loader",
        ":llm_tensor_loader",
        ":qwen3_model_builder",
        "//devtools/rust:rust_okay_here",  # copybara:strip
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/webgpu:environment",
        "//ml_drift/webgpu:inference_context",
        "//ml_drift/webgpu:spatial_tensor",
        "//ml_drift/webgpu:webgpu_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

# bazel run -c opt //ml_drift/samples/llm:llm_performance_profiling_opencl -- --model=gemma4:12b
cc_binary(
    name = "llm_performance_profiling_opencl",
    srcs = ["llm_performance_profiling.cc"],
    copts = ["-DML_DRIFT_LLM_PROFILE_OPENCL=1"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":gemma3_model_builder",
        ":gemma4_model_builder",
        ":llm_config",
        ":llm_file_tensor_loader",
        ":llm_tensor_loader",
        ":qwen3_model_builder",
        "//devtools/rust:rust_okay_here",  # copybara:strip
        "//ml_drift/cl:cl_command_queue",
        "//ml_drift/cl:environment",
        "//ml_drift/cl:inference_context",
        "//ml_drift/cl:opencl_wrapper",
        "//ml_drift/cl:tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ] + select({
        "//conditions:default": [],
    }),
)

objc_library(
    name = "llm_performance_profiling_metal_lib",
    srcs = ["llm_performance_profiling.cc"],
    copts = [
        "-x",
        "objective-c++",
        "-DML_DRIFT_LLM_PROFILE_METAL=1",
    ],
    deps = [
        ":gemma3_model_builder",
        ":gemma4_model_builder",
        ":llm_config",
        ":llm_file_tensor_loader",
        ":llm_tensor_loader",
        ":qwen3_model_builder",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/metal:common",
        "//ml_drift/metal:environment",
        "//ml_drift/metal:inference_context",
        "//ml_drift/metal:metal_spatial_tensor",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

# bazel run -c opt //ml_drift/samples/llm:llm_performance_profiling_metal -- --model=gemma4:12b
cc_binary(
    name = "llm_performance_profiling_metal",
    deps = [
        ":llm_performance_profiling_metal_lib",
        "//devtools/rust:rust_okay_here",  # copybara:strip
        "//third_party/apple_frameworks:Metal",
    ],
)
