# Copyright 2024 The ML Drift Authors.
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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//ml_drift/common:build_config.bzl", "mld_gpu_tests_tags", "mld_platform_alias")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

config_setting(
    name = "opencl_delegate_no_gl",
    values = {"copt": "-DCL_DELEGATE_NO_GL"},
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":gpu_object",
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:status",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    linkstatic = True,
    tags = mld_gpu_tests_tags(),
    deps = [
        ":buffer",
        ":cl_test",
        "//ml_drift/common:types",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cl_test",
    testonly = 1,
    hdrs = ["cl_test.h"],
    deps = [
        ":environment",
        ":opencl_wrapper",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "cl_arguments",
    srcs = ["cl_arguments.cc"],
    hdrs = ["cl_arguments.h"],
    deps = [
        ":buffer",
        ":cl_context",
        ":gpu_object",
        ":opencl_wrapper",
        ":qcom_thin_filter",
        ":tensor",
        ":util",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:qcom_thin_filter_desc",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "cl_arguments_test",
    srcs = ["cl_arguments_test.cc"],
    linkstatic = True,
    tags = mld_gpu_tests_tags(),
    deps = [
        ":cl_arguments",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cl_api_util",
    hdrs = ["cl_api_util.h"],
    deps = [
        ":cl_command_queue",
        "//ml_drift/common:status",
    ],
)

cc_library(
    name = "cl_command_buffer",
    srcs = ["cl_command_buffer.cc"],
    hdrs = ["cl_command_buffer.h"],
    deps = [
        ":cl_command_queue",
        ":cl_event",
        ":cl_operation",
        ":command_buffer",
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cl_command_queue",
    srcs = ["cl_command_queue.cc"],
    hdrs = ["cl_command_queue.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":cl_event",
        ":cl_kernel",
        ":opencl_wrapper",
        ":util",
        ":util_types",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common/task:profiling_info",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "cl_context",
    srcs = ["cl_context.cc"],
    hdrs = ["cl_context.h"],
    deps = [
        ":cl_device",
        ":cl_image_format",
        ":opencl_wrapper",
        ":util",
        ":util_types",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cl_device",
    srcs = ["cl_device.cc"],
    hdrs = ["cl_device.h"],
    deps = [
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "cl_errors",
    hdrs = ["cl_errors.h"],
    deps = [
        ":util",
        "//ml_drift/common:status",
        "@opencl_headers",
    ],
)

cc_library(
    name = "cl_event",
    srcs = ["cl_event.cc"],
    hdrs = ["cl_event.h"],
    deps = [
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cl_image_format",
    srcs = ["cl_image_format.cc"],
    hdrs = ["cl_image_format.h"],
    deps = [
        ":opencl_wrapper",
        "//ml_drift/common:data_type",
    ],
)

cc_library(
    name = "cl_kernel",
    srcs = ["cl_kernel.cc"],
    hdrs = ["cl_kernel.h"],
    deps = [
        ":cl_program",
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:status",
        "@com_google_absl//absl/strings",
        "@opencl_headers",
    ],
)

cc_library(
    name = "cl_memory",
    srcs = ["cl_memory.cc"],
    hdrs = ["cl_memory.h"],
    deps = [
        ":opencl_wrapper",
        "//ml_drift/common:access_type",
        "@opencl_headers",
    ],
)

cc_library(
    name = "cl_operation",
    srcs = ["cl_operation.cc"],
    hdrs = ["cl_operation.h"],
    deps = [
        ":buffer",
        ":cl_arguments",
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_event",
        ":cl_kernel",
        ":opencl_wrapper",
        ":program_cache",
        ":tensor",
        ":util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "cl_program",
    srcs = ["cl_program.cc"],
    hdrs = ["cl_program.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common/task:compiler_options",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_library(
    name = "command_buffer",
    hdrs = ["command_buffer.h"],
    deps = [
        ":cl_command_queue",
        ":cl_event",
        ":cl_operation",
        "//ml_drift/common:status",
    ],
)

flatbuffer_cc_library(
    name = "compiled_program_cache_cc_fbs",
    srcs = ["compiled_program_cache.fbs"],
    flatc_args = ["--scoped-enums"],
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = ["environment.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_program",
        ":program_cache",
        ":util",
        ":util_types",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:status",
        "//ml_drift/common/task:tensor_desc",
        "@XNNPACK",
    ],
)

cc_library(
    name = "gpu_object",
    hdrs = ["gpu_object.h"],
    deps = [
        ":opencl_wrapper",
        "//ml_drift/common:status",
        "//ml_drift/common/task:gpu_object_desc",
    ],
)

cc_library(
    name = "inference_context",
    srcs = ["inference_context.cc"],
    hdrs = ["inference_context.h"],
    deps = [
        ":buffer",
        ":cl_command_buffer",
        ":cl_command_queue",
        ":cl_device",
        ":cl_event",
        ":cl_operation",
        ":command_buffer",
        ":environment",
        ":memory_manager",
        ":opencl_wrapper",
        ":program_cache",
        ":qcom_command_buffer",
        ":serialization_cc_fbs",
        ":tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:gpu_model_cc_fbs",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:ir_model_util",
        "//ml_drift/common:model",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:serialization_base",
        "//ml_drift/common/task:serialization_base_cc_fbs",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "inference_context_test",
    srcs = ["inference_context_test.cc"],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        ":cl_test",
        ":inference_context",
        ":tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory_manager",
    srcs = ["memory_manager.cc"],
    hdrs = ["memory_manager.h"],
    deps = [
        ":buffer",
        ":cl_context",
        ":environment",
        ":tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:memory_management",
        "//ml_drift/common:model",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "opencl_wrapper",
    srcs = ["opencl_wrapper.cc"],
    hdrs = ["opencl_wrapper.h"],
    linkopts = select({
        "//ml_drift:android": [
            "-ldl",  # opencl_wrapper calls dlopen()
            "-lm",
        ],
        "//conditions:default": ["-ldl"],  # opencl_wrapper calls dlopen()
    }),
    deps = [
        "//ml_drift/common:status",
        "@com_google_absl//absl/strings",
        "@opencl_headers",
    ] + mld_platform_alias("qcom_wrapper", "//ml_drift/cl/"),
)

cc_library(
    name = "program_cache",
    srcs = ["program_cache.cc"],
    hdrs = ["program_cache.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":cl_kernel",
        ":cl_program",
        ":compiled_program_cache_cc_fbs",
        "//ml_drift/common:status",
        "//ml_drift/common/task:compiler_options",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@farmhash_archive//:farmhash",
        "@flatbuffers",
    ],
)

cc_library(
    name = "qcom_command_buffer",
    hdrs = ["qcom_command_buffer.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":command_buffer",
        "//ml_drift/common:status",
    ] + mld_platform_alias("qcom_command_buffer", "//ml_drift/cl/"),
)

cc_library(
    name = "qcom_thin_filter",
    srcs = ["qcom_thin_filter.cc"],
    hdrs = ["qcom_thin_filter.h"],
    deps = [
        ":cl_context",
        ":gpu_object",
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:status",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:qcom_thin_filter_desc",
        "@opencl_headers",
    ],
)

flatbuffer_cc_library(
    name = "serialization_cc_fbs",
    srcs = ["serialization.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I .",
    ],
    include_paths = [
        "ml_drift",
        "ml_drift/common",
        "ml_drift/common/task",
    ],
    includes = [
        "//ml_drift/common:gpu_model_cc_fbs_includes",
        "//ml_drift/common/task:serialization_base_cc_fbs_includes",
    ],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":cl_image_format",
        ":cl_memory",
        ":gpu_object",
        ":opencl_wrapper",
        ":util",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/strings",
        "@opencl_headers",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    linkstatic = 1,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":environment",
        ":opencl_wrapper",
        ":tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":opencl_wrapper",
        ":util_types",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + mld_platform_alias("util", "//ml_drift/cl/"),
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    linkstatic = 1,
    deps = [
        ":util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util_types",
    hdrs = ["util_types.h"],
)

cc_test(
    name = "memory_manager_test",
    srcs = ["memory_manager_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cl_test",
        ":memory_manager",
        ":tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:shape",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
