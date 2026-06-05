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

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "arguments",
    srcs = ["arguments.cc"],
    hdrs = ["arguments.h"],
    deps = [
        ":buffer",
        ":gpu_object",
        ":spatial_tensor",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    deps = [
        ":gpu_object",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "compute_pipeline_cache",
    hdrs = ["compute_pipeline_cache.h"],
    deps = [
        ":preprocessor",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:executor",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@farmhash_archive//:farmhash",
    ],
)

cc_library(
    name = "webgpu_weights_manager",
    srcs = ["webgpu_weights_manager.cc"],
    hdrs = ["webgpu_weights_manager.h"],
    deps = [
        ":compute_task",
        ":environment",
        ":spatial_tensor",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:status",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "webgpu_weights_manager_test",
    srcs = ["webgpu_weights_manager_test.cc"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":execution_environment",
        ":inference_context",
        ":spatial_tensor",
        ":webgpu_api_util",
        ":webgpu_headers",
        ":webgpu_weights_manager",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "compute_task",
    srcs = ["compute_task.cc"],
    hdrs = ["compute_task.h"],
    deps = [
        ":arguments",
        ":buffer",
        ":compute_pipeline_cache",
        ":environment",
        ":preprocessor",
        ":spatial_tensor",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@farmhash_archive//:farmhash",
    ],
)

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    deps = [
        ":buffer",
        ":compute_task",
        ":environment",
        ":spatial_tensor",
        ":webgpu_headers",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common/kernels:conversion",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
    ],
)

cc_test(
    name = "converter_test",
    srcs = ["converter_test.cc"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":buffer",
        ":converter",
        ":spatial_tensor",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/webgpu/testing:webgpu_test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = ["environment.h"],
    defines = [
        "DAWN_ENABLE_BACKEND_VULKAN",
        "ML_DRIFT_DAWN_NATIVE",
    ],
    deps = [
        ":buffer",
        ":compute_pipeline_cache",
        ":instance",
        ":webgpu_headers",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common/task:tensor_desc",
        "@XNNPACK",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@dawn//:libdawn",
    ],
)

cc_library(
    name = "execution_environment",
    hdrs = ["execution_environment.h"],
    defines = [
        "DAWN_ENABLE_BACKEND_VULKAN",
        "ML_DRIFT_DAWN_NATIVE",
    ],
    deps = [
        ":environment",
        "@dawn//:libdawn",
    ],
)

cc_library(
    name = "gpu_object",
    hdrs = ["gpu_object.h"],
    deps = [
        ":webgpu_headers",
        "//ml_drift/common:status",
        "//ml_drift/common/task:gpu_object_desc",
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
    name = "inference_context",
    srcs = ["inference_context.cc"],
    hdrs = ["inference_context.h"],
    deps = [
        ":compute_task",
        ":environment",
        ":memory_manager",
        ":metrics_collector",
        ":serialization_cc_fbs",
        ":spatial_tensor",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_cc_fbs",
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
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":environment",
        ":inference_context",
        ":spatial_tensor",
        ":webgpu_headers",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "instance",
    srcs = ["instance.cc"],
    hdrs = ["instance.h"],
    deps = [
        ":webgpu_headers",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "memory_manager",
    srcs = ["memory_manager.cc"],
    hdrs = ["memory_manager.h"],
    deps = [
        ":buffer",
        ":environment",
        ":spatial_tensor",
        ":webgpu_headers",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:memory_management",
        "//ml_drift/common:model",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "metrics_collector",
    srcs = ["metrics_collector.cc"],
    hdrs = [
        "metrics_collector.h",
        "metrics_collector_internal.h",
    ],
    deps = [
        ":buffer",
        ":environment",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "preprocessor",
    srcs = ["preprocessor.cc"],
    hdrs = ["preprocessor.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "preprocessor_test",
    srcs = ["preprocessor_test.cc"],
    deps = [
        ":preprocessor",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "registry",
    hdrs = ["registry.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:model",
        "//ml_drift/common:status",
        "//ml_drift/common/kernels:reduce_parser",
        "//ml_drift/common/selectors:operation_selector",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "spatial_tensor",
    srcs = ["spatial_tensor.cc"],
    hdrs = ["spatial_tensor.h"],
    deps = [
        ":environment",
        ":gpu_object",
        ":instance",
        ":webgpu_api_util",
        ":webgpu_headers",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "spatial_tensor_test",
    timeout = "short",
    srcs = ["spatial_tensor_test.cc"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":environment",
        ":spatial_tensor",
        ":webgpu_headers",
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
    name = "webgpu_api_util",
    srcs = ["webgpu_api_util.cc"],
    hdrs = ["webgpu_api_util.h"],
    #    linkopts = select({
    #        "//tools/cc_target_os:emscripten": [
    #            "-s ASYNCIFY_IMPORTS=ReadBufferDataJs,WaitUntilCompletedJs",
    #        ],
    #        "//conditions:default": [],
    #    }),
    deps =
        [
            ":instance",
            ":webgpu_headers",
            "//ml_drift/common:data_type",
            "//ml_drift/common:executor",
            "//ml_drift/common:status",
            "//ml_drift/common:util",
            "@com_google_absl//absl/debugging:leak_check",
            "@com_google_absl//absl/log:absl_log",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/synchronization",
            "@com_google_absl//absl/time",
        ],
)

cc_library(
    name = "webgpu_headers",
    hdrs = ["webgpu_headers.h"],
    deps = ["@dawn//:libdawn"],
)
