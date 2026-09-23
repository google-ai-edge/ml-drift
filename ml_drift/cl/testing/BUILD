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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//ml_drift/common:build_config.bzl", "mld_gpu_tests_tags")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "cl_test",
    testonly = 1,
    srcs = ["cl_test.cc"],
    hdrs = ["cl_test.h"],
    deps = [
        "//ml_drift/cl:cl_operation",
        "//ml_drift/cl:environment",
        "//ml_drift/cl:opencl_wrapper",
        "//ml_drift/cl:tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "gpu_model_external_weights_test",
    srcs = ["gpu_model_external_weights_test.cc"],
    tags = mld_gpu_tests_tags(),
    deps = [
        ":cl_test",
        "//ml_drift/common:gpu_model_external_weights_test_util",
        "//ml_drift/common:shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gpu_model_linking_test",
    srcs = ["gpu_model_linking_test.cc"],
    tags = mld_gpu_tests_tags(),
    deps = [
        ":cl_test",
        "//ml_drift/common:gpu_model_linking_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gpu_weights_conversion_test",
    srcs = ["gpu_weights_conversion_test.cc"],
    tags = mld_gpu_tests_tags(),
    deps = [
        ":cl_test",
        "//ml_drift/cl:cl_command_queue",
        "//ml_drift/cl:environment",
        "//ml_drift/cl:inference_context",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:gpu_weights_conversion_test_util",
        "//ml_drift/common:model",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "operation_perf_test",
    srcs = ["operation_perf_test.cc"],
    linkstatic = True,
    tags = [
        "local",
    ],
    deps = [
        ":perf_util",
        "//ml_drift/cl:opencl_wrapper",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "perf_util",
    testonly = 1,
    srcs = ["perf_util.cc"],
    hdrs = ["perf_util.h"],
    deps = [
        "//ml_drift/cl:cl_operation",
        "//ml_drift/cl:environment",
        "//ml_drift/cl:tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/kernels:conv_constants",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:conv_wave_matrix_mali",
        "//ml_drift/common/kernels:conv_wave_memory",
        "//ml_drift/common/kernels:depthwise_conv",
        "//ml_drift/common/kernels:depthwise_conv_3x3",
        "//ml_drift/common/kernels:depthwise_conv_tiled",
        "//ml_drift/common/kernels:depthwise_conv_wave_memory",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:fully_connected_oi",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/kernels:softmax",
        "//ml_drift/common/kernels/special:conv_softmax_conv",
        "//ml_drift/common/selectors:convolution_selector",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)
