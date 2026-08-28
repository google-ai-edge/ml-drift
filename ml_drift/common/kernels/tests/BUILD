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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test", "objc_library")
load("//ml_drift/common:build_config.bzl", "mld_gpu_tests_tags")
load("//ml_drift/common/kernels/tests:mld_kernel_test.bzl", "mld_kernel_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    [
        "kernel_test.h",
        "opencl_kernel_test.cc",
        "webgpu_kernel_test.cc",
    ],
    visibility = [
        "//ml_drift/common/kernels/big_tests:__subpackages__",
        "//ml_drift/common/kernels/special/tests:__subpackages__",
    ],
)

cc_library(
    name = "accumulate_input_channels_test_util",
    testonly = 1,
    srcs = ["accumulate_input_channels_test_util.cc"],
    hdrs = ["accumulate_input_channels_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:accumulate_input_channels",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "accumulate_input_channels_test",
    srcs = [
        "accumulate_input_channels_test.cc",
    ],
    deps = [
        ":accumulate_input_channels_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "add_test_util",
    testonly = 1,
    srcs = ["add_test_util.cc"],
    hdrs = ["add_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:add",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "add_test",
    srcs = [
        "add_test.cc",
    ],
    deps = [
        ":add_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bitcast_test_util",
    testonly = 1,
    srcs = ["bitcast_test_util.cc"],
    hdrs = ["bitcast_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:bitcast",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "bitcast_test",
    srcs = [
        "bitcast_test.cc",
    ],
    deps = [
        ":bitcast_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

mld_kernel_test(
    name = "bucket_argsort_test",
    srcs = ["bucket_argsort_test.cc"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:bucket_argsort",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "buffer_desc_test_util",
    testonly = 1,
    srcs = ["buffer_desc_test_util.cc"],
    hdrs = ["buffer_desc_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "buffer_desc_test",
    srcs = [
        "buffer_desc_test.cc",
    ],
    deps = [
        ":buffer_desc_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cast_test_util",
    testonly = 1,
    srcs = ["cast_test_util.cc"],
    hdrs = ["cast_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:cast",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "cast_test",
    srcs = [
        "cast_test.cc",
    ],
    deps = [
        ":cast_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "concat_test_util",
    testonly = 1,
    srcs = ["concat_test_util.cc"],
    hdrs = ["concat_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:concat_xy",
        "//ml_drift/common/kernels:concat_z",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "concat_test",
    srcs = [
        "concat_test.cc",
    ],
    deps = [
        ":concat_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_apple_mpp_test_util",
    testonly = True,
    srcs = ["conv_apple_mpp_test_util.cc"],
    hdrs = ["conv_apple_mpp_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:conv_apple_mpp",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_apple_mpp_test",
    srcs = ["conv_apple_mpp_test.cc"],
    deps = [
        ":conv_apple_mpp_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:conv_apple_mpp",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_constants_test_util",
    testonly = 1,
    srcs = ["conv_constants_test_util.cc"],
    hdrs = ["conv_constants_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:conv_constants",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_constants_test",
    srcs = [
        "conv_constants_test.cc",
    ],
    deps = [
        ":conv_constants_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_generic_test_util",
    testonly = 1,
    srcs = ["conv_generic_test_util.cc"],
    hdrs = ["conv_generic_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:conv_weights_converter",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_generic_test",
    timeout = "long",
    srcs = [
        "conv_generic_test.cc",
    ],
    deps = [
        ":conv_generic_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_transpose_test_util",
    testonly = 1,
    srcs = ["conv_transpose_test_util.cc"],
    hdrs = ["conv_transpose_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:convolution_transposed",
        "//ml_drift/common/kernels:convolution_transposed_2x2",
        "//ml_drift/common/kernels:convolution_transposed_3x3",
        "//ml_drift/common/kernels:convolution_transposed_4x4",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_transpose_test",
    srcs = [
        "conv_transpose_test.cc",
    ],
    deps = [
        ":conv_transpose_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:convolution_transposed",
        "//ml_drift/common/kernels:convolution_transposed_2x2",
        "//ml_drift/common/kernels:convolution_transposed_3x3",
        "//ml_drift/common/kernels:convolution_transposed_4x4",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_transpose_thin_test_util",
    testonly = 1,
    srcs = ["conv_transpose_thin_test_util.cc"],
    hdrs = ["conv_transpose_thin_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:convolution_transposed_3x3_thin",
        "//ml_drift/common/kernels:convolution_transposed_thin",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_transpose_thin_test",
    srcs = [
        "conv_transpose_thin_test.cc",
    ],
    platforms = ["opencl"],
    deps = [
        ":conv_transpose_thin_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:convolution_transposed_3x3_thin",
        "//ml_drift/common/kernels:convolution_transposed_thin",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_wave_matrix_mali_test_util",
    testonly = 1,
    srcs = ["conv_wave_matrix_mali_test_util.cc"],
    hdrs = ["conv_wave_matrix_mali_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:conv_wave_matrix_mali",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_wave_matrix_mali_test",
    srcs = [
        "conv_wave_matrix_mali_test.cc",
    ],
    platforms = ["opencl"],
    deps = [
        ":conv_wave_matrix_mali_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:conv_wave_matrix_mali",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_wave_matrix_test_util",
    testonly = 1,
    srcs = ["conv_wave_matrix_test_util.cc"],
    hdrs = ["conv_wave_matrix_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/kernels:conv_wave_matrix",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_wave_matrix_test",
    srcs = [
        "conv_wave_matrix_test.cc",
    ],
    platforms = [
        "opencl",
        "pelong",
        "vulkan",
        "webgpu",
        "mac",
    ],
    deps = [
        ":conv_wave_matrix_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_wave_memory_test_util",
    testonly = 1,
    srcs = ["conv_wave_memory_test_util.cc"],
    hdrs = ["conv_wave_memory_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/kernels:conv_wave_memory",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_wave_memory_test",
    srcs = [
        "conv_wave_memory_test.cc",
    ],
    platforms = ["opencl"],
    deps = [
        ":conv_wave_memory_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:conv_wave_memory",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_weights_converter_test_util",
    testonly = 1,
    srcs = ["conv_weights_converter_test_util.cc"],
    hdrs = ["conv_weights_converter_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:conv_weights_converter",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "conv_weights_converter_test",
    timeout = "long",
    srcs = [
        "conv_weights_converter_test.cc",
    ],
    deps = [
        ":conv_weights_converter_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cumsum_test_util",
    testonly = 1,
    srcs = ["cumsum_test_util.cc"],
    hdrs = ["cumsum_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:cumsum",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "cumsum_test",
    srcs = [
        "cumsum_test.cc",
    ],
    deps = [
        ":cumsum_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "depthwise_conv_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_test_util.cc"],
    hdrs = ["depthwise_conv_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:depthwise_conv",
        "//ml_drift/common/kernels:depthwise_conv_3x3",
        "//ml_drift/common/kernels:depthwise_conv_tiled",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "depthwise_conv_test",
    srcs = [
        "depthwise_conv_test.cc",
    ],
    deps = [
        ":depthwise_conv_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "depthwise_conv_wave_memory_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_wave_memory_test_util.cc"],
    hdrs = ["depthwise_conv_wave_memory_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:depthwise_conv_wave_memory",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "depthwise_conv_wave_memory_test",
    srcs = [
        "depthwise_conv_wave_memory_test.cc",
    ],
    platforms = ["opencl"],
    deps = [
        ":depthwise_conv_wave_memory_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:depthwise_conv_wave_memory",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dot_general_test_util",
    testonly = 1,
    srcs = ["dot_general_test_util.cc"],
    hdrs = ["dot_general_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:dot_general",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "dot_general_test",
    srcs = [
        "dot_general_test.cc",
    ],
    deps = [
        ":dot_general_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dynamic_update_slice_test_util",
    testonly = 1,
    srcs = ["dynamic_update_slice_test_util.cc"],
    hdrs = ["dynamic_update_slice_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:dynamic_update_slice",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "dynamic_update_slice_test",
    srcs = [
        "dynamic_update_slice_test.cc",
    ],
    deps = [
        ":dynamic_update_slice_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "elementwise_test_util",
    testonly = 1,
    srcs = ["elementwise_test_util.cc"],
    hdrs = ["elementwise_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "elementwise_test",
    srcs = [
        "elementwise_test.cc",
    ],
    deps = [
        ":elementwise_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "embedding_lookup_test_util",
    testonly = 1,
    srcs = ["embedding_lookup_test_util.cc"],
    hdrs = ["embedding_lookup_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:embedding_lookup",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "embedding_lookup_test",
    srcs = [
        "embedding_lookup_test.cc",
    ],
    deps = [
        ":embedding_lookup_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fully_connected_test_util",
    testonly = 1,
    srcs = ["fully_connected_test_util.cc"],
    hdrs = ["fully_connected_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "fully_connected_test",
    srcs = [
        "fully_connected_test.cc",
    ],
    deps = [
        ":fully_connected_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gather_test_util",
    testonly = 1,
    srcs = ["gather_test_util.cc"],
    hdrs = ["gather_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:gather",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "gather_test",
    srcs = [
        "gather_test.cc",
    ],
    deps = [
        ":gather_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

mld_kernel_test(
    name = "layer_norm_test",
    srcs = [
        "layer_norm_test.cc",
    ],
    platforms = ["opencl"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:model",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:tensor_matcher",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "lstm_test_util",
    testonly = 1,
    srcs = ["lstm_test_util.cc"],
    hdrs = ["lstm_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:lstm",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "lstm_test",
    srcs = [
        "lstm_test.cc",
    ],
    deps = [
        ":lstm_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "max_unpooling_test_util",
    testonly = 1,
    srcs = ["max_unpooling_test_util.cc"],
    hdrs = ["max_unpooling_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:max_unpooling",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "max_unpooling_test",
    srcs = [
        "max_unpooling_test.cc",
    ],
    deps = [
        ":max_unpooling_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mean_stddev_normalization_test_util",
    testonly = 1,
    srcs = ["mean_stddev_normalization_test_util.cc"],
    hdrs = ["mean_stddev_normalization_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:mean_stddev_normalization",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "mean_stddev_normalization_test",
    srcs = [
        "mean_stddev_normalization_test.cc",
    ],
    deps = [
        ":mean_stddev_normalization_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "one_hot_test_util",
    testonly = 1,
    srcs = ["one_hot_test_util.cc"],
    hdrs = ["one_hot_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:one_hot",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "one_hot_test",
    srcs = [
        "one_hot_test.cc",
    ],
    deps = [
        ":one_hot_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "padding_test_util",
    testonly = 1,
    srcs = ["padding_test_util.cc"],
    hdrs = ["padding_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:padding",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "padding_test",
    srcs = [
        "padding_test.cc",
    ],
    deps = [
        ":padding_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pooling_test_util",
    testonly = 1,
    srcs = ["pooling_test_util.cc"],
    hdrs = ["pooling_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:pooling",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "pooling_test",
    srcs = [
        "pooling_test.cc",
    ],
    deps = [
        ":pooling_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

mld_kernel_test(
    name = "positional_embedding_test",
    srcs = [
        "positional_embedding_test.cc",
    ],
    platforms = ["opencl"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:positional_embedding",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "prelu_test_util",
    testonly = 1,
    srcs = ["prelu_test_util.cc"],
    hdrs = ["prelu_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:prelu",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "prelu_test",
    srcs = [
        "prelu_test.cc",
    ],
    deps = [
        ":prelu_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "random_philox_test_util",
    testonly = 1,
    srcs = ["random_philox_test_util.cc"],
    hdrs = ["random_philox_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:random_philox",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "random_philox_test",
    srcs = [
        "random_philox_test.cc",
    ],
    platforms = [
        "opencl",
        "ios",
        "mac",
    ],
    deps = [
        ":random_philox_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduce_test_util",
    testonly = 1,
    srcs = ["reduce_test_util.cc"],
    hdrs = ["reduce_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "reduce_test",
    srcs = [
        "reduce_test.cc",
    ],
    deps = [
        ":reduce_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "relu_test_util",
    testonly = 1,
    srcs = ["relu_test_util.cc"],
    hdrs = ["relu_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:relu",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "relu_test",
    srcs = [
        "relu_test.cc",
    ],
    deps = [
        ":relu_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "resampler_test_util",
    testonly = 1,
    srcs = ["resampler_test_util.cc"],
    hdrs = ["resampler_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:resampler",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "resampler_test",
    srcs = [
        "resampler_test.cc",
    ],
    deps = [
        ":resampler_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reshape_test_util",
    testonly = 1,
    srcs = ["reshape_test_util.cc"],
    hdrs = ["reshape_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:reshape",
        "//ml_drift/common/kernels:reshapex4",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "reshape_test",
    srcs = [
        "reshape_test.cc",
    ],
    deps = [
        ":reshape_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "resize_test_util",
    testonly = True,
    srcs = ["resize_test_util.cc"],
    hdrs = ["resize_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:resize",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "resize_test",
    srcs = [
        "resize_test.cc",
    ],
    deps = [
        ":resize_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reverse_test_util",
    testonly = 1,
    srcs = ["reverse_test_util.cc"],
    hdrs = ["reverse_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:reverse",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "reverse_test",
    srcs = [
        "reverse_test.cc",
    ],
    deps = [
        ":reverse_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "rope_test_util",
    testonly = 1,
    srcs = ["rope_test_util.cc"],
    hdrs = ["rope_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:rope",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "rope_test",
    srcs = [
        "rope_test.cc",
    ],
    deps = [
        ":rope_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "select_v2_test_util",
    testonly = 1,
    srcs = ["select_v2_test_util.cc"],
    hdrs = ["select_v2_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:select_v2",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "select_v2_test",
    srcs = [
        "select_v2_test.cc",
    ],
    deps = [
        ":select_v2_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "softmax_test_util",
    testonly = 1,
    srcs = ["softmax_test_util.cc"],
    hdrs = ["softmax_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:softmax",
        "//ml_drift/common/kernels:softmax1x1",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "softmax_test",
    timeout = "long",
    srcs = [
        "softmax_test.cc",
    ],
    deps = [
        ":softmax_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "space_to_depth_test_util",
    testonly = 1,
    srcs = ["space_to_depth_test_util.cc"],
    hdrs = ["space_to_depth_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:space_to_depth",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "space_to_depth_test",
    srcs = [
        "space_to_depth_test.cc",
    ],
    deps = [
        ":space_to_depth_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "split_test_util",
    testonly = 1,
    srcs = ["split_test_util.cc"],
    hdrs = ["split_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:split",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "split_test",
    srcs = [
        "split_test.cc",
    ],
    deps = [
        ":split_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
    ],
)

cc_library(
    name = "strided_slice_test_util",
    testonly = 1,
    srcs = ["strided_slice_test_util.cc"],
    hdrs = ["strided_slice_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:strided_slice",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "strided_slice_test",
    srcs = [
        "strided_slice_test.cc",
    ],
    deps = [
        ":strided_slice_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tile_test_util",
    testonly = 1,
    srcs = ["tile_test_util.cc"],
    hdrs = ["tile_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:tile",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "tile_test",
    srcs = [
        "tile_test.cc",
    ],
    deps = [
        ":tile_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "top_k_test_util",
    testonly = 1,
    srcs = ["top_k_test_util.cc"],
    hdrs = ["top_k_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:top_k",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "top_k_test",
    srcs = [
        "top_k_test.cc",
    ],
    deps = [
        ":top_k_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "transpose_test_util",
    testonly = 1,
    srcs = ["transpose_test_util.cc"],
    hdrs = ["transpose_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:transpose",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "transpose_test",
    srcs = [
        "transpose_test.cc",
    ],
    deps = [
        ":transpose_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "winograd_test_util",
    testonly = 1,
    srcs = ["winograd_test_util.cc"],
    hdrs = ["winograd_test_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

mld_kernel_test(
    name = "winograd_test",
    srcs = [
        "winograd_test.cc",
    ],
    deps = [
        ":winograd_test_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/strings",
    ],
)

test_suite(
    name = "all_tests_opencl",
    tests = [
        ":add_test_opencl",
        ":bitcast_test_opencl",
        ":cast_test_opencl",
        ":concat_test_opencl",
        ":conv_constants_test_opencl",
        ":conv_generic_test_opencl",
        ":conv_transpose_test_opencl",
        ":conv_transpose_thin_test_opencl",
        ":conv_wave_matrix_mali_test_opencl",
        ":conv_wave_matrix_test_opencl",
        ":depthwise_conv_test_opencl",
        ":depthwise_conv_wave_memory_test_opencl",
        ":dot_general_test_opencl",
        ":dynamic_update_slice_test_opencl",
        ":elementwise_test_opencl",
        ":embedding_lookup_test_opencl",
        ":fully_connected_test_opencl",
        ":gather_test_opencl",
        ":layer_norm_test_opencl",
        ":lstm_test_opencl",
        ":max_unpooling_test_opencl",
        ":mean_stddev_normalization_test_opencl",
        ":one_hot_test_opencl",
        ":padding_test_opencl",
        ":pooling_test_opencl",
        ":positional_embedding_test_opencl",
        ":prelu_test_opencl",
        ":random_philox_test_opencl",
        ":reduce_test_opencl",
        ":relu_test_opencl",
        ":resampler_test_opencl",
        ":reshape_test_opencl",
        ":resize_test_opencl",
        ":rope_test_opencl",
        ":select_v2_test_opencl",
        ":softmax_test_opencl",
        ":space_to_depth_test_opencl",
        ":split_test_opencl",
        ":strided_slice_test_opencl",
        ":tile_test_opencl",
        ":top_k_test_opencl",
        ":transpose_test_opencl",
        ":winograd_test_opencl",
    ],
)

test_suite(
    name = "all_tests_webgpu",
    tests = [
        ":add_test_webgpu",
        ":bitcast_test_webgpu",
        ":cast_test_webgpu",
        ":concat_test_webgpu",
        ":conv_constants_test_webgpu",
        ":conv_generic_test_webgpu",
        ":conv_transpose_test_webgpu",
        ":conv_wave_matrix_test_webgpu",
        ":conv_weights_converter_test_webgpu",
        ":cumsum_test_webgpu",
        ":depthwise_conv_test_webgpu",
        ":dot_general_test_webgpu",
        ":dynamic_update_slice_test_webgpu",
        ":elementwise_test_webgpu",
        ":embedding_lookup_test_webgpu",
        ":fully_connected_test_webgpu",
        ":gather_test_webgpu",
        ":lstm_test_webgpu",
        ":max_unpooling_test_webgpu",
        ":mean_stddev_normalization_test_webgpu",
        ":one_hot_test_webgpu",
        ":padding_test_webgpu",
        ":pooling_test_webgpu",
        ":prelu_test_webgpu",
        ":reduce_test_webgpu",
        ":relu_test_webgpu",
        ":resampler_test_webgpu",
        ":reshape_test_webgpu",
        ":resize_test_webgpu",
        ":rope_test_webgpu",
        ":select_v2_test_webgpu",
        ":softmax_test_webgpu",
        ":space_to_depth_test_webgpu",
        ":split_test_webgpu",
        ":strided_slice_test_webgpu",
        ":tile_test_webgpu",
        ":top_k_test_webgpu",
        ":transpose_test_webgpu",
        ":winograd_test_webgpu",
    ],
)
