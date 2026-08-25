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

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//ml_drift/common:build_config.bzl", "mld_platform_alias")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "convolution_selector",
    srcs = ["convolution_selector.cc"],
    hdrs = ["convolution_selector.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:conv_apple_mpp",
        "//ml_drift/common/kernels:conv_constants",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:conv_wave_matrix",
        "//ml_drift/common/kernels:conv_wave_matrix_mali",
        "//ml_drift/common/kernels:conv_wave_memory",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    srcs = ["convolution_transposed_selector.cc"],
    hdrs = ["convolution_transposed_selector.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common/kernels:convolution_transposed",
        "//ml_drift/common/kernels:convolution_transposed_2x2",
        "//ml_drift/common/kernels:convolution_transposed_3x3",
        "//ml_drift/common/kernels:convolution_transposed_3x3_thin",
        "//ml_drift/common/kernels:convolution_transposed_4x4",
        "//ml_drift/common/kernels:convolution_transposed_thin",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "default_selector",
    hdrs = ["default_selector.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:model",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status:statusor",
    ] + mld_platform_alias("default_selector", "//ml_drift/common/selectors/"),
)

cc_library(
    name = "dw_convolution_selector",
    srcs = ["dw_convolution_selector.cc"],
    hdrs = ["dw_convolution_selector.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common/kernels:depthwise_conv",
        "//ml_drift/common/kernels:depthwise_conv_3x3",
        "//ml_drift/common/kernels:depthwise_conv_tiled",
        "//ml_drift/common/kernels:depthwise_conv_wave_memory",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "operation_selector",
    srcs = ["operation_selector.cc"],
    hdrs = ["operation_selector.h"],
    deps = [
        ":default_selector",
        ":simple_selectors",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:mean_stddev_normalization",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "simple_selectors",
    srcs = ["simple_selectors.cc"],
    hdrs = ["simple_selectors.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:model_hints",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:add",
        "//ml_drift/common/kernels:concat_xy",
        "//ml_drift/common/kernels:concat_z",
        "//ml_drift/common/kernels:conv_weights_converter",
        "//ml_drift/common/kernels:cumsum",
        "//ml_drift/common/kernels:dynamic_update_slice",
        "//ml_drift/common/kernels:embedding_lookup",
        "//ml_drift/common/kernels:lstm",
        "//ml_drift/common/kernels:max_unpooling",
        "//ml_drift/common/kernels:one_hot",
        "//ml_drift/common/kernels:padding",
        "//ml_drift/common/kernels:pooling",
        "//ml_drift/common/kernels:prelu",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/kernels:relu",
        "//ml_drift/common/kernels:resampler",
        "//ml_drift/common/kernels:resize",
        "//ml_drift/common/kernels:reverse",
        "//ml_drift/common/kernels:scatter_nd",
        "//ml_drift/common/kernels:select_v2",
        "//ml_drift/common/kernels:space_to_depth",
        "//ml_drift/common/kernels:split",
        "//ml_drift/common/kernels:tile",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "special_selector",
    srcs = ["special_selector.cc"],
    hdrs = ["special_selector.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:model_hints",
        "//ml_drift/common/kernels:mean_stddev_normalization_parser",
        "//ml_drift/common/kernels:mish",
        "//ml_drift/common/kernels:reduce_parser",
        "//ml_drift/common/kernels/special:concat_conv",
        "//ml_drift/common/kernels/special:conv_pointwise",
        "//ml_drift/common/kernels/special:dw7x7_conv2to6_concat_conv8to8",
        "//ml_drift/common/kernels/special:thin_local_memory_fuser",
        "//ml_drift/common/kernels/special:thin_pointwise_fuser",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
    ],
)
