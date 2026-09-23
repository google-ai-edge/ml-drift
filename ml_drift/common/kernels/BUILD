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

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "accumulate_input_channels",
    srcs = ["accumulate_input_channels.cc"],
    hdrs = ["accumulate_input_channels.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bitcast",
    srcs = ["bitcast.cc"],
    hdrs = ["bitcast.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bucket_argsort",
    srcs = ["bucket_argsort.cc"],
    hdrs = ["bucket_argsort.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:types",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cast",
    srcs = ["cast.cc"],
    hdrs = ["cast.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "concat_xy",
    srcs = ["concat_xy.cc"],
    hdrs = ["concat_xy.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "concat_z",
    srcs = ["concat_z.cc"],
    hdrs = ["concat_z.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "conv_apple_mpp",
    srcs = ["conv_apple_mpp.cc"],
    hdrs = ["conv_apple_mpp.h"],
    deps = [
        ":fully_connected_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "conv_constants",
    srcs = ["conv_constants.cc"],
    hdrs = ["conv_constants.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "conv_generic",
    srcs = ["conv_generic.cc"],
    hdrs = ["conv_generic.h"],
    deps = [
        ":fully_connected_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "conv_wave_matrix",
    srcs = ["conv_wave_matrix.cc"],
    hdrs = ["conv_wave_matrix.h"],
    deps = [
        ":fully_connected_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "conv_wave_matrix_mali",
    srcs = ["conv_wave_matrix_mali.cc"],
    hdrs = ["conv_wave_matrix_mali.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "conv_wave_memory",
    srcs = ["conv_wave_memory.cc"],
    hdrs = ["conv_wave_memory.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:util",
        "//ml_drift/common/task:wave_memory_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "conv_weights_converter",
    srcs = ["conv_weights_converter.cc"],
    hdrs = ["conv_weights_converter.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed",
    srcs = ["convolution_transposed.cc"],
    hdrs = ["convolution_transposed.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_transposed_2x2",
    srcs = ["convolution_transposed_2x2.cc"],
    hdrs = ["convolution_transposed_2x2.h"],
    deps = [
        ":convolution_transposed_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_transposed_3x3",
    srcs = ["convolution_transposed_3x3.cc"],
    hdrs = ["convolution_transposed_3x3.h"],
    deps = [
        ":convolution_transposed_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_thin",
    srcs = ["convolution_transposed_3x3_thin.cc"],
    hdrs = ["convolution_transposed_3x3_thin.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_transposed_4x4",
    srcs = ["convolution_transposed_4x4.cc"],
    hdrs = ["convolution_transposed_4x4.h"],
    deps = [
        ":convolution_transposed_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_transposed_thin",
    srcs = ["convolution_transposed_thin.cc"],
    hdrs = ["convolution_transposed_thin.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "convolution_transposed_util",
    srcs = ["convolution_transposed_util.cc"],
    hdrs = ["convolution_transposed_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:wave_memory_util",
    ],
)

cc_library(
    name = "cumsum",
    srcs = ["cumsum.cc"],
    hdrs = ["cumsum.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "depthwise_conv_3x3",
    srcs = ["depthwise_conv_3x3.cc"],
    hdrs = ["depthwise_conv_3x3.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "depthwise_conv_tiled",
    srcs = ["depthwise_conv_tiled.cc"],
    hdrs = ["depthwise_conv_tiled.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "depthwise_conv_wave_memory",
    srcs = ["depthwise_conv_wave_memory.cc"],
    hdrs = ["depthwise_conv_wave_memory.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:wave_memory_util",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        ":fully_connected_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

exports_files(
    srcs = [
        "fully_connected.cc",
        "fully_connected.h",
    ],
    visibility = [
        "@//ml_drift_delegate/delegate/serialization_weight_cache:__pkg__",
    ],
)

cc_library(
    name = "fully_connected_oi",
    srcs = ["fully_connected_oi.cc"],
    hdrs = ["fully_connected_oi.h"],
    deps = [
        ":fully_connected_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fully_connected_util",
    srcs = ["fully_connected_util.cc"],
    hdrs = ["fully_connected_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gather",
    srcs = ["gather.cc"],
    hdrs = ["gather.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "lstm",
    srcs = ["lstm.cc"],
    hdrs = ["lstm.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:precision",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "mean_stddev_normalization",
    srcs = ["mean_stddev_normalization.cc"],
    hdrs = ["mean_stddev_normalization.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mish",
    srcs = ["mish.cc"],
    hdrs = ["mish.h"],
    deps = [
        ":elementwise",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "one_hot",
    srcs = ["one_hot.cc"],
    hdrs = ["one_hot.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "positional_embedding",
    srcs = ["positional_embedding.cc"],
    hdrs = ["positional_embedding.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "quantize_and_dequantize",
    srcs = ["quantize_and_dequantize.cc"],
    hdrs = ["quantize_and_dequantize.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "random_philox",
    srcs = ["random_philox.cc"],
    hdrs = ["random_philox.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "reduce",
    srcs = ["reduce.cc"],
    hdrs = ["reduce.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:util",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduce_parser",
    srcs = ["reduce_parser.cc"],
    hdrs = ["reduce_parser.h"],
    deps = [
        ":reduce",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "resampler",
    srcs = ["resampler.cc"],
    hdrs = ["resampler.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "reshapex4",
    srcs = ["reshapex4.cc"],
    hdrs = ["reshapex4.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:work_group_picking",
    ],
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reverse",
    srcs = ["reverse.cc"],
    hdrs = ["reverse.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "rope",
    srcs = ["rope.cc"],
    hdrs = ["rope.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "scatter_nd",
    srcs = ["scatter_nd.cc"],
    hdrs = ["scatter_nd.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "select_v2",
    srcs = ["select_v2.cc"],
    hdrs = ["select_v2.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "softmax1x1",
    srcs = ["softmax1x1.cc"],
    hdrs = ["softmax1x1.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "space_to_depth",
    srcs = ["space_to_depth.cc"],
    hdrs = ["space_to_depth.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "split",
    srcs = ["split.cc"],
    hdrs = ["split.h"],
    deps = [
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "strided_slice",
    srcs = ["strided_slice.cc"],
    hdrs = ["strided_slice.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "tile",
    srcs = ["tile.cc"],
    hdrs = ["tile.h"],
    deps = [
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "top_k",
    srcs = ["top_k.cc"],
    hdrs = ["top_k.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "transpose",
    srcs = ["transpose.cc"],
    hdrs = ["transpose.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "winograd",
    srcs = ["winograd.cc"],
    hdrs = ["winograd.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dot_general",
    srcs = ["dot_general.cc"],
    hdrs = ["dot_general.h"],
    deps = [
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "dynamic_update_slice",
    srcs = ["dynamic_update_slice.cc"],
    hdrs = ["dynamic_update_slice.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "embedding_lookup",
    srcs = ["embedding_lookup.cc"],
    hdrs = ["embedding_lookup.h"],
    deps = [
        ":fully_connected",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "mean_stddev_normalization_parser",
    srcs = ["mean_stddev_normalization_parser.cc"],
    hdrs = ["mean_stddev_normalization_parser.h"],
    deps = [
        ":mean_stddev_normalization",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)
