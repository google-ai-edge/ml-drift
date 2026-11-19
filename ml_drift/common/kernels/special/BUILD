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
    name = "concat_conv",
    srcs = ["concat_conv.cc"],
    hdrs = ["concat_conv.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
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
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv2x2_max_pool2x2",
    srcs = ["conv2x2_max_pool2x2.cc"],
    hdrs = ["conv2x2_max_pool2x2.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_pointwise",
    srcs = ["conv_pointwise.cc"],
    hdrs = ["conv_pointwise.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_softmax_conv",
    srcs = ["conv_softmax_conv.cc"],
    hdrs = ["conv_softmax_conv.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "//ml_drift/common/task:wave_memory_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8",
    srcs = ["dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8.cc"],
    hdrs = ["dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dw7x7_conv2to6_concat_conv8to8",
    srcs = ["dw7x7_conv2to6_concat_conv8to8.cc"],
    hdrs = ["dw7x7_conv2to6_concat_conv8to8.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:qcom_thin_filter_desc",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dw3x3_conv16to16_conv16to32_add_conv32to16",
    srcs = ["dw3x3_conv16to16_conv16to32_add_conv32to16.cc"],
    hdrs = ["dw3x3_conv16to16_conv16to32_add_conv32to16.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32",
    srcs = ["dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32.cc"],
    hdrs = ["dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "experts_remap",
    srcs = ["experts_remap.cc"],
    hdrs = ["experts_remap.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:types",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "thin_local_memory_fuser",
    srcs = ["thin_local_memory_fuser.cc"],
    hdrs = ["thin_local_memory_fuser.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:resize",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "thin_pointwise_fuser",
    srcs = ["thin_pointwise_fuser.cc"],
    hdrs = ["thin_pointwise_fuser.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:prelu",
        "//ml_drift/common/kernels:relu",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:compiler_options",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)
