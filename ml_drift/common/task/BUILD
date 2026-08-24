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
    licenses = ["notice"],
)

cc_library(
    name = "arguments",
    srcs = ["arguments.cc"],
    hdrs = ["arguments.h"],
    deps = [
        ":buffer_desc",
        ":gpu_object_desc",
        ":serialization_base_cc_fbs",
        ":util",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:types",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "buffer_desc",
    srcs = ["buffer_desc.cc"],
    hdrs = ["buffer_desc.h"],
    deps = [
        ":gpu_object_desc",
        ":util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compiler_options",
    srcs = ["compiler_options.cc"],
    hdrs = ["compiler_options.h"],
)

cc_library(
    name = "gpu_object_desc",
    hdrs = ["gpu_object_desc.h"],
    deps = [
        ":serialization_base_cc_fbs",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "gpu_operation",
    srcs = ["gpu_operation.cc"],
    hdrs = ["gpu_operation.h"],
    deps = [
        ":arguments",
        ":buffer_desc",
        ":compiler_options",
        ":gpu_object_desc",
        ":gpu_tensor",
        ":serialization_base_cc_fbs",
        ":tensor_desc",
        ":tuning_type",
        ":util",
        ":wave_matrix_util",
        ":wave_memory_util",
        ":weights_layout",
        ":work_group_picking",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "gpu_operation_test",
    srcs = ["gpu_operation_test.cc"],
    linkstatic = 1,
    deps = [
        ":buffer_desc",
        ":gpu_object_desc",
        ":gpu_operation",
        ":tensor_desc",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/kernels:conv_generic",
        "@XNNPACK",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_tensor",
    hdrs = ["gpu_tensor.h"],
    deps = [
        ":tensor_desc",
    ],
)

cc_library(
    name = "profiling_info",
    srcs = ["profiling_info.cc"],
    hdrs = ["profiling_info.h"],
    deps = [
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "qcom_thin_filter_desc",
    srcs = ["qcom_thin_filter_desc.cc"],
    hdrs = ["qcom_thin_filter_desc.h"],
    deps = [
        ":gpu_object_desc",
        "//ml_drift/common:gpu_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "serialization_base",
    srcs = ["serialization_base.cc"],
    hdrs = ["serialization_base.h"],
    deps = [
        ":arguments",
        ":buffer_desc",
        ":gpu_object_desc",
        ":gpu_operation",
        ":serialization_base_cc_fbs",
        ":tensor_desc",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@flatbuffers//:runtime_cc",
    ],
)

flatbuffer_cc_library(
    name = "serialization_base_cc_fbs",
    srcs = ["serialization_base.fbs"],
    flatc_args = [
        "--scoped-enums",
    ],
)

cc_library(
    name = "tensor_desc",
    srcs = ["tensor_desc.cc"],
    hdrs = ["tensor_desc.h"],
    deps = [
        ":gpu_object_desc",
        ":serialization_base_cc_fbs",
        ":util",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "tensor_desc_test",
    srcs = ["tensor_desc_test.cc"],
    linkstatic = 1,
    deps = [
        ":tensor_desc",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "testing_ref_ops",
    testonly = True,
    srcs = ["testing_ref_ops.cc"],
    hdrs = ["testing_ref_ops.h"],
    deps = [
        ":gpu_operation",
        ":testing_util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "testing_util",
    testonly = 1,
    srcs = ["testing_util.cc"],
    hdrs = ["testing_util.h"],
    deps = [
        ":gpu_operation",
        ":profiling_info",
        ":tensor_desc",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tuning_type",
    hdrs = ["tuning_type.h"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":gpu_object_desc",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    linkstatic = 1,
    deps = [
        ":util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "wave_matrix_util",
    srcs = ["wave_matrix_util.cc"],
    hdrs = ["wave_matrix_util.h"],
    deps = [
        ":util",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "wave_memory_util",
    srcs = ["wave_memory_util.cc"],
    hdrs = ["wave_memory_util.h"],
    deps = [
        ":arguments",
        ":buffer_desc",
        ":compiler_options",
        ":gpu_object_desc",
        ":util",
        ":weights_conversion",
        ":weights_layout",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "weights_conversion",
    srcs = ["weights_conversion.cc"],
    hdrs = ["weights_conversion.h"],
    deps = [
        ":tensor_desc",
        ":weights_layout",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

exports_files(
    srcs = [
        "weights_conversion.cc",
        "weights_conversion.h",
        "weights_layout.cc",
        "weights_layout.h",
    ],
    visibility = [
        "@//ml_drift_delegate/delegate/serialization_weight_cache:__pkg__",
    ],
)

cc_test(
    name = "weights_conversion_test",
    srcs = ["weights_conversion_test.cc"],
    linkstatic = 1,
    deps = [
        ":weights_conversion",
        ":weights_layout",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/default:status_matchers",
        "@XNNPACK",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "weights_layout",
    srcs = ["weights_layout.cc"],
    hdrs = ["weights_layout.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "work_group_picking",
    srcs = ["work_group_picking.cc"],
    hdrs = ["work_group_picking.h"],
    deps = [
        ":tuning_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
    ],
)

cc_test(
    name = "work_group_picking_test",
    srcs = ["work_group_picking_test.cc"],
    deps = [
        ":tuning_type",
        ":work_group_picking",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:types",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
