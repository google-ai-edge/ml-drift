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
load("//ml_drift/common:build_config.bzl", "mld_platform_alias")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "add_quant_adjustments",
    srcs = ["add_quant_adjustments.cc"],
    hdrs = ["add_quant_adjustments.h"],
    deps = [
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "add_quant_adjustments_test",
    srcs = ["add_quant_adjustments_test.cc"],
    linkstatic = 1,
    deps = [
        ":add_quant_adjustments",
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fuse_add_to_conv",
    srcs = ["fuse_add_to_conv.cc"],
    hdrs = ["fuse_add_to_conv.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "fuse_add_to_conv_test",
    srcs = ["fuse_add_to_conv_test.cc"],
    linkstatic = 1,
    deps = [
        ":fuse_add_to_conv",
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fuse_mul_to_conv",
    srcs = ["fuse_mul_to_conv.cc"],
    hdrs = ["fuse_mul_to_conv.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "fuse_mul_to_conv_test",
    srcs = ["fuse_mul_to_conv_test.cc"],
    linkstatic = 1,
    deps = [
        ":fuse_mul_to_conv",
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "global_pooling_to_reduce_op",
    srcs = ["global_pooling_to_reduce_op.cc"],
    hdrs = ["global_pooling_to_reduce_op.h"],
    deps = [
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "global_pooling_to_reduce_op_test",
    srcs = ["global_pooling_to_reduce_op_test.cc"],
    linkstatic = 1,
    deps = [
        ":global_pooling_to_reduce_op",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "make_fully_connected",
    srcs = ["make_fully_connected.cc"],
    hdrs = ["make_fully_connected.h"],
    deps = [
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
    ],
)

cc_test(
    name = "make_fully_connected_test",
    srcs = ["make_fully_connected_test.cc"],
    linkstatic = 1,
    deps = [
        ":make_fully_connected",
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "make_padding",
    srcs = ["make_padding.cc"],
    hdrs = ["make_padding.h"],
    deps = [
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "make_padding_test",
    srcs = ["make_padding_test.cc"],
    linkstatic = 1,
    deps = [
        ":make_padding",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "matching",
    hdrs = ["matching.h"],
)

cc_library(
    name = "merge_padding_with",
    srcs = ["merge_padding_with.cc"],
    hdrs = ["merge_padding_with.h"],
    deps = [
        ":matching",
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "merge_padding_with_test",
    srcs = ["merge_padding_with_test.cc"],
    linkstatic = 1,
    deps = [
        ":merge_padding_with",
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_transformations",
    srcs = ["model_transformations.cc"],
    hdrs = ["model_transformations.h"],
    deps = [
        ":add_quant_adjustments",
        ":fuse_add_to_conv",
        ":fuse_mul_to_conv",
        ":global_pooling_to_reduce_op",
        ":make_fully_connected",
        ":make_padding",
        ":merge_padding_with",
        ":remove_noop",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "@com_google_absl//absl/status",
    ] + mld_platform_alias("custom_transformations", "//ml_drift/common/"),
)

cc_library(
    name = "remove_noop",
    srcs = ["remove_noop.cc"],
    hdrs = ["remove_noop.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "remove_noop_test",
    srcs = ["remove_noop_test.cc"],
    linkstatic = 1,
    deps = [
        ":remove_noop",
        "//ml_drift/common:data_type",
        "//ml_drift/common:model",
        "//ml_drift/common:model_transformer",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
