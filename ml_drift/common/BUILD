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
load("//ml_drift/common:build_config.bzl", "mld_platform_alias")

cc_library(
    name = "demangle",
    hdrs = ["demangle.h"],
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "convert",
    srcs = ["convert.cc"],
    hdrs = ["convert.h"],
    deps = [
        ":data_type",
        ":shape",
        ":status",
        ":tensor",
        ":types",
        ":util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

exports_files(["custom_transformations.h"])

cc_library(
    name = "access_type",
    hdrs = ["access_type.h"],
)

cc_library(
    name = "executor",
    hdrs = ["executor.h"],
)

cc_library(
    name = "gpu_info",
    srcs = ["gpu_info.cc"],
    hdrs = ["gpu_info.h"],
    deps = [
        ":data_type",
        ":types",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_model",
    srcs = ["gpu_model.cc"],
    hdrs = ["gpu_model.h"],
    deps = [
        ":gpu_model_cc_fbs",
        ":model",
        ":model_hints",
        ":precision",
        ":status",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:serialization_base",
        "//ml_drift/common/task:serialization_base_cc_fbs",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "gpu_model_builder",
    srcs = ["gpu_model_builder.cc"],
    hdrs = ["gpu_model_builder.h"],
    deps = [
        ":data_type",
        ":flops_util",
        ":gpu_info",
        ":gpu_model",
        ":merge_nodes",
        ":model",
        ":model_hints",
        ":op_attrs",
        ":op_base",
        ":op_registry",
        ":operations",
        ":precision",
        ":shape",
        ":status",
        ":tensor",
        ":tensor_handle",
        ":types",
        ":util",
        ":winograd_util",
        "//ml_drift/common/kernels:accumulate_input_channels",
        "//ml_drift/common/kernels:bitcast",
        "//ml_drift/common/kernels:cast",
        "//ml_drift/common/kernels:conv_apple_mpp",
        "//ml_drift/common/kernels:conv_wave_matrix",
        "//ml_drift/common/kernels:conv_weights_converter",
        "//ml_drift/common/kernels:cumsum",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:embedding_lookup",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:gather",
        "//ml_drift/common/kernels:mean_stddev_normalization",
        "//ml_drift/common/kernels:positional_embedding",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/kernels:reshape",
        "//ml_drift/common/kernels:reshapex4",
        "//ml_drift/common/kernels:rope",
        "//ml_drift/common/kernels:select_v2",
        "//ml_drift/common/kernels:softmax",
        "//ml_drift/common/kernels:softmax1x1",
        "//ml_drift/common/kernels:space_to_depth",
        "//ml_drift/common/kernels:strided_slice",
        "//ml_drift/common/kernels:top_k",
        "//ml_drift/common/kernels:transpose",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/kernels/special:conv_softmax_conv",
        "//ml_drift/common/selectors:convolution_selector",
        "//ml_drift/common/selectors:convolution_transposed_selector",
        "//ml_drift/common/selectors:dw_convolution_selector",
        "//ml_drift/common/selectors:simple_selectors",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

flatbuffer_cc_library(
    name = "gpu_model_cc_fbs",
    srcs = ["gpu_model.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I .",
    ],
    include_paths = [
        "external/ml_drift",
        "ml_drift/common/task",
    ],
    includes = [
        "//ml_drift/common/task:serialization_base_cc_fbs_includes",
    ],
)

cc_library(
    name = "gpu_model_external_weights_test_util",
    testonly = True,
    srcs = ["gpu_model_external_weights_test_util.cc"],
    hdrs = ["gpu_model_external_weights_test_util.h"],
    deps = [
        ":data_type",
        ":gpu_model",
        ":gpu_model_builder",
        ":gpu_model_util",
        ":model",
        ":operations",
        ":precision",
        ":shape",
        ":status",
        ":tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_model_linking_test_util",
    testonly = True,
    srcs = ["gpu_model_linking_test_util.cc"],
    hdrs = ["gpu_model_linking_test_util.h"],
    deps = [
        ":data_type",
        ":gpu_model",
        ":gpu_model_util",
        ":model",
        ":model_hints",
        ":operations",
        ":precision",
        ":shape",
        ":status",
        ":tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/kernels:cast",
        "//ml_drift/common/kernels:concat_z",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:padding",
        "//ml_drift/common/kernels:prelu",
        "//ml_drift/common/kernels:reshape",
        "//ml_drift/common/kernels:reshapex4",
        "//ml_drift/common/kernels:strided_slice",
        "//ml_drift/common/kernels:transpose",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/transformations:model_transformations",
        "@XNNPACK",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_model_util",
    srcs = ["gpu_model_util.cc"],
    hdrs = ["gpu_model_util.h"],
    deps = [
        ":data_type",
        ":gpu_info",
        ":gpu_model",
        ":gpu_model_builder",
        ":model",
        ":operations",
        ":precision",
        ":shape",
        ":status",
        ":tensor",
        "//ml_drift/common/selectors:operation_selector",
        "//ml_drift/common/selectors:special_selector",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_weights_conversion_test_util",
    srcs = ["gpu_weights_conversion_test_util.cc"],
    hdrs = ["gpu_weights_conversion_test_util.h"],
    deps = [
        ":data_type",
        ":model",
        ":operations",
        ":shape",
        ":status",
        ":tensor",
        "@XNNPACK",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "kernel_info",
    hdrs = ["kernel_info.h"],
)

cc_library(
    name = "data_type",
    srcs = ["data_type.cc"],
    hdrs = ["data_type.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "data_type_test",
    srcs = ["data_type_test.cc"],
    linkstatic = 1,
    deps = [
        ":data_type",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flops_util",
    srcs = ["flops_util.cc"],
    hdrs = ["flops_util.h"],
    deps = [":shape"],
)

cc_library(
    name = "ir_model",
    srcs = ["ir_model.cc"],
    hdrs = ["ir_model.h"],
    deps = [
        ":data_type",
        ":shape",
        ":status",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "ir_model_test",
    srcs = ["ir_model_test.cc"],
    deps = [
        ":data_type",
        ":ir_model",
        ":shape",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ir_model_util_test",
    srcs = ["ir_model_util_test.cc"],
    deps = [
        ":data_type",
        ":gpu_info",
        ":gpu_model",
        ":ir_model",
        ":ir_model_util",
        ":model_hints",
        ":operations",
        ":precision",
        ":shape",
        ":tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ir_model_util",
    srcs = ["ir_model_util.cc"],
    hdrs = ["ir_model_util.h"],
    deps = [
        ":data_type",
        ":gpu_info",
        ":gpu_model",
        ":gpu_model_builder",
        ":ir_model",
        ":model",
        ":operations",
        ":precision",
        ":shape",
        ":status",
        ":tensor",
        "//ml_drift/common/selectors:operation_selector",
        "//ml_drift/common/selectors:special_selector",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "memory_management",
    srcs = [
        "memory_management.cc",
        "memory_management/greedy_by_breadth_assignment.cc",
        "memory_management/greedy_by_size_assignment.cc",
        "memory_management/internal.cc",
        "memory_management/min_cost_flow_assignment.cc",
        "memory_management/types.cc",
    ],
    hdrs = [
        "memory_management.h",
        "memory_management/equality_assignment.h",
        "memory_management/greedy_by_breadth_assignment.h",
        "memory_management/greedy_by_size_assignment.h",
        "memory_management/greedy_in_order_assignment.h",
        "memory_management/internal.h",
        "memory_management/min_cost_flow_assignment.h",
        "memory_management/naive_assignment.h",
        "memory_management/types.h",
    ],
    deps = [
        ":shape",
        ":status",
        ":types",
        ":util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "merge_nodes",
    srcs = ["merge_nodes.cc"],
    hdrs = ["merge_nodes.h"],
    deps = [
        ":gpu_info",
        ":gpu_model",
        ":model",
        ":shape",
        ":status",
        ":util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "merge_nodes_test",
    srcs = ["merge_nodes_test.cc"],
    linkstatic = 1,
    deps = [
        ":data_type",
        ":gpu_info",
        ":gpu_model",
        ":gpu_model_builder",
        ":model",
        ":operations",
        ":shape",
        ":status",
        ":tensor",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/kernels:reduce",
        "//ml_drift/common/task:tensor_desc",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = ["model.h"],
    deps = [
        ":shape",
        ":status",
        ":tensor",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    linkstatic = 1,
    deps = [
        ":model",
        ":shape",
        ":status",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_hints",
    hdrs = ["model_hints.h"],
)

cc_library(
    name = "model_transformer",
    srcs = ["model_transformer.cc"],
    hdrs = ["model_transformer.h"],
    deps = [
        ":model",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

# TODO(impjdi): Add unit test for model_transformer.

cc_library(
    name = "tensor_handle",
    hdrs = ["tensor_handle.h"],
    deps = [
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_base",
    hdrs = ["op_base.h"],
    deps = [
        ":demangle",
        ":op_attrs",
        ":status",
        ":tensor_handle",
        ":types",
        "//ml_drift/common/task:gpu_operation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_api",
    hdrs = ["op_api.h"],
    deps = [
        ":data_type",
        ":gpu_model_builder",
        ":op_attrs",
        ":op_base",
        ":op_registry",
        ":types",
        "//ml_drift/common/task:gpu_operation",
    ],
)

cc_library(
    name = "op_registry",
    srcs = ["op_registry.cc"],
    hdrs = ["op_registry.h"],
    deps = [
        ":op_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "operations",
    srcs = ["operations.cc"],
    hdrs = ["operations.h"],
    deps = [
        ":data_type",
        ":shape",
        ":status",
        ":tensor",
        ":util",
        "//ml_drift/common/task:weights_layout",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_test(
    name = "operations_test",
    srcs = ["operations_test.cc"],
    deps = [
        ":data_type",
        ":operations",
        ":shape",
        ":tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "precision",
    srcs = ["precision.cc"],
    hdrs = ["precision.h"],
    deps = [":data_type"],
)

# TODO(impjdi): Add unit test for operations.

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    linkstatic = 1,
    deps = [
        ":shape",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + mld_platform_alias("status_macros", "//ml_drift/common/"),
)

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
    deps = [
        ":data_type",
        ":shape",
        ":types",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":data_type",
        ":shape",
        ":tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensor_matcher",
    testonly = True,
    hdrs = ["tensor_matcher.h"],
    deps = [
        ":tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = ["@fp16"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":data_type",
        ":gpu_info",
        ":precision",
        ":shape",
        ":types",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "memory_management_test",
    srcs = ["memory_management_test.cc"],
    linkstatic = 1,
    deps = [
        ":memory_management",
        ":shape",
        ":status",
        ":types",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_management_internal_test",
    srcs = ["memory_management/internal_test.cc"],
    linkstatic = 1,
    deps = [
        ":memory_management",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "memory_management_types_test",
    srcs = ["memory_management/types_test.cc"],
    linkstatic = 1,
    deps = [
        ":memory_management",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gpu_info_test",
    srcs = ["gpu_info_test.cc"],
    deps = [
        ":gpu_info",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    linkstatic = 1,
    deps = [
        ":gpu_info",
        ":precision",
        ":util",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "winograd_util",
    srcs = ["winograd_util.cc"],
    hdrs = ["winograd_util.h"],
    deps = [
        ":data_type",
        ":gpu_info",
        ":operations",
        ":shape",
        ":tensor",
        ":util",
        "@XNNPACK",
    ],
)

cc_test(
    name = "winograd_util_test",
    srcs = ["winograd_util_test.cc"],
    linkstatic = 1,
    deps = [
        ":data_type",
        ":shape",
        ":tensor",
        ":winograd_util",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "greedy_in_order_assignment_test",
    srcs = ["memory_management/greedy_in_order_assignment_test.cc"],
    deps = [
        ":memory_management",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

config_setting(
    name = "opencl_delegate_no_gl",
    values = {"copt": "-DCL_DELEGATE_NO_GL"},
)

cc_library(
    name = "api_common",
    srcs = ["api_common.cc"],
    hdrs = ["api_common.h"],
    deps = [
        ":util",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "api_cl_gl_vk",
    srcs = ["api_cl_gl_vk.cc"],
    hdrs = ["api_cl_gl_vk.h"],
    defines = ["CL_DELEGATE_NO_GL"],
    deps = [
        ":api_common",
        ":data_type",
        ":status",
        ":util",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@opencl_headers",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "spi",
    hdrs = ["spi.h"],
    deps = [
        ":access_type",
        ":api_cl_gl_vk",
        ":status",
    ],
)

cc_library(
    name = "op_attrs",
    hdrs = ["op_attrs.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
