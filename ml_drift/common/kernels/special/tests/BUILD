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

load("@rules_cc//cc:defs.bzl", "cc_test")
load("//ml_drift/common:build_config.bzl", "mld_gpu_tests_tags")

package(
    default_hdrs_check = "strict",
    default_testonly = 1,
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_test(
    name = "concat_conv_test",
    srcs = [
        "concat_conv_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:concat_conv",
        "//ml_drift/common/kernels/special:thin_local_memory_fuser",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "conv2x2_max_pool2x2_test",
    srcs = [
        "conv2x2_max_pool2x2_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:conv2x2_max_pool2x2",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "conv_softmax_conv_test",
    srcs = [
        "conv_softmax_conv_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels/special:conv_softmax_conv",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@XNNPACK",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dw7x7_conv2to6_concat_conv8to8_test",
    srcs = [
        "dw7x7_conv2to6_concat_conv8to8_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:dw7x7_conv2to6_concat_conv8to8",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dw3x3_conv16to16_conv16to32_add_conv32to16_test",
    srcs = [
        "dw3x3_conv16to16_conv16to32_add_conv32to16_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:dw3x3_conv16to16_conv16to32_add_conv32to16",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8_test",
    srcs = [
        "dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32_test",
    srcs = [
        "dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "thin_local_memory_fuser_test",
    srcs = [
        "thin_local_memory_fuser_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:thin_local_memory_fuser",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "thin_pointwise_fuser_test",
    srcs = [
        "thin_pointwise_fuser_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels/special:thin_pointwise_fuser",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)
