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
    linkstatic = 1,
)

cc_test(
    name = "convolution_test",
    srcs = [
        "convolution_test.cc",
        "//ml_drift/common/kernels/tests:kernel_test.h",
        "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
    ],
    linkstatic = 1,
    tags = mld_gpu_tests_tags(),
    deps = [
        "//ml_drift/cl/testing:cl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common:winograd_util",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_ref_ops",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "@XNNPACK",
        "@com_google_googletest//:gtest",
    ],
)
