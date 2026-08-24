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
    name = "qcom_command_buffer",
    srcs = ["qcom_command_buffer.cc"],
    deps = [
        "//ml_drift/cl:cl_context",
        "//ml_drift/cl:cl_device",
        "//ml_drift/cl:command_buffer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "qcom_wrapper",
    hdrs = ["qcom_wrapper.h"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    deps = [
        "//ml_drift/cl:opencl_wrapper",
        "//ml_drift/cl:util_types",
        "//ml_drift/common:gpu_info",
        "@com_google_absl//absl/status",
    ],
)
