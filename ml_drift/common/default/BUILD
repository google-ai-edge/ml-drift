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
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["//third_party/odml/litert/ml_drift/tflite:custom_parsers.h"],
    deps = [
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//third_party/odml/litert/ml_drift/tflite:operation_parser",
        "//third_party/odml/litert/ml_drift/tflite:unimplemented_operation_parser",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "custom_transformations",
    srcs = ["custom_transformations.cc"],
    hdrs = ["//ml_drift/common:custom_transformations.h"],
    deps = [
        "//ml_drift/common:model_transformer",
    ],
)

cc_library(
    name = "status_macros",
    srcs = ["status_macros.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = True,
    srcs = ["status_matchers.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)
