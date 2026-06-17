# Copyright 2026 The ML Drift Authors.
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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "transform",
    srcs = ["transform.cc"],
    hdrs = ["transform.h"],
    deps = [
        "//ml_drift/common:ir_model",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "remove_noop_test",
    srcs = ["remove_noop_test.cc"],
    deps = [
        ":transform",
        "//ml_drift/common:data_type",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:operations",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
