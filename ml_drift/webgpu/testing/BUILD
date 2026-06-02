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

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_test(
    name = "gpu_weights_conversion_test",
    srcs = ["gpu_weights_conversion_test.cc"],
    tags = [
        # TODO: b/454117562 - Fix the failures of the test and re-enable it.
        "notap",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":webgpu_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:gpu_weights_conversion_test_util",
        "//ml_drift/common:model",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/webgpu:execution_environment",
        "//ml_drift/webgpu:inference_context",
        "//ml_drift/webgpu:spatial_tensor",
        "//ml_drift/webgpu:webgpu_api_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

# For running on mac:
# blaze build -c opt --config=darwin_arm64 ml_drift/webgpu/testing:operation_perf_test
cc_test(
    name = "operation_perf_test",
    srcs = ["operation_perf_test.cc"],
    tags = [
        "local",
    ],
    deps = [
        ":perf_util",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common/default:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "perf_util",
    srcs = ["perf_util.cc"],
    hdrs = ["perf_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/kernels:conv_constants",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:conv_wave_matrix",
        "//ml_drift/common/kernels:conv_wave_memory",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/webgpu:compute_task",
        "//ml_drift/webgpu:environment",
        "//ml_drift/webgpu:execution_environment",
        "//ml_drift/webgpu:spatial_tensor",
        "//ml_drift/webgpu:webgpu_headers",
        "@XNNPACK",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "webgpu_test",
    testonly = 1,
    srcs = ["webgpu_test.cc"],
    hdrs = ["webgpu_test.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:status",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/webgpu:compute_task",
        "//ml_drift/webgpu:execution_environment",
        "//ml_drift/webgpu:preprocessor",
        "//ml_drift/webgpu:spatial_tensor",
        "//ml_drift/webgpu:webgpu_headers",
        "@com_google_googletest//:gtest",
    ],
)
