load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gl_test",
    testonly = 1,
    srcs = ["gl_test.cc"],
    hdrs = ["gl_test.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/gl:egl_environment",
        "//ml_drift/gl:gl_operation",
        "//ml_drift/gl:gl_spatial_tensor",
        "//ml_drift/gl:portable",
        "//ml_drift/gl:request_gpu_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "gpu_weights_conversion_test",
    srcs = ["gpu_weights_conversion_test.cc"],
    tags = [
        # TODO: Fix the failures of the test and re-enable it.
        "notap",
        "requires-gpu-sm90-full",
    ],
    deps = [
        ":gl_test",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:gpu_weights_conversion_test_util",
        "//ml_drift/common:model",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:util",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/common/task:weights_conversion",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/gl:egl_environment",
        "//ml_drift/gl:gl_inference_context",
        "//ml_drift/gl:gl_spatial_tensor",
        "//ml_drift/gl:portable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "operation_perf_test",
    srcs = ["operation_perf_test.cc"],
    tags = ["local"],
    deps = [
        ":perf_util",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "perf_util",
    srcs = ["perf_util.cc"],
    hdrs = ["perf_util.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:operations",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/kernels:conv_constants",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:conv_wave_memory",
        "//ml_drift/common/kernels:depthwise_conv_tiled",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/gl:egl_environment",
        "//ml_drift/gl:gl_operation",
        "//ml_drift/gl:gl_spatial_tensor",
        "//ml_drift/gl:portable",
        "//ml_drift/gl:request_gpu_info",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/time",
    ],
)
