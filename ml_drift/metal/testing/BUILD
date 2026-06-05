load("@rules_cc//cc:defs.bzl", "objc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

objc_library(
    name = "test_util",
    testonly = 1,
    srcs = [
        "test_util.cc",
    ],
    hdrs = ["test_util.h"],
    copts = [
        "-ObjC++",
    ],
    deps = [
        "//ml_drift/common:convert",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/default:status_matchers",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:testing_util",
        "//ml_drift/metal:compute_task",
        "//ml_drift/metal:metal_device",
        "//ml_drift/metal:metal_spatial_tensor",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)
