load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "llm_ops",
    srcs = glob(["*.cc"]),
    deps = [
        "//ml_drift/common:op_api",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor_handle",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)
