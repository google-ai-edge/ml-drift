licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)

cc_library(
    name = "ruy",
    srcs = glob(
        ["ruy/**/*.cc"],
        exclude = [
            "ruy/benchmark.cc",
            "ruy/*_test.cc",
            "ruy/test*.cc",
            "ruy/profiler/test.cc",
        ],
    ),
    hdrs = glob(
        ["ruy/**/*.h"],
        exclude = [
            "ruy/test.h",
        ],
    ),
    includes = ["."],
    linkopts = select({
        ":windows": [],
        "//conditions:default": ["-pthread"],
    }),
    visibility = ["//visibility:public"],
)
