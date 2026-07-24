licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "libdawn",
    hdrs = glob([
        "include/**/*.h*",
    ]),
    includes = [
        "include",
    ],
    srcs = [
        "lib/libdawn.so",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "webgpu_dawn",
    actual = ":libdawn",
    visibility = ["//visibility:public"],
)

alias(
    name = "webgpu_headers",
    actual = ":libdawn",
    visibility = ["//visibility:public"],
)
