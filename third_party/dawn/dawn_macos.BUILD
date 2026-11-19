licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "libdawn",
    srcs = select({
        "@platforms//os:android": [],
        "//conditions:default": [
            "lib/libdawn.dylib",
        ],
    }),
    hdrs = glob([
        "include/**/*.h*",
    ]),
    includes = [
        "include",
    ],
    linkopts = select({
        "@platforms//os:android": [
            "-Wl,--warn-unresolved-symbols",
        ],
        "//conditions:default": [],
    }),
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
