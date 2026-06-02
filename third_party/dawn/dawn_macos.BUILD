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
        "lib/libdawn.dylib",
    ],
    visibility = ["//visibility:public"],
)
