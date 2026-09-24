package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "gles2_headers",
    hdrs = [
        "GLES2/gl2.h",
        "GLES2/gl2ext.h",
        "GLES2/gl2platform.h",
    ],
    includes = ["."],
    linkopts = ["-lGLESv2"],
)

cc_library(
    name = "gles3_headers",
    hdrs = [
        "GLES3/gl3.h",
        "GLES3/gl31.h",
        "GLES3/gl32.h",
        "GLES3/gl3platform.h",
    ],
    includes = ["."],
    linkopts = select({
        "@platforms//os:android": ["-lGLESv3"],
        "//conditions:default": ["-lGLESv2"],
    }),
)
