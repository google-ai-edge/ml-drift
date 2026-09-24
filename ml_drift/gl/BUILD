load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//ml_drift/common:build_config.bzl", "mld_gpu_tests_tags")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    deps = [
        ":gl_buffer",
        ":gl_operation",
        ":gl_spatial_tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:types",
        "//ml_drift/common/kernels:conversion",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

GL_TEST_DEPS = select({
    "//conditions:default": [
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_matchers",
    ],
})

cc_test(
    name = "converter_test",
    srcs = ["converter_test.cc"],
    tags = mld_gpu_tests_tags(),
    deps = [
        ":converter",
        ":gl_buffer",
        ":gl_spatial_tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/gl/testing:gl_test",
        "@com_google_absl//absl/status",
    ] + GL_TEST_DEPS,
)

cc_library(
    name = "egl_context",
    srcs = ["egl_context.cc"],
    hdrs = ["egl_context.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "egl_environment",
    srcs = ["egl_environment.cc"],
    hdrs = ["egl_environment.h"],
    deps = [
        ":egl_context",
        ":egl_surface",
        ":gl_call",
        ":portable",
        ":request_gpu_info",
        "@XNNPACK",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ] + select({
        "//conditions:default": ["@com_google_absl//absl/debugging:leak_check"],
    }),
)

cc_library(
    name = "egl_surface",
    srcs = ["egl_surface.cc"],
    hdrs = ["egl_surface.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "gl_arguments",
    srcs = ["gl_arguments.cc"],
    hdrs = ["gl_arguments.h"],
    deps = [
        ":gl_buffer",
        ":gl_spatial_tensor",
        ":gl_texture_helper",
        ":gpu_object",
        ":portable",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gl_buffer",
    srcs = ["gl_buffer.cc"],
    hdrs = ["gl_buffer.h"],
    deps = [
        ":gpu_object",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "gl_call",
    hdrs = ["gl_call.h"],
    deps = [
        ":gl_errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "gl_errors",
    srcs = ["gl_errors.cc"],
    hdrs = ["gl_errors.h"],
    deps = [
        ":portable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gl_inference_context",
    srcs = ["gl_inference_context.cc"],
    hdrs = ["gl_inference_context.h"],
    deps = [
        ":gl_inference_context_cc_fbs",
        ":gl_operation",
        ":gl_spatial_tensor",
        ":memory_manager",
        ":portable",
        ":request_gpu_info",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:gpu_model_cc_fbs",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:ir_model_util",
        "//ml_drift/common:model",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:serialization_base",
        "//ml_drift/common/task:serialization_base_cc_fbs",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
    ],
)

flatbuffer_cc_library(
    name = "gl_inference_context_cc_fbs",
    srcs = ["gl_inference_context.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I .",
    ],
    includes = [
        "//ml_drift/common:gpu_model_cc_fbs_includes",
        "//ml_drift/common/task:serialization_base_cc_fbs_includes",
    ],
)

cc_library(
    name = "gl_operation",
    srcs = ["gl_operation.cc"],
    hdrs = ["gl_operation.h"],
    deps = [
        ":gl_arguments",
        ":gl_buffer",
        ":gl_shader",
        ":gl_spatial_tensor",
        ":portable",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:types",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@farmhash_archive//:farmhash",
    ],
)

cc_library(
    name = "gl_shader",
    srcs = ["gl_shader.cc"],
    hdrs = ["gl_shader.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "gl_spatial_tensor",
    srcs = ["gl_spatial_tensor.cc"],
    hdrs = ["gl_spatial_tensor.h"],
    deps = [
        ":gl_texture_helper",
        ":gpu_object",
        ":portable",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "gl_spatial_tensor_test",
    srcs = ["gl_spatial_tensor_test.cc"],
    tags = mld_gpu_tests_tags(),
    deps = [
        ":gl_spatial_tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/gl/testing:gl_test",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ] + GL_TEST_DEPS,
)

cc_library(
    name = "gl_sync",
    srcs = ["gl_sync.cc"],
    hdrs = ["gl_sync.h"],
    deps = [
        ":gl_buffer",
        ":gl_call",
        ":gl_errors",
        ":portable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "gl_texture_helper",
    srcs = ["gl_texture_helper.cc"],
    hdrs = ["gl_texture_helper.h"],
    deps = [
        ":portable",
        "//ml_drift/common:data_type",
    ],
)

cc_library(
    name = "gpu_object",
    hdrs = ["gpu_object.h"],
    deps = [
        ":portable",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common/task:gpu_object_desc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "memory_manager",
    srcs = ["memory_manager.cc"],
    hdrs = ["memory_manager.h"],
    deps = [
        ":gl_buffer",
        ":gl_spatial_tensor",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:memory_management",
        "//ml_drift/common:model",
        "//ml_drift/common:types",
        "//ml_drift/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
    ],
)

# Call binaries with --define gles_workstation_fix=yes as a workaround for the
# "version `GLIBC_2.33' not found" error on gLinux workstations. This extra define is needed in
# order not to break TAP tests, where the workaround does not apply.
# TODO(b/218917604): Remove workaround linkopts as soon as they are no longer needed.
config_setting(
    name = "enable_gles_workstation_fix",
    define_values = {
        "gles_workstation_fix": "yes",
    },
)

cc_library(
    name = "portable",
    hdrs = [
        "portable_egl.h",
        "portable_gl31.h",
    ],
    linkopts = select({
        "//ml_drift/gl:enable_gles_workstation_fix": ["-Wl,--dynamic-linker=/lib64/ld-linux-x86-64.so.2"],
        "//conditions:default": [],
    }),
    deps = [
        "@egl_headers",
        "@opengl_headers//:gles2_headers",
        "@opengl_headers//:gles3_headers",
    ],
)

cc_library(
    name = "request_gpu_info",
    srcs = ["request_gpu_info.cc"],
    hdrs = ["request_gpu_info.h"],
    deps = [
        ":gl_errors",
        ":portable",
        "//ml_drift/common:gpu_info",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "request_gpu_info_test",
    srcs = ["request_gpu_info_test.cc"],
    tags = mld_gpu_tests_tags(),
    deps = [
        ":request_gpu_info",
        "//ml_drift/common:gpu_info",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
