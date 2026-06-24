load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:defs.bzl", "objc_library")
load("@tflite", "workspace_root")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

MIN_IOS_VERSION = "15.0"

objc_library(
    name = "buffer",
    srcs = ["buffer.mm"],
    hdrs = ["buffer.h"],
    deps = [
        ":common",
        ":gpu_object",
        "//ml_drift/common:status",
        "//ml_drift/common/task:buffer_desc",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/types:span",
    ],
)

objc_library(
    name = "buffer_test_lib",
    testonly = 1,
    srcs = ["buffer_test.mm"],
    deps = [
        ":buffer",
        "//ml_drift/common:types",
        "//third_party/apple_frameworks:Metal",
        "//third_party/apple_frameworks:XCTest",
    ],
)

objc_library(
    name = "common",
    srcs = ["common.mm"],
    hdrs = ["common.h"],
    deps = [
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "common_test_lib",
    testonly = 1,
    srcs = ["common_test.mm"],
    deps = [
        ":common",
        "//third_party/apple_frameworks:XCTest",
    ],
)

objc_library(
    name = "compute_task",
    srcs = ["compute_task.mm"],
    hdrs = ["compute_task.h"],
    deps = [
        ":buffer",
        ":common",
        ":environment",
        ":metal_arguments",
        ":metal_spatial_tensor",
        "//ml_drift/common:kernel_info",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:tuning_type",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

objc_library(
    name = "converter",
    srcs = ["converter.mm"],
    hdrs = ["converter.h"],
    deps = [
        ":buffer",
        ":compute_task",
        ":environment",
        ":metal_spatial_tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common/kernels:conversion",
        "//ml_drift/common/task:tensor_desc",
        "//third_party/apple_frameworks:Metal",
    ],
)

objc_library(
    name = "converter_test_lib",
    testonly = 1,
    srcs = ["converter_test.mm"],
    deps = [
        ":buffer",
        ":converter",
        ":environment",
        ":metal_spatial_tensor",
        "//ml_drift/metal/testing:test_util",
        "//third_party/apple_frameworks:Metal",
        "//third_party/apple_frameworks:XCTest",
    ],
)

objc_library(
    name = "environment",
    srcs = ["environment.mm"],
    hdrs = ["environment.h"],
    deps = [
        ":common",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//third_party/apple_frameworks:Metal",
        "@XNNPACK",
    ],
)

objc_library(
    name = "gpu_object",
    hdrs = ["gpu_object.h"],
    deps = [
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:status",
        "//ml_drift/common/task:gpu_object_desc",
        "//third_party/apple_frameworks:Metal",
    ],
)

objc_library(
    name = "inference_context",
    srcs = ["inference_context.mm"],
    hdrs = ["inference_context.h"],
    features = ["-layering_check"],
    deps = [
        ":common",
        ":compute_task",
        ":environment",
        ":inference_context_cc_fbs",
        ":memory_manager",
        ":metal_spatial_tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_cc_fbs",
        "//ml_drift/common:gpu_model_util",
        "//ml_drift/common:ir_model",
        "//ml_drift/common:ir_model_util",
        "//ml_drift/common:model",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:profiling_info",
        "//ml_drift/common/task:serialization_base",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:tuning_type",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

flatbuffer_cc_library(
    name = "inference_context_cc_fbs",
    srcs = ["inference_context.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I " + workspace_root,
    ],
    includes = [
        "//ml_drift/common:gpu_model_cc_fbs_includes",
        "//ml_drift/common/task:serialization_base_cc_fbs_includes",
    ],
)

objc_library(
    name = "memory_manager",
    srcs = ["memory_manager.mm"],
    hdrs = ["memory_manager.h"],
    features = ["-layering_check"],
    deps = [
        ":buffer",
        ":common",
        ":environment",
        ":metal_spatial_tensor",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:memory_management",
        "//ml_drift/common:model",
        "//ml_drift/common:status",
        "//ml_drift/common:types",
        "//ml_drift/common:util",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
    ],
)

objc_library(
    name = "metal_api",
    hdrs = ["metal_api.h"],
    deps = [
        ":environment",
        ":inference_context",
    ],
)

objc_library(
    name = "metal_arguments",
    srcs = ["metal_arguments.mm"],
    hdrs = ["metal_arguments.h"],
    deps = [
        ":buffer",
        ":environment",
        ":gpu_object",
        ":metal_spatial_tensor",
        "//ml_drift/common:status",
        "//ml_drift/common:util",
        "//ml_drift/common/task:arguments",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:util",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "metal_device",
    hdrs = ["metal_device.h"],
    deps = [
        ":environment",
    ],
)

objc_library(
    name = "metal_spatial_tensor",
    srcs = ["metal_spatial_tensor.mm"],
    hdrs = ["metal_spatial_tensor.h"],
    deps = [
        ":common",
        ":gpu_object",
        "//ml_drift/common:access_type",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:status",
        "//ml_drift/common/task:buffer_desc",
        "//ml_drift/common/task:gpu_object_desc",
        "//ml_drift/common/task:gpu_tensor",
        "//ml_drift/common/task:tensor_desc",
        "//third_party/apple_frameworks:Metal",
    ],
)

objc_library(
    name = "metal_spatial_tensor_test_lib",
    testonly = 1,
    srcs = ["metal_spatial_tensor_test.mm"],
    deps = [
        ":metal_spatial_tensor",
        "//ml_drift/common:types",
        "//third_party/apple_frameworks:Metal",
        "//third_party/apple_frameworks:XCTest",
    ],
)

objc_library(
    name = "metal_weights_manager",
    srcs = ["metal_weights_manager.mm"],
    hdrs = ["metal_weights_manager.h"],
    deps = [
        ":common",
        ":compute_task",
        ":environment",
        ":metal_spatial_tensor",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:status",
        "//ml_drift/common/task:gpu_operation",
        "//ml_drift/common/task:gpu_tensor",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

objc_library(
    name = "common_tests_lib",
    testonly = 1,
    srcs = [
        ":buffer_test.mm",
        ":common_test.mm",
        ":converter_test.mm",
        ":metal_spatial_tensor_test.mm",
        ":metal_weights_manager_test.mm",
    ],
    hdrs = [
    ],
    deps = [
        ":buffer",
        ":common",
        ":converter",
        ":environment",
        ":inference_context",
        ":metal_spatial_tensor",
        ":metal_weights_manager",
        "//ml_drift/common:data_type",
        "//ml_drift/common:gpu_info",
        "//ml_drift/common:gpu_model",
        "//ml_drift/common:gpu_model_builder",
        "//ml_drift/common:model",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:tensor",
        "//ml_drift/common:types",
        "//ml_drift/common/task:tensor_desc",
        "//ml_drift/common/task:weights_layout",
        "//ml_drift/metal/testing:test_util",
        "//third_party/apple_frameworks:Metal",
        "//third_party/apple_frameworks:XCTest",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
    ],
)

objc_library(
    name = "perf_util",
    testonly = 1,
    srcs = ["perf_util.mm"],
    hdrs = ["perf_util.h"],
    deps = [
        ":compute_task",
        ":environment",
        ":metal_spatial_tensor",
        "//ml_drift/common:data_type",
        "//ml_drift/common:flops_util",
        "//ml_drift/common:precision",
        "//ml_drift/common:shape",
        "//ml_drift/common:status",
        "//ml_drift/common/kernels:conv_apple_mpp",
        "//ml_drift/common/kernels:conv_generic",
        "//ml_drift/common/kernels:conv_wave_matrix",
        "//ml_drift/common/kernels:conv_wave_memory",
        "//ml_drift/common/kernels:depthwise_conv_tiled",
        "//ml_drift/common/kernels:elementwise",
        "//ml_drift/common/kernels:fully_connected",
        "//ml_drift/common/kernels:quantize_and_dequantize",
        "//ml_drift/common/kernels:softmax",
        "//ml_drift/common/kernels:winograd",
        "//ml_drift/common/kernels/special:conv_softmax_conv",
        "//ml_drift/common/task:testing_util",
        "//third_party/apple_frameworks:Metal",
        "@XNNPACK",
        "@com_google_absl//absl/random",
    ],
)
