"""Macro for multi-backend C++ tests."""

load("@rules_cc//cc:defs.bzl", "cc_test", "objc_library")
load("//ml_drift/common:build_config.bzl", "mld_gpu_full_tests_tags", "mld_gpu_tests_tags")

def mld_kernel_test(
        name,
        srcs = [],
        deps = [],
        platforms = ["opencl", "gl", "webgpu", "mac"],
        **kwargs):
    """"Multi-backend C++ test macro.

    Can be used as a drop-in replacement for cc_test. Given name "my_test", defines these targets:
      my_test_opencl
      my_test_gl
      my_test_webgpu
      my_test_objc_lib (objc library for mac)
      my_test_mac

    Args:
      name: name of the test target to define.
      srcs: source files.
      deps: dependencies.
      platforms: list of platforms to include.
      **kwargs: all other arguments are forwarded to test targets.
    """
    test_kwargs = dict(kwargs)
    library_kwargs = dict(kwargs)

    # Remove test-only attributes from library_kwargs
    test_only_attrs = ["timeout", "size", "flaky", "shard_count", "args"]
    for attr in test_only_attrs:
        if attr in library_kwargs:
            library_kwargs.pop(attr)

    target_suffixes = []  # List of tests added by this macro for suite.
    if "opencl" in platforms:
        cc_test(
            name = name + "_opencl",
            srcs = srcs + [
                "//ml_drift/common/kernels/tests:kernel_test.h",
                "//ml_drift/common/kernels/tests:opencl_kernel_test.cc",
            ],
            linkstatic = True,
            deps = deps + [
                "@com_google_googletest//:gtest", "@com_google_absl//absl/status:status_matchers",
                "//ml_drift/cl/testing:cl_test",
            ],
            tags = mld_gpu_tests_tags(),
            **test_kwargs
        )
        target_suffixes.append("_opencl")

    if "gl" in platforms:
        cc_test(
            name = name + "_gl",
            srcs = srcs + [
                "//ml_drift/common/kernels/tests:kernel_test.h",
                "//ml_drift/common/kernels/tests:gl_kernel_test.cc",
            ],
            linkstatic = True,
            deps = deps + [
                "@com_google_googletest//:gtest", "@com_google_absl//absl/status:status_matchers",
                "//ml_drift/gl/testing:gl_test",
            ],
            tags = mld_gpu_tests_tags(),
            **test_kwargs
        )
        target_suffixes.append("_gl")

    if "webgpu" in platforms:
        cc_test(
            name = name + "_webgpu",
            srcs = srcs + [
                "//ml_drift/common/kernels/tests:kernel_test.h",
                "//ml_drift/common/kernels/tests:webgpu_kernel_test.cc",
            ],
            linkstatic = True,
            deps = deps + [
                "@com_google_googletest//:gtest", "@com_google_absl//absl/status:status_matchers",
                "//ml_drift/webgpu/testing:webgpu_test",
            ],
            tags = mld_gpu_full_tests_tags(),
            **test_kwargs
        )
        target_suffixes.append("_webgpu")

    if "mac" in platforms:
        objc_library(
            name = name + "_objc_lib",
            srcs = srcs + [
                "//ml_drift/common/kernels/tests:kernel_test.h",
                "//ml_drift/common/kernels/tests:metal_kernel_test.mm",
            ],
            testonly = True,
            alwayslink = True,
            deps = deps + [
                "@com_google_googletest//:gtest",
                "@com_google_absl//absl/status:status_matchers",
                "//third_party/apple_frameworks:Metal",
                "//ml_drift/metal/testing:test_util",
            ],
            **library_kwargs
        )
        cc_test(
            name = name + "_mac",
            tags = ["mac", "nobuilder", "nozapfhahn"],
            deps = [
                name + "_objc_lib",
            ],
            **test_kwargs
        )
