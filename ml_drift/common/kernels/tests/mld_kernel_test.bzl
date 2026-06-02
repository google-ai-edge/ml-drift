"""Macro for multi-backend C++ tests."""

load("@rules_cc//cc:defs.bzl", "cc_test")
load("//ml_drift/common:build_config.bzl", "mld_gpu_tests_tags")

def mld_kernel_test(
        name,
        srcs = [],
        deps = [],
        platforms = ["opencl", "webgpu"],
        **kwargs):
    """"Multi-backend C++ test macro.

    Can be used as a drop-in replacement for cc_test. Given name "my_test", defines these targets:
      my_test_opencl
      my_test_webgpu

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
                "@com_google_googletest//:gtest", "//ml_drift/common/default:status_matchers",
                "//ml_drift/cl/testing:cl_test",
            ],
            tags = mld_gpu_tests_tags(),
            **test_kwargs
        )
        target_suffixes.append("_opencl")

    if "webgpu" in platforms:
        cc_test(
            name = name + "_webgpu",
            srcs = srcs + [
                "//ml_drift/common/kernels/tests:kernel_test.h",
                "//ml_drift/common/kernels/tests:webgpu_kernel_test.cc",
            ],
            linkstatic = True,
            deps = deps + [
                "@com_google_googletest//:gtest", "//ml_drift/common/default:status_matchers",
                "//ml_drift/webgpu/testing:webgpu_test",
            ],
            tags = mld_gpu_tests_tags(),
            **test_kwargs
        )
        target_suffixes.append("_webgpu")
