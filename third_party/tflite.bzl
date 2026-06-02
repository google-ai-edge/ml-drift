"""Bazel module extension for building TFLite as a shared library."""

_module_attrs = {
    "url": attr.string(mandatory = True),
    "sha256": attr.string(mandatory = True),
    "strip_prefix": attr.string(mandatory = True),
}

_repo_attrs = _module_attrs | {
    "build_for_android": attr.bool(mandatory = True),
}

def _execute(ctx, cmd, **kwargs):
    ctx.execute(cmd.split(" "), quiet = False, **kwargs)

def _configure_for_x86(ctx):
    ctx.execute([
        "cmake",
        "-DTENSORFLOW_SOURCE_DIR=.",
        "-DBUILD_SHARED_LIBS=ON",
        "-DTFLITE_ENABLE_XNNPACK=OFF",
        "-DTFLITE_ENABLE_GPU=ON",
        "-DCMAKE_CXX_FLAGS=-DCL_TARGET_OPENCL_VERSION=300",
        "tensorflow/lite",
    ], quiet = False)

def _configure_for_android(ctx):
    # Now configure cmake to build the TFLite library for cross-compilation on Android.
    android_ndk_home = ctx.getenv("ANDROID_NDK_HOME")
    if android_ndk_home == None:
        fail("The environment variable ANDROID_NDK_HOME needs to be defined.")

    ctx.execute([
        "cmake",
        "-DTENSORFLOW_SOURCE_DIR=.",
        "-DCMAKE_TOOLCHAIN_FILE=%s/build/cmake/android.toolchain.cmake" % android_ndk_home,
        "-DANDROID_ABI=arm64-v8a",
        "-DBUILD_SHARED_LIBS=ON",
        "-DTFLITE_ENABLE_XNNPACK=OFF",
        "-DTFLITE_ENABLE_GPU=ON",
        "-DCMAKE_CXX_FLAGS=-DCL_TARGET_OPENCL_VERSION=300",
        "tensorflow/lite",
    ], quiet = False)

def _tflite_repo_rule_impl(ctx):
    url = ctx.attr.url
    sha256 = ctx.attr.sha256
    strip_prefix = ctx.attr.strip_prefix
    build_for_android = ctx.attr.build_for_android

    # Download and extract the TensorFlow library.
    ctx.download_and_extract(
        url = url,
        sha256 = sha256,
        strip_prefix = strip_prefix,
    )

    execute = lambda cmd: _execute(ctx, cmd)

    # Configure cmake to build the shared TFLite library.
    if build_for_android:
        _configure_for_android(ctx)
    else:
        _configure_for_x86(ctx)

    # Build the shared library with cmake.
    execute("cmake --build . -j128")

    # Remove all BUILD files from source tree since otherwise glob will not match them and miss the
    # required headers.
    execute("find -name BUILD -delete")

    # Make sure this is recognized as a repository.
    ctx.file("MODULE.bazel", "")

    # Add a build file providing the appropriate headers and shared library.
    ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

cc_library(
    name='tflite',
    hdrs = glob(['tensorflow/**/*.h']),
    srcs = ['libtensorflow-lite.so'],
    deps = ["@eigen//:eigen", "@flatbuffers//:flatbuffers", "@gemmlowp//:gemmlowp", "@ruy//:ruy"],
)
""")

_tflite_repo_rule = repository_rule(
    implementation = _tflite_repo_rule_impl,
    attrs = _repo_attrs,
)

def _tflite_impl(ctx):
    # There is assumed to be only one module that calls this extension once thus providing only
    # one set of parameters.
    url = ""
    sha256 = ""
    strip_prefix = ""
    num_times_parameters_assigned = 0
    for mod in ctx.modules:
        for build in mod.tags.build:
            url = build.url
            sha256 = build.sha256
            strip_prefix = build.strip_prefix
            num_times_parameters_assigned += 1

    # Verify that there was only one set of parameters.
    if num_times_parameters_assigned != 1:
        fail("A single set of parameters is expected.")

    # Instantiate the repository and build the TFLite shared library.
    _tflite_repo_rule(name = "tflite", url = url, sha256 = sha256, strip_prefix = strip_prefix, build_for_android = False)

    # Do the same for android.
    _tflite_repo_rule(name = "tflite-android", url = url, sha256 = sha256, strip_prefix = strip_prefix, build_for_android = True)

tflite = module_extension(
    tag_classes = {
        "build": tag_class(
            attrs = _module_attrs,
        ),
    },
    implementation = _tflite_impl,
)
