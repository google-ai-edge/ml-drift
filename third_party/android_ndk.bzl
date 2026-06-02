"""Module extension for conditionally configuring Android NDK."""

load("@bazel_tools//tools/build_defs/repo:repo.bzl", "repo_id")
load("@rules_android_ndk//:toolchains/android_ndk_repository.bzl", "android_ndk_repository")

def _if_android_ndk_home_impl(module_ctx):
    if module_ctx.os.environ.get("ANDROID_NDK_HOME"):
        configured = False
        for mod in module_ctx.modules:
            for tag in mod.tags.configure:
                if not configured:
                    android_ndk_repository(
                        name = "androidndk",
                        api_level = tag.api_level,
                    )
                    native.register_toolchains("@androidndk//:all")
                    configured = True
        if configured:
            return [repo_id("@androidndk")]
    return []

if_android_ndk_home_tag = tag_class(attrs = {"api_level": attr.int()})

if_android_ndk_home_extension = module_extension(
    implementation = _if_android_ndk_home_impl,
    tag_classes = {"configure": if_android_ndk_home_tag},
)
