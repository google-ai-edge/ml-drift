"""Module extension for configuring Android NDK."""

load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")

def _if_android_ndk_home_impl(module_ctx):
    for mod in module_ctx.modules:
        for tag in mod.tags.configure:
            android_ndk_repository(
                name = "androidndk",
                api_level = tag.api_level,
            )
            return

if_android_ndk_home_tag = tag_class(attrs = {"api_level": attr.int()})

if_android_ndk_home_extension = module_extension(
    implementation = _if_android_ndk_home_impl,
    tag_classes = {"configure": if_android_ndk_home_tag},
)
