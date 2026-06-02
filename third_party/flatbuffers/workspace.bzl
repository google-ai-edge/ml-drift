"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def _flatbuffers_ext_impl(mctx):
    for mod in mctx.modules:
        for dep in mod.tags.build:
            third_party_http_archive(
                name = "flatbuffers",
                strip_prefix = dep.strip_prefix,
                sha256 = dep.sha256,
                urls = dep.urls,
                build_file = dep.build_file,
                delete = dep.delete,
                link_files = dep.link_files,
            )

flatbuffers = module_extension(
    implementation = _flatbuffers_ext_impl,
    tag_classes = {
        "build": tag_class(attrs = {
            "sha256": attr.string(),
            "urls": attr.string_list(),
            "strip_prefix": attr.string(),
            "build_file": attr.string(),
            "delete": attr.string_list(),
            "link_files": attr.string_dict(),
        }),
    },
)
