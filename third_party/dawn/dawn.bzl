"""Bazel module extension for building Dawn as a local repository."""

def _local_repository_impl(ctx):
    ctx.report_progress("Fetching files from %s" % ctx.attr.path)
    for folder in ["include", "lib"]:
        ctx.execute(["mkdir", ctx.path(folder)])
        copy_result = ctx.execute(["bash", "-c", "cp -RL %s/%s/. %s" % (ctx.attr.path, folder, ctx.path(folder))])
        if copy_result.return_code != 0:
            fail("Error copying dir %s from %s: %s" % (folder, ctx.attr.path, copy_result.stderr))
    ctx.template("BUILD", ctx.attr.build_file_label, {}, False)

_local_repository = repository_rule(
    implementation = _local_repository_impl,
    attrs = {
        "path": attr.string(mandatory = True),
        "build_file_label": attr.label(mandatory = True),
    },
)

def _local_non_bazel_impl(module_ctx):
    for mod in module_ctx.modules:
        for tag in mod.tags.local_repo:
            _local_repository(
                name = tag.repo_name,
                build_file_label = tag.build_file,
                path = tag.path,
            )

local_repo_tag = tag_class(attrs = {
    "repo_name": attr.string(),
    "build_file": attr.label(),
    "path": attr.string(),
})

local_non_bazel_repo = module_extension(
    implementation = _local_non_bazel_impl,
    tag_classes = {"local_repo": local_repo_tag},
)
