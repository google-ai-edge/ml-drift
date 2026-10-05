load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_rust//rust:defs.bzl", "rust_static_library")

package(
    default_visibility = ["//visibility:public"],
)

rust_static_library(
    name = "huggingface_tokenizer_impl",
    srcs = ["rust/src/lib.rs"],
    edition = "2018",
    proc_macro_deps = [],
    rustc_flags = [
        "-A",
        "dangerous_implicit_autorefs",
    ],
    deps = [
        "@crate_index//:serde_json",
        "@crate_index//:tokenizers",
    ],
)

cc_library(
    name = "huggingface_tokenizer",
    srcs = ["src/huggingface_tokenizer.cc"],
    hdrs = [
        "include/tokenizers_c.h",
        "include/tokenizers_cpp.h",
    ],
    includes = ["include"],
    linkopts = select({
        "@platforms//os:windows": [
            "kernel32.lib",
            "ntdll.lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":huggingface_tokenizer_impl",
    ],
)
