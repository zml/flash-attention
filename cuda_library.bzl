load("@rules_cc//cc:defs.bzl", "cc_library")

def cuda_library(copts = [], defines = [], archs = [], additional_compiler_inputs = [], **kwargs):
    """Wrapper over cc_library which adds default CUDA options."""

    gpu_archs = ["--cuda-gpu-arch=" + arch for arch in archs]

    cc_library(
        copts = copts + [
            "-x", "cuda",
            "--cuda-path=$(location {})".format(Label("@cuda//cuda:cuda_path")),
        ] + gpu_archs,
        additional_compiler_inputs = additional_compiler_inputs + [
            Label("@cuda//cuda:cuda_path"),
        ],
        defines = defines + [
            "_ALLOW_UNSUPPORTED_LIBCPP",
        ],
        **kwargs,
    )

