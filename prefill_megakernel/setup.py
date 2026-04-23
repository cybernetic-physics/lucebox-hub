"""Build the prefill megakernel extension."""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    import torch
except ImportError:
    torch = None


def _detect_arch():
    arch = os.environ.get("PREFILL_MEGA_ARCH")
    if arch:
        return arch
    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"
    return "sm_80"


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


arch = _detect_arch()
block_size = _int_env("PM_BLOCK_SIZE", 512)
num_blocks = _int_env("PM_NUM_BLOCKS", 148)

setup(
    name="prefill_megakernel",
    ext_modules=[
        CUDAExtension(
            name="prefill_megakernel_C",
            sources=[
                "torch_bindings.cpp",
                "kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    f"-arch={arch}",
                    "--use_fast_math",
                    "-std=c++17",
                    f"-DPM_BLOCK_SIZE={block_size}",
                    f"-DPM_NUM_BLOCKS={num_blocks}",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
