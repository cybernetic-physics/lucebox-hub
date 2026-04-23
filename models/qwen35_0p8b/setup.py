import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    import torch
except ImportError:
    torch = None


def _detect_arch():
    arch = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if arch:
        return arch

    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major == 12 and minor in (0, 1):
            return f"sm_{major}{minor}a"
        return f"sm_{major}{minor}"

    return "sm_86"


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


arch = _detect_arch()
block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 512)
lm_block_size = _int_env("MEGAKERNEL_LM_BLOCK_SIZE", 256)
prefill_dn_block_size = _int_env("MEGAKERNEL_PREFILL_DN_BLOCK_SIZE", 128)
prefill_dn_blocks_per_head = _int_env("MEGAKERNEL_PREFILL_DN_BLOCKS_PER_HEAD", 8)
prefill_fa_block_size = _int_env("MEGAKERNEL_PREFILL_FA_BLOCK_SIZE", 256)

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=[
                "torch_bindings.cpp",
                "kernel.cu",
                "kernel_gb10_nvfp4.cu",
                "prefill.cu",
                "prefill_megakernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    f"-arch={arch}",
                    "--use_fast_math",
                    "-std=c++17",
                    f"-DBLOCK_SIZE={block_size}",
                    f"-DLM_BLOCK_SIZE={lm_block_size}",
                    f"-DPREFILL_DN_BLOCK_SIZE={prefill_dn_block_size}",
                    f"-DPREFILL_DN_BLOCKS_PER_HEAD={prefill_dn_blocks_per_head}",
                    f"-DPREFILL_FA_BLOCK_SIZE={prefill_fa_block_size}",
                ],
            },
            libraries=["cublas", "cublasLt"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
