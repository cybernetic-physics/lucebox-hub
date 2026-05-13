import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    import torch
except ImportError:
    torch = None


def _detect_caps() -> list[str]:
    """Pick the target SM architectures for nvcc.

    Priority:
      1. $MEGAKERNEL_CUDA_ARCHS = "sm_86,sm_121a" (any sm_NNa list, comma-separated).
         Single-arch shorthand "sm_86" is also accepted here.
      2. $MEGAKERNEL_CUDA_ARCH = "sm_86" (legacy single-arch env, kept for
         backwards compatibility with the 3090-train build invocation).
      3. torch.cuda.get_device_capability() of the visible device.
      4. sm_86 fallback.

    Both kernel.cu (sm_86 BF16 megakernel) and kernel_gb10_nvfp4.cu
    (sm_121a NVFP4 decode) compile cleanly on all listed archs because
    arch-specific code paths are gated by `#if __CUDA_ARCH__ >= 1200`.
    """
    archs_env = os.environ.get("MEGAKERNEL_CUDA_ARCHS")
    if archs_env:
        return [a.strip() for a in archs_env.split(",") if a.strip()]

    single = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if single:
        return [single]

    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # cap 12 == Blackwell consumer/GB10 (sm_120/121); use the
        # arch-specific `a` variant so tcgen05/mxf4 intrinsics resolve.
        if major >= 12:
            return [f"sm_{major}{minor}a"]
        return [f"sm_{major}{minor}"]

    return ["sm_86"]


def _arch_to_gencode(arch: str) -> str:
    """sm_86 -> compute_86,code=sm_86; sm_121a -> compute_121a,code=sm_121a."""
    n = arch.removeprefix("sm_")
    return f"arch=compute_{n},code=sm_{n}"


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


archs = _detect_caps()
gencode_flags: list[str] = []
for a in archs:
    gencode_flags += ["-gencode", _arch_to_gencode(a)]

block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 512)
lm_block_size = _int_env("MEGAKERNEL_LM_BLOCK_SIZE", 256)

print(f"[megakernel-setup] target archs: {archs}")
print(f"[megakernel-setup] nvcc gencode: {' '.join(gencode_flags)}")

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
                "dn_chunked_3090.cu",  # 3090-tuned chunked DN forward (V_SPLITS=4, C=32)
                "fa_attn_aten.cpp",  # cuDNN FA-2 wrapper used by prefill.cu
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    *gencode_flags,
                    "--use_fast_math",
                    "-std=c++17",
                    f"-DBLOCK_SIZE={block_size}",
                    f"-DLM_BLOCK_SIZE={lm_block_size}",
                ],
            },
            libraries=["cublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
