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

    The NVFP4 decode kernel (`kernel_gb10_nvfp4.cu`) is gated by
    `__CUDA_ARCH__ >= 1200`; the Python-visible NVFP4 op set is gated
    on the host side by `MEGAKERNEL_HAS_NVFP4`, defined here iff any
    listed arch is sm_120+ (Blackwell, e.g. GB10's sm_121a).
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


def _arch_major(arch: str) -> int:
    """sm_86 -> 8, sm_121a -> 12."""
    n = arch.removeprefix("sm_")
    digits = "".join(ch for ch in n if ch.isdigit())
    if not digits:
        return 0
    return int(digits[:-1]) if len(digits) >= 2 else int(digits)


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


archs = _detect_caps()
# Prefer `-arch=` for single-arch builds (matches the standalone
# megakernel build); fall back to multiple `-gencode` for multi-arch.
if len(archs) == 1:
    gencode_flags: list[str] = [f"-arch={archs[0]}"]
else:
    gencode_flags = []
    for a in archs:
        gencode_flags += ["-gencode", _arch_to_gencode(a)]

# Enable the NVFP4 (cuBLASLt FP4 LM head + sm_120 decode kernel) bindings
# when any target arch is sm_120+. Cleanly drops out for sm_86-only builds,
# so the 3090 .so is bit-identical to the upstream 3090-train build.
has_nvfp4 = any(_arch_major(a) >= 12 for a in archs)

block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 512)
lm_block_size = _int_env("MEGAKERNEL_LM_BLOCK_SIZE", 256)

nvcc_defines = [
    f"-DBLOCK_SIZE={block_size}",
    f"-DLM_BLOCK_SIZE={lm_block_size}",
]
cxx_defines = [
    f"-DBLOCK_SIZE={block_size}",
    f"-DLM_BLOCK_SIZE={lm_block_size}",
]
if has_nvfp4:
    nvcc_defines.append("-DMEGAKERNEL_HAS_NVFP4=1")
    cxx_defines.append("-DMEGAKERNEL_HAS_NVFP4=1")

# Sources: kernel_gb10_nvfp4.cu compiles to an empty TU on sm_86 (header
# guard at top of the file errors only inside a __CUDA_ARCH__ < 1200
# device compilation; host TU compilation is fine), so it's safe to keep
# it in the source list for all builds. cublasLt is only linked when an
# NVFP4-capable arch is targeted.
sources = [
    "torch_bindings.cpp",
    "kernel.cu",
    "kernel_gb10_nvfp4.cu",
    "prefill.cu",
    "prefill_megakernel.cu",
    "dn_chunked_3090.cu",  # 3090-tuned chunked DN forward (V_SPLITS=4, C=32)
    "fa_attn_aten.cpp",  # cuDNN FA-2 wrapper used by prefill.cu
]
# BF16 prefill body + cuBLASLt FP4 LM head — Blackwell only (uses
# cublasLt block-scaled FP4 path). Drop from sm_86-only builds.
if has_nvfp4:
    sources.append("prefill_bw.cu")
libraries = ["cublas"]
if has_nvfp4:
    libraries.append("cublasLt")

print(f"[megakernel-setup] target archs: {archs}")
print(f"[megakernel-setup] nvcc gencode: {' '.join(gencode_flags)}")
print(f"[megakernel-setup] NVFP4 bindings: {'on' if has_nvfp4 else 'off'}")

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", *cxx_defines],
                "nvcc": [
                    "-O3",
                    *gencode_flags,
                    "--use_fast_math",
                    "-std=c++17",
                    *nvcc_defines,
                ],
            },
            libraries=libraries,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
