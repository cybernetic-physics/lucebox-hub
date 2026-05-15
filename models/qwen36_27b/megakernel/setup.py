"""Build the Qwen3.x templated megakernel scaffold.

Compiles `kernel_decode.cu` + `torch_bindings.cpp` into the
`qwen3x_C` torch extension. Produces both Cfg_0p8B and Cfg_27B
specializations of the MLP smoke kernel as a proof-of-concept for the
template-on-Cfg approach.

Build:
    cd models/qwen36_27b/megakernel
    /home/sparkz/rl/.venv/bin/python3 setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    import torch
except ImportError:
    torch = None


def _detect_arch() -> str:
    env = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if env:
        return env
    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 12:
            return f"sm_{major}{minor}a"
        return f"sm_{major}{minor}"
    return "sm_86"


arch = _detect_arch()
print(f"[qwen3x_C build] target arch: {arch}")

setup(
    name="qwen3x_C",
    ext_modules=[
        CUDAExtension(
            name="qwen3x_C",
            sources=["torch_bindings.cpp", "kernel_decode.cu",
                     "kernel_decode_full.cu", "prefill_megakernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    f"-arch={arch}",
                    "--use_fast_math",
                    "-std=c++17",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
