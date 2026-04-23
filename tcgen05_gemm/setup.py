import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

arch = os.environ.get("TCGEN05_ARCH", "sm_100a")

setup(
    name="tcgen05_gemm",
    ext_modules=[CUDAExtension(
        name="tcgen05_gemm_C",
        sources=["torch_bindings.cpp", "kernel.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", f"-arch={arch}", "--use_fast_math", "-std=c++17"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
