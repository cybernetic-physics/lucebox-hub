"""Build cutlass_train_C via torch's CUDAExtension toolchain.

Simpler than the CMake path while the kernels are still landing —
CMake is the production build; this setup.py is the dev fast-iter path.

Usage:
    cd models/qwen35_0p8b/trainer/cutlass_train
    MAX_JOBS=4 python3 setup.py build_ext --inplace
    python3 test_gemm.py
"""
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS_DIR = os.environ.get("CUTLASS_DIR", "/root/cutlass")
# CUTLASS Blackwell kernels need the 'a' arch variant (tcgen05, TMA).
cc = torch.cuda.get_device_capability()
arch = f"sm_{cc[0]}{cc[1]}a"

setup(
    name="cutlass_train",
    ext_modules=[CUDAExtension(
        name="cutlass_train_C",
        sources=[
            "torch_bindings.cpp",
            "gemm_bf16_sm100.cu",
            "fmha_bwd.cu",
            "deltanet_bwd.cu",
        ],
        include_dirs=[
            f"{CUTLASS_DIR}/include",
            f"{CUTLASS_DIR}/tools/util/include",
            f"{CUTLASS_DIR}/examples/77_blackwell_fmha",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", f"-arch={arch}", "--use_fast_math", "-std=c++17",
                     "--expt-relaxed-constexpr", "--expt-extended-lambda",
                     # CUTLASS needs these for sm_100 tcgen05/TMA intrinsics.
                     "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
                     ],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
