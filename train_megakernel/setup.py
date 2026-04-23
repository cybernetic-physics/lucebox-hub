import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

arch = os.environ.get("TRAIN_MEGA_ARCH",
                      f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")

setup(
    name="train_megakernel",
    ext_modules=[CUDAExtension(
        name="train_megakernel_C",
        sources=["torch_bindings.cpp", "kernel.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", f"-arch={arch}", "--use_fast_math", "-std=c++17",
                     "-DPM_BLOCK_SIZE=512", "-DPM_NUM_BLOCKS=148"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
