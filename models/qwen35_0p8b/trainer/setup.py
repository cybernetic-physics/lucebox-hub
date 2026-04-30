import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

cc = torch.cuda.get_device_capability()
arch = os.environ.get("TRAIN_MEGA_ARCH", f"sm_{cc[0]}{cc[1]}")

# PM_NUM_BLOCKS is the *upper bound* on the cooperative-launch grid; the
# kernel further clamps to cudaOccupancyMaxActiveBlocksPerMultiprocessor
# at runtime, so this can stay >= the largest GPU's SM count without
# affecting smaller GPUs. Default the per-arch value:
#   B200  148 SMs  -> 148
#   3090   82 SMs  -> 148 (clamps down at launch, no over-subscription)
# A user can override with TRAIN_MEGA_NUM_BLOCKS to bench tighter caps.
sm_count_envs = {
    (10, 0): 148,    # B200
    (12, 0): 132,    # consumer Blackwell
    (9, 0): 132,     # H100
    (8, 6): 82,      # RTX 3090
    (8, 9): 76,      # RTX 4090
    (8, 0): 108,     # A100
}
default_num_blocks = sm_count_envs.get(cc, 148)
num_blocks = int(os.environ.get("TRAIN_MEGA_NUM_BLOCKS", default_num_blocks))
block_size = int(os.environ.get("TRAIN_MEGA_BLOCK_SIZE", 512))

setup(
    name="train_megakernel",
    ext_modules=[CUDAExtension(
        name="train_megakernel_C",
        sources=["torch_bindings.cpp", "kernel.cu", "dn_bwd.cu", "dn_chunked.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", f"-arch={arch}", "--use_fast_math", "-std=c++17",
                     "-lineinfo",
                     f"-DPM_BLOCK_SIZE={block_size}",
                     f"-DPM_NUM_BLOCKS={num_blocks}"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
