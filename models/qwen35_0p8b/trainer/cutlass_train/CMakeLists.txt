# CUTLASS training pipeline for Qwen3.5-0.8B on B200 (sm_100).
#
# Builds cutlass_train_C.so which is loadable from Python as
# torch.ops.cutlass_train_C.*. Requires CUTLASS 3.x headers; point
# CUTLASS_DIR at the repo root (e.g. /root/cutlass).
#
# Usage:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 -DCUTLASS_DIR=/root/cutlass
#   cmake --build build -j
#
# Expected build time: ~90s for the GEMM scaffold; FMHA bwd instantiation
# will push this to several minutes when fully wired.

cmake_minimum_required(VERSION 3.22)
project(cutlass_train_C LANGUAGES CUDA CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

if(NOT DEFINED CUTLASS_DIR)
    set(CUTLASS_DIR "/root/cutlass" CACHE PATH "Path to CUTLASS repo root")
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 100)

add_library(cutlass_train_C SHARED
    gemm_bf16_sm100.cu
    fmha_bwd.cu
    deltanet_bwd.cu
    torch_bindings.cpp
)

target_include_directories(cutlass_train_C PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/77_blackwell_fmha
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(cutlass_train_C
    ${TORCH_LIBRARIES}
)

target_compile_options(cutlass_train_C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -std=c++17
        --expt-relaxed-constexpr --expt-extended-lambda
        -D_GLIBCXX_USE_CXX11_ABI=1>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1>
)

set_target_properties(cutlass_train_C PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
    SUFFIX ".so"
)
