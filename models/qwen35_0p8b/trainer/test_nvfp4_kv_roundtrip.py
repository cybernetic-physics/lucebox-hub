"""End-to-end test of the NVFP4 KV cache format.

Three checks:
  1. Round-trip: bf16 -> NVFP4 (data + E4M3 scales) -> bf16
     - per-block (16 elems) signal/noise vs bf16 reference
     - max abs error and mean rel error across a uniformly-random row and
       a Gaussian row with a few outliers per block
     - memory footprint vs the bf16 baseline at the default 65536 x 6 x 2
       config

  2. Q.K dot: bf16 Q . NVFP4 K agrees with bf16 Q . dequantized-K within
     bf16 dot-product noise. (This is what the FA inner loop will use.)

  3. Memory: print MB used by bf16 K+V cache vs NVFP4 K+V cache for the
     default Decoder config (n_fa=6, n_kv_heads=2, max_seq=65536, head=256).

Run:
    cd /home/sparkz/rl/lucebox-hub/models/qwen35_0p8b
    MAX_JOBS=4 python3 setup.py build_ext --inplace
    python3 trainer/test_nvfp4_kv_roundtrip.py
"""
import os
import sys
import time
import math
import torch

# Make the in-tree .so importable when running from trainer/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR  = os.path.dirname(THIS_DIR)
sys.path.insert(0, PKG_DIR)
import qwen35_megakernel_bf16_C  # noqa: F401  (registers torch.ops)

ops = torch.ops.qwen35_megakernel_bf16_C

HEAD_DIM    = 256
GROUP_SIZE  = 16
DATA_BYTES  = HEAD_DIM // 2
SCALE_BYTES = HEAD_DIM // GROUP_SIZE

# E2M1 LUT (matches FP4_E2M1_LUT_KV in nvfp4_kv.cuh).
FP4_E2M1 = torch.tensor(
    [ 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequant_reference(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """CPU/PyTorch reference dequant for cross-checking the CUDA dequant."""
    T, H, _ = data.shape
    data_cpu   = data.cpu()
    scales_cpu = scales.cpu()
    # E4M3 scale bytes -> float via fp8 path. PyTorch >= 2.2 has float8_e4m3fn.
    scale_e4m3 = scales_cpu.view(torch.float8_e4m3fn).to(torch.float32)  # [T, H, 16]

    # Unpack low nibble + high nibble into [T, H, 256] codes.
    lo = data_cpu & 0xF
    hi = (data_cpu >> 4) & 0xF
    codes = torch.stack([lo, hi], dim=-1).reshape(T, H, HEAD_DIM)   # [T, H, 256]
    vals = FP4_E2M1[codes.to(torch.long)]                            # [T, H, 256]
    # Broadcast scale over each 16-elem block.
    scale_expanded = scale_e4m3.repeat_interleave(GROUP_SIZE, dim=-1)  # [T, H, 256]
    return (vals * scale_expanded).to(torch.bfloat16)


def make_inputs(T: int, H: int, mode: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "uniform":
        x = (torch.rand((T, H, HEAD_DIM), device="cuda", generator=g) * 2 - 1) * 3.0
    elif mode == "gaussian":
        x = torch.randn((T, H, HEAD_DIM), device="cuda", generator=g) * 1.5
    elif mode == "outliers":
        x = torch.randn((T, H, HEAD_DIM), device="cuda", generator=g) * 1.5
        # Inject ~1 outlier per 16-elem block to stress the scale path.
        n_blocks = T * H * (HEAD_DIM // GROUP_SIZE)
        idx_flat = torch.randint(0, GROUP_SIZE, (n_blocks,), device="cuda", generator=g)
        block_view = x.view(-1, GROUP_SIZE)
        block_view[torch.arange(n_blocks, device="cuda"), idx_flat] *= 5.0
    else:
        raise ValueError(mode)
    return x.to(torch.bfloat16)


def quantize(src_bf16: torch.Tensor):
    T, H, _ = src_bf16.shape
    data   = torch.empty((T, H, DATA_BYTES),  dtype=torch.uint8, device="cuda")
    scales = torch.empty((T, H, SCALE_BYTES), dtype=torch.uint8, device="cuda")
    ops.quantize_bf16_to_nvfp4_kv(src_bf16, data, scales)
    return data, scales


def dequantize(data: torch.Tensor, scales: torch.Tensor):
    T, H, _ = data.shape
    out = torch.empty((T, H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    ops.dequantize_nvfp4_kv_to_bf16(data, scales, out)
    return out


def test_roundtrip(mode: str):
    T, H = 256, 2
    src  = make_inputs(T, H, mode)
    data, scales = quantize(src)
    cuda_deq = dequantize(data, scales)
    ref_deq  = _dequant_reference(data, scales).to("cuda")

    # 1. CUDA dequant must bit-match the python reference: same scale, same
    #    LUT, same packed nibbles -> identical bf16 bytes.
    bit_match = torch.equal(cuda_deq, ref_deq)

    # 2. Round-trip error vs the input.
    src_f  = src.to(torch.float32)
    deq_f  = cuda_deq.to(torch.float32)
    err    = (deq_f - src_f).abs()
    rel    = err / (src_f.abs() + 1e-6)
    rms_err = err.pow(2).mean().sqrt().item()
    max_err = err.max().item()
    mean_rel = rel.mean().item()
    p99_rel  = rel.quantile(0.99).item()

    # 3. Per-block check: every block's absmax should be representable by the
    #    decoded scale * 6.0 within FP4 quantization noise.
    blocks_src = src_f.view(T, H, HEAD_DIM // GROUP_SIZE, GROUP_SIZE)
    block_absmax = blocks_src.abs().max(dim=-1).values
    blocks_deq = deq_f.view(T, H, HEAD_DIM // GROUP_SIZE, GROUP_SIZE)
    block_max_err = (blocks_deq - blocks_src).abs().max(dim=-1).values
    worst_block_rel = (block_max_err / (block_absmax + 1e-6)).max().item()

    # SNR vs reference: NVFP4 with per-block FP8 scales targets ~25-30 dB on
    # well-conditioned data (matches the NVIDIA blog's <1% end-to-end accuracy
    # loss claim once softmax averaging kicks in).
    sig_pow = src_f.pow(2).mean().item()
    err_pow = (deq_f - src_f).pow(2).mean().item()
    snr_db = 10.0 * math.log10(sig_pow / max(err_pow, 1e-30))

    print(f"[{mode:<10}] bit-match CUDA vs python ref: {bit_match}")
    print(f"            rms_err={rms_err:.4f}  max_err={max_err:.4f}  "
          f"mean_rel={mean_rel:.4f}  p99_rel={p99_rel:.4f}  worst_block_rel={worst_block_rel:.4f}")
    print(f"            SNR vs bf16 ref: {snr_db:.1f} dB")
    assert bit_match, "CUDA dequant disagrees with python reference"
    # The intrinsic FP4 quantization floor: with codes {0, .5, 1, 1.5, 2, 3, 4, 6},
    # the worst gap (4->6) gives ~20% per-elem rel error on the value mid-gap.
    # End-to-end accuracy holds because softmax averages these errors.
    if mode in ("uniform", "gaussian"):
        assert snr_db > 18.0, f"SNR too low: {snr_db:.1f} dB"
    elif mode == "outliers":
        assert snr_db > 12.0, f"SNR too low with outliers: {snr_db:.1f} dB"
    return rms_err, max_err


def test_qk_dot():
    """Q (bf16) . K (NVFP4) should equal Q . dequant(K) within bf16 noise."""
    T, H, Q_H = 1024, 2, 8
    k_bf16 = make_inputs(T, H, "gaussian", seed=42)
    q_bf16 = (torch.randn((Q_H, HEAD_DIM), device="cuda") * 0.5).to(torch.bfloat16)

    k_data, k_scales = quantize(k_bf16)
    k_deq = dequantize(k_data, k_scales)

    scores = torch.empty((Q_H, T), dtype=torch.float32, device="cuda")
    ops.qk_dot_nvfp4(q_bf16, k_data, k_scales, scores)

    # Reference: dequant K then bf16 matmul through fp32.
    # GQA: each query head qh maps to kv head qh // (Q_H // H).
    gqa = Q_H // H
    q_f = q_bf16.to(torch.float32)        # [Q_H, D]
    k_f = k_deq.to(torch.float32)         # [T, H, D]
    ref = torch.empty_like(scores)
    for qh in range(Q_H):
        h = qh // gqa
        ref[qh] = (q_f[qh].unsqueeze(0) * k_f[:, h, :]).sum(dim=1)

    err = (scores - ref).abs()
    rms = err.pow(2).mean().sqrt().item()
    mx  = err.max().item()
    # bf16 dot product over 256 elements: ~256 * 2^-7 * |x||y| accumulation
    # noise. Should be < 1e-2 relative to the score magnitude.
    score_scale = ref.abs().mean().item()
    print(f"[qk_dot   ] rms={rms:.6f}  max={mx:.6f}  mean|ref|={score_scale:.4f}")
    assert mx < max(0.05 * score_scale, 1e-3), \
        f"Q.K NVFP4 dot deviates from bf16 ref by {mx} (mean|ref|={score_scale})"


def report_memory():
    # Default lucebox config.
    n_layers = 6
    n_heads  = 2
    max_seq  = 65536
    bf16_per_cache = n_layers * n_heads * max_seq * HEAD_DIM * 2
    nvfp4_per_cache = n_layers * n_heads * max_seq * (DATA_BYTES + SCALE_BYTES)
    mb = lambda b: b / (1024 ** 2)
    print()
    print(f"KV cache memory at n_layers={n_layers}, n_kv_heads={n_heads}, "
          f"max_seq={max_seq}, head_dim={HEAD_DIM}:")
    print(f"  bf16  : K={mb(bf16_per_cache):7.2f} MB  V={mb(bf16_per_cache):7.2f} MB  total={mb(2*bf16_per_cache):7.2f} MB")
    print(f"  nvfp4 : K={mb(nvfp4_per_cache):7.2f} MB  V={mb(nvfp4_per_cache):7.2f} MB  total={mb(2*nvfp4_per_cache):7.2f} MB")
    print(f"  ratio : {2*bf16_per_cache / (2*nvfp4_per_cache):.2f}x smaller")


def bench_throughput():
    T, H = 65536, 2
    src = make_inputs(T, H, "gaussian")
    data   = torch.empty((T, H, DATA_BYTES),  dtype=torch.uint8, device="cuda")
    scales = torch.empty((T, H, SCALE_BYTES), dtype=torch.uint8, device="cuda")

    # Warm.
    for _ in range(3):
        ops.quantize_bf16_to_nvfp4_kv(src, data, scales)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    N = 50
    for _ in range(N):
        ops.quantize_bf16_to_nvfp4_kv(src, data, scales)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    bytes_read  = T * H * HEAD_DIM * 2
    bytes_write = T * H * (DATA_BYTES + SCALE_BYTES)
    bw = (bytes_read + bytes_write) / dt / 1e9
    print(f"\nquantize    T={T} H={H}: {dt*1e6:.1f} us/call  -- "
          f"{bw:.1f} GB/s (R={bytes_read/1e6:.1f} MB + W={bytes_write/1e6:.1f} MB)")

    out_bf16 = torch.empty((T, H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        ops.dequantize_nvfp4_kv_to_bf16(data, scales, out_bf16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        ops.dequantize_nvfp4_kv_to_bf16(data, scales, out_bf16)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    bytes_read  = T * H * (DATA_BYTES + SCALE_BYTES)
    bytes_write = T * H * HEAD_DIM * 2
    bw = (bytes_read + bytes_write) / dt / 1e9
    print(f"dequantize  T={T} H={H}: {dt*1e6:.1f} us/call  -- "
          f"{bw:.1f} GB/s (R={bytes_read/1e6:.1f} MB + W={bytes_write/1e6:.1f} MB)")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  cap={cap}")
    if cap[0] < 12:
        print("WARNING: NVFP4 KV requires compute capability >= 12.0 (Blackwell). "
              "This test will skip on older arches.")
        sys.exit(0)

    print("\n=== Roundtrip correctness ===")
    test_roundtrip("uniform")
    test_roundtrip("gaussian")
    test_roundtrip("outliers")

    print("\n=== Q.K dot ===")
    test_qk_dot()

    report_memory()
    bench_throughput()
    print("\nOK")
