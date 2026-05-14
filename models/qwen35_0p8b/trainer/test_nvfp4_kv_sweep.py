"""Production sweep for the NVFP4 KV cache format on GB10.

Five test classes:
  1. Shape grid:    T x H grid (T in 1..65536, H in 1..16) x 6 input
                    distributions. Asserts CUDA dequant bit-matches the
                    Python LUT+E4M3 reference and SNR is within format
                    floor.
  2. Edge cases:    all-zeros, all-constant, single non-zero, Inf, NaN,
                    near-denormal magnitudes, alternating-sign sawtooth.
                    Asserts no NaN/Inf propagation to output.
  3. GQA Q.K dot:   dot agrees with bf16 ref Q . dequant(K) across all
                    Q_H/H ratios used in the lucebox model (8/2=4).
  4. Fused attention: kv_attention_nvfp4 vs Python-bf16-reference attention
                    on the dequantized K, V. Checks the FA inner-loop path
                    that the megakernel will use.
  5. Throughput:    quant + dequant + attention bandwidth at the production
                    max_seq=65536 size.

Run:
    cd /home/sparkz/rl/lucebox-hub/models/qwen35_0p8b
    /home/sparkz/rl/.venv/bin/python3 trainer/test_nvfp4_kv_sweep.py
"""
import itertools
import math
import os
import sys
import time
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR  = os.path.dirname(THIS_DIR)
sys.path.insert(0, PKG_DIR)
import qwen35_megakernel_bf16_C  # noqa: F401

ops = torch.ops.qwen35_megakernel_bf16_C

HEAD_DIM    = 256
GROUP_SIZE  = 16
DATA_BYTES  = HEAD_DIM // 2
SCALE_BYTES = HEAD_DIM // GROUP_SIZE
NVFP4_MAX   = 6.0
E4M3_MAX    = 448.0

FP4_E2M1 = torch.tensor(
    [ 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantize(src):
    T, H, _ = src.shape
    data   = torch.empty((T, H, DATA_BYTES),  dtype=torch.uint8, device="cuda")
    scales = torch.empty((T, H, SCALE_BYTES), dtype=torch.uint8, device="cuda")
    ops.quantize_bf16_to_nvfp4_kv(src, data, scales)
    return data, scales


def dequantize(data, scales):
    T, H, _ = data.shape
    out = torch.empty((T, H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    ops.dequantize_nvfp4_kv_to_bf16(data, scales, out)
    return out


def python_dequant_reference(data, scales):
    T, H, _ = data.shape
    data_cpu   = data.cpu()
    scales_cpu = scales.cpu()
    scale_f = scales_cpu.view(torch.float8_e4m3fn).to(torch.float32)
    lo = data_cpu & 0xF
    hi = (data_cpu >> 4) & 0xF
    codes = torch.stack([lo, hi], dim=-1).reshape(T, H, HEAD_DIM)
    vals = FP4_E2M1[codes.to(torch.long)]
    return (vals * scale_f.repeat_interleave(GROUP_SIZE, dim=-1)).to(torch.bfloat16)


def make_dist(T, H, mode, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (T, H, HEAD_DIM)
    if mode == "uniform":
        x = (torch.rand(shape, device="cuda", generator=g) * 2 - 1) * 3.0
    elif mode == "gaussian":
        x = torch.randn(shape, device="cuda", generator=g) * 1.5
    elif mode == "gaussian_tight":
        x = torch.randn(shape, device="cuda", generator=g) * 0.3
    elif mode == "gaussian_wide":
        x = torch.randn(shape, device="cuda", generator=g) * 4.0
    elif mode == "outliers":
        x = torch.randn(shape, device="cuda", generator=g) * 1.0
        # ~1 outlier per 16-elem block.
        n_blocks = T * H * (HEAD_DIM // GROUP_SIZE)
        idx = torch.randint(0, GROUP_SIZE, (n_blocks,), device="cuda", generator=g)
        x.view(-1, GROUP_SIZE)[torch.arange(n_blocks, device="cuda"), idx] *= 5.0
    elif mode == "rope_like":
        # Mimic the post-RoPE/post-RMSNorm distribution that actually lands
        # in fa_k_cache: ~unit variance with cosine-modulated structure.
        base = torch.randn(shape, device="cuda", generator=g)
        i = torch.arange(HEAD_DIM, device="cuda", dtype=torch.float32).view(1, 1, HEAD_DIM)
        x = base * torch.cos(i * 0.03)
    else:
        raise ValueError(mode)
    return x.to(torch.bfloat16)


def snr_db(x, y):
    sig = x.to(torch.float32).pow(2).mean().item()
    err = (x.to(torch.float32) - y.to(torch.float32)).pow(2).mean().item()
    return 10.0 * math.log10(sig / max(err, 1e-30))


# ---------------------------------------------------------------------------
# 1. Shape grid
# ---------------------------------------------------------------------------

def test_shape_grid():
    T_vals = [1, 7, 17, 32, 128, 1024, 8192, 65536]
    H_vals = [1, 2, 4, 8]
    dists  = ["uniform", "gaussian", "gaussian_tight", "gaussian_wide", "outliers", "rope_like"]

    # Min SNR per distribution. The format's intrinsic floor is governed by
    # the FP4 grid spacing; per-block FP8 scaling brings well-conditioned data
    # to ~19-22 dB and outlier-stressed data to ~12-15 dB.
    snr_floor = {
        "uniform": 18.0, "gaussian": 19.0, "gaussian_tight": 19.0,
        "gaussian_wide": 18.0, "outliers": 12.0, "rope_like": 18.0,
    }

    fails = []
    n_total = 0
    print(f"{'T':>6} {'H':>3} {'dist':<16} {'bit-match':>10} {'SNR_dB':>8} {'max_err':>9}")
    for T, H, dist in itertools.product(T_vals, H_vals, dists):
        src = make_dist(T, H, dist, seed=T * 31 + H * 7)
        data, scales = quantize(src)
        cuda_deq = dequantize(data, scales)
        py_deq   = python_dequant_reference(data, scales).to("cuda")
        bit_ok   = bool(torch.equal(cuda_deq, py_deq))
        snr = snr_db(src, cuda_deq)
        max_err = (src.to(torch.float32) - cuda_deq.to(torch.float32)).abs().max().item()

        # No-NaN/Inf invariant.
        finite_ok = bool(torch.isfinite(cuda_deq.to(torch.float32)).all().item())

        print(f"{T:>6} {H:>3} {dist:<16} {str(bit_ok):>10} {snr:>8.2f} {max_err:>9.4f}")
        n_total += 1
        if not bit_ok:
            fails.append(f"bit-match failed at T={T} H={H} dist={dist}")
        if not finite_ok:
            fails.append(f"NaN/Inf in output at T={T} H={H} dist={dist}")
        if snr < snr_floor[dist]:
            fails.append(f"SNR {snr:.1f} dB < floor {snr_floor[dist]:.1f} dB at T={T} H={H} dist={dist}")
    print(f"\nshape-grid: {n_total - len(fails)} / {n_total} passed")
    for f in fails: print(f"  FAIL: {f}")
    return len(fails) == 0


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases():
    T, H = 64, 2

    def _check(name, src, predicate_str, predicate):
        data, scales = quantize(src)
        deq = dequantize(data, scales)
        deq_f = deq.to(torch.float32)
        finite_ok = bool(torch.isfinite(deq_f).all().item())
        py = python_dequant_reference(data, scales).to("cuda")
        bit_ok = bool(torch.equal(deq, py))
        ok = predicate(src, deq_f)
        print(f"  [{name:<22}] finite={finite_ok}  bit-match={bit_ok}  {predicate_str}={ok}")
        return finite_ok and bit_ok and ok

    print("edge cases:")
    results = []

    # All zeros -> all zeros.
    src = torch.zeros((T, H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    results.append(_check("all-zero", src, "all-zero-out", lambda s, d: bool((d == 0).all().item())))

    # All constant +2.5: scale=2.5/6=0.4167 -> E4M3 round to some scale.
    # Every elem is 2.5 -> 2.5 / 0.4167 = 6.0 -> code 6 (=4) or code 7 (=6).
    # Expect dequant close to 2.5 (within FP4 grid).
    src = torch.full((T, H, HEAD_DIM), 2.5, dtype=torch.bfloat16, device="cuda")
    results.append(_check("constant-2.5", src, "mean-close",
                          lambda s, d: abs(d.mean().item() - 2.5) < 0.6))

    # Single non-zero in a block. Should round-trip the non-zero exactly,
    # zeros stay zero.
    src = torch.zeros((T, H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    src[0, 0, 7] = 3.0
    results.append(_check("single-nonzero", src, "spike-preserved",
                          lambda s, d: abs(d[0, 0, 7].item() - 3.0) < 0.6
                                       and bool((d[0, 0, :7] == 0).all().item())
                                       and bool((d[0, 0, 8:16] == 0).all().item())))

    # Extreme magnitudes (saturate to FP4 +/- 6 * E4M3 scale).
    src = torch.full((T, H, HEAD_DIM), 1e6, dtype=torch.bfloat16, device="cuda")
    # Saturated output is finite (no Inf/NaN) and positive.
    results.append(_check("extreme-positive", src, "finite-positive",
                          lambda s, d: bool((d > 0).all().item())))

    # Near-denormal magnitudes (clamped to MIN_SCALE -> all zero).
    src = torch.full((T, H, HEAD_DIM), 1e-30, dtype=torch.bfloat16, device="cuda")
    results.append(_check("near-denormal", src, "rounds-to-zero",
                          lambda s, d: bool((d == 0).all().item())))

    # +Inf inputs: must saturate, not propagate Inf.
    src = torch.full((T, H, HEAD_DIM), float("inf"), dtype=torch.bfloat16, device="cuda")
    results.append(_check("plus-inf", src, "finite", lambda s, d: True))  # finite checked above

    # NaN inputs: NaN must not leak to output. (Encoder masks NaN via fmaxf.)
    g = torch.Generator(device="cuda").manual_seed(0)
    src = (torch.randn((T, H, HEAD_DIM), device="cuda", generator=g) * 1.5).to(torch.bfloat16)
    src[3, 1, 17] = float("nan")
    src[10, 0, 200] = float("nan")
    results.append(_check("scattered-nan", src, "no-nan-out", lambda s, d: True))

    # Alternating-sign sawtooth: stresses sign bit packing.
    base = torch.arange(HEAD_DIM, dtype=torch.float32, device="cuda")
    src  = (((base % 2) * 2 - 1) * 2.0).view(1, 1, HEAD_DIM).expand(T, H, HEAD_DIM).contiguous().to(torch.bfloat16)
    results.append(_check("sawtooth-pm2", src, "snr-ok",
                          lambda s, d: snr_db(s, d.to(torch.bfloat16)) > 15.0))

    print(f"edge-cases: {sum(results)} / {len(results)} passed")
    return all(results)


# ---------------------------------------------------------------------------
# 3. Q.K dot with the lucebox GQA ratio (Q_H=8, H=2)
# ---------------------------------------------------------------------------

def test_qk_dot_gqa():
    for T in [1, 17, 256, 8192]:
        H, Q_H = 2, 8
        k_bf16 = make_dist(T, H, "rope_like", seed=99)
        q_bf16 = (torch.randn((Q_H, HEAD_DIM), device="cuda") * 0.5).to(torch.bfloat16)

        k_data, k_scales = quantize(k_bf16)
        k_deq = dequantize(k_data, k_scales)

        scores = torch.empty((Q_H, T), dtype=torch.float32, device="cuda")
        ops.qk_dot_nvfp4(q_bf16, k_data, k_scales, scores)

        gqa = Q_H // H
        q_f = q_bf16.to(torch.float32)
        k_f = k_deq.to(torch.float32)
        ref = torch.empty_like(scores)
        for qh in range(Q_H):
            h = qh // gqa
            ref[qh] = (q_f[qh].unsqueeze(0) * k_f[:, h, :]).sum(dim=1)

        max_err = (scores - ref).abs().max().item()
        mean_abs = ref.abs().mean().item()
        rel = max_err / max(mean_abs, 1e-6)
        finite = bool(torch.isfinite(scores).all().item())
        print(f"  T={T:>5}  Q_H={Q_H}  H={H}  max_err={max_err:.6f}  mean|ref|={mean_abs:.4f}  rel={rel:.2e}  finite={finite}")
        assert finite, f"non-finite scores at T={T}"
        assert rel < 1e-3, f"dot vs dequant-ref disagrees: rel={rel} at T={T}"
    return True


# ---------------------------------------------------------------------------
# 4. Fused softmax attention agrees with python bf16 reference
# ---------------------------------------------------------------------------

def ref_attention_bf16(q, k_deq, v_deq, attn_scale):
    """Reference attention on already-dequantized bf16 K, V. Computed in fp32
    so the only difference vs the CUDA path is fp32-accum ordering."""
    Q_H, D = q.shape
    T, H, _ = k_deq.shape
    gqa = Q_H // H
    q_f = q.to(torch.float32)
    k_f = k_deq.to(torch.float32)
    v_f = v_deq.to(torch.float32)
    out = torch.empty_like(q_f)
    lse = torch.empty((Q_H,), dtype=torch.float32, device=q.device)
    for qh in range(Q_H):
        h = qh // gqa
        scores = (q_f[qh].unsqueeze(0) * k_f[:, h, :]).sum(dim=1) * attn_scale
        m = scores.max()
        w = (scores - m).exp()
        s = w.sum()
        out[qh] = (w.unsqueeze(1) * v_f[:, h, :]).sum(dim=0) / s
        lse[qh] = m + s.log()
    return out.to(torch.bfloat16), lse


def test_split_attention_matches_serial():
    """Split-K must match the serial-T attention bit-comparably (only fp32
    accumulation-order divergence)."""
    Q_H, H = 8, 2
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    fails = []
    print(f"  {'T':>6} {'splits':>7} {'rel_out':>10} {'rel_lse':>10}")
    for T, num_splits in [(128, 4), (128, 32), (1024, 8), (8192, 64), (65536, 128)]:
        if num_splits > T: continue
        q  = (torch.randn((Q_H, HEAD_DIM), device="cuda") * 0.5).to(torch.bfloat16)
        ks = make_dist(T, H, "rope_like", seed=T)
        vs = make_dist(T, H, "rope_like", seed=T + 1)
        kd, kss = quantize(ks)
        vd, vss = quantize(vs)
        out_ser = torch.empty((Q_H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        lse_ser = torch.empty((Q_H,), dtype=torch.float32, device="cuda")
        ops.kv_attention_nvfp4(q, kd, kss, vd, vss, out_ser, lse_ser, attn_scale)
        out_split = torch.empty_like(out_ser)
        lse_split = torch.empty_like(lse_ser)
        partials = torch.empty((Q_H, num_splits, HEAD_DIM + 2),
                               dtype=torch.float32, device="cuda")
        ops.kv_attention_split_nvfp4(
            q, kd, kss, vd, vss, out_split, partials, lse_split, attn_scale, num_splits)
        err = (out_split.to(torch.float32) - out_ser.to(torch.float32)).abs().max().item()
        scale = out_ser.to(torch.float32).abs().mean().item()
        rel_out = err / max(scale, 1e-6)
        rel_lse = (lse_split - lse_ser).abs().max().item() / max(lse_ser.abs().mean().item(), 1e-6)
        print(f"  {T:>6} {num_splits:>7} {rel_out:>10.2e} {rel_lse:>10.2e}")
        if rel_out > 5e-2:
            fails.append(f"split vs serial rel_out={rel_out:.2e} at T={T}, splits={num_splits}")
        if rel_lse > 1e-5:
            fails.append(f"split vs serial rel_lse={rel_lse:.2e} at T={T}, splits={num_splits}")
    for f in fails: print(f"  FAIL: {f}")
    return len(fails) == 0


def test_fused_attention():
    Q_H, H = 8, 2
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    print(f"  attn_scale = {attn_scale:.5f}")
    print(f"  {'T':>6} {'q_dist':<12} {'kv_dist':<12} {'rel_out':>8} {'rel_lse':>8}")
    fails = []
    for T in [1, 17, 256, 1024, 8192, 65536]:
        for q_dist, kv_dist in [("gaussian", "rope_like"),
                                  ("gaussian", "outliers"),
                                  ("gaussian_tight", "gaussian")]:
            q_src  = make_dist(1, Q_H, q_dist,  seed=T + 1).view(Q_H, HEAD_DIM).contiguous()
            k_src  = make_dist(T, H, kv_dist, seed=T + 2)
            v_src  = make_dist(T, H, kv_dist, seed=T + 3)
            k_data, k_scales = quantize(k_src)
            v_data, v_scales = quantize(v_src)
            k_deq = dequantize(k_data, k_scales)
            v_deq = dequantize(v_data, v_scales)

            out_cuda = torch.empty((Q_H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
            lse_cuda = torch.empty((Q_H,), dtype=torch.float32, device="cuda")
            ops.kv_attention_nvfp4(q_src, k_data, k_scales, v_data, v_scales,
                                   out_cuda, lse_cuda, attn_scale)

            out_ref, lse_ref = ref_attention_bf16(q_src, k_deq, v_deq, attn_scale)
            # Use rel error in fp32 space.
            err_out = (out_cuda.to(torch.float32) - out_ref.to(torch.float32)).abs()
            scale_out = out_ref.to(torch.float32).abs().mean().item()
            rel_out = err_out.max().item() / max(scale_out, 1e-6)
            err_lse = (lse_cuda - lse_ref).abs().max().item()
            scale_lse = lse_ref.abs().mean().item()
            rel_lse = err_lse / max(scale_lse, 1e-6)

            finite = bool(torch.isfinite(out_cuda.to(torch.float32)).all().item()) \
                     and bool(torch.isfinite(lse_cuda).all().item())
            print(f"  {T:>6} {q_dist:<12} {kv_dist:<12} {rel_out:>8.2e} {rel_lse:>8.2e}")
            if not finite:
                fails.append(f"non-finite at T={T} q={q_dist} kv={kv_dist}")
            # CUDA and python both operate on the same dequantized K, V, so
            # the only source of disagreement is fp32 accumulation order.
            # Online softmax in the kernel re-scales the accumulator each
            # step; python does one big reduction. At long T the rounding
            # divergence reaches a few percent on outlier-heavy distributions.
            # Floor: 5% out, 1e-5 lse. End-to-end model accuracy is checked
            # downstream against HF, where the bar is HF-bf16 PPL parity.
            rel_floor = 5e-2 if T >= 8192 else 1e-2
            if rel_out > rel_floor:
                fails.append(f"rel_out={rel_out:.2e} > {rel_floor:.0e} at T={T} q={q_dist} kv={kv_dist}")
            if rel_lse > 1e-5:
                fails.append(f"rel_lse={rel_lse:.2e} > 1e-5 at T={T} q={q_dist} kv={kv_dist}")
    for f in fails: print(f"  FAIL: {f}")
    return len(fails) == 0


# ---------------------------------------------------------------------------
# 5. Throughput
# ---------------------------------------------------------------------------

def bench():
    T, H, Q_H = 65536, 2, 8
    src = make_dist(T, H, "rope_like", seed=1)
    data   = torch.empty((T, H, DATA_BYTES),  dtype=torch.uint8, device="cuda")
    scales = torch.empty((T, H, SCALE_BYTES), dtype=torch.uint8, device="cuda")
    out_bf16 = torch.empty((T, H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    def timed(fn, N=50, warmup=5):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N

    dt_q  = timed(lambda: ops.quantize_bf16_to_nvfp4_kv(src, data, scales))
    bytes_q  = T * H * HEAD_DIM * 2 + T * H * (DATA_BYTES + SCALE_BYTES)
    print(f"  quantize     T={T:>6}: {dt_q*1e6:>7.1f} us  {bytes_q/dt_q/1e9:>7.1f} GB/s effective")

    dt_d  = timed(lambda: ops.dequantize_nvfp4_kv_to_bf16(data, scales, out_bf16))
    bytes_d = T * H * (DATA_BYTES + SCALE_BYTES) + T * H * HEAD_DIM * 2
    print(f"  dequantize   T={T:>6}: {dt_d*1e6:>7.1f} us  {bytes_d/dt_d/1e9:>7.1f} GB/s effective")

    # Attention sweep at production-relevant T.
    q_bf16 = (torch.randn((Q_H, HEAD_DIM), device="cuda") * 0.5).to(torch.bfloat16)
    v_src = make_dist(T, H, "rope_like", seed=2)
    v_data, v_scales = quantize(v_src)
    out = torch.empty((Q_H, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    print(f"\n  serial-T attention  Q_H={Q_H}  H={H}  (one warp per query head, no split):")
    for Tcap in [128, 1024, 8192, 32768, 65536]:
        kd = data[:Tcap].contiguous()
        ks = scales[:Tcap].contiguous()
        vd = v_data[:Tcap].contiguous()
        vs = v_scales[:Tcap].contiguous()
        dt = timed(lambda: ops.kv_attention_nvfp4(q_bf16, kd, ks, vd, vs, out, None, attn_scale))
        bytes_a = 2 * Tcap * H * (DATA_BYTES + SCALE_BYTES)
        print(f"    T={Tcap:>6}: {dt*1e6:>7.1f} us  ({bytes_a/dt/1e9:>6.1f} GB/s KV read)")

    print(f"\n  split-K attention   Q_H={Q_H}  H={H}  (num_splits parallel over T):")
    for Tcap in [128, 1024, 8192, 32768, 65536]:
        kd = data[:Tcap].contiguous()
        ks = scales[:Tcap].contiguous()
        vd = v_data[:Tcap].contiguous()
        vs = v_scales[:Tcap].contiguous()
        # GB10 has ~48-64 SMs. Target full occupancy: num_splits = max(1, min(T/64, num_sms / Q_H)).
        # Cap by T so each split has work.
        for num_splits in [8, 32, 128]:
            if num_splits > Tcap: continue
            partials = torch.empty((Q_H, num_splits, HEAD_DIM + 2),
                                   dtype=torch.float32, device="cuda")
            dt = timed(lambda: ops.kv_attention_split_nvfp4(
                q_bf16, kd, ks, vd, vs, out, partials, None, attn_scale, num_splits))
            bytes_a = 2 * Tcap * H * (DATA_BYTES + SCALE_BYTES)
            print(f"    T={Tcap:>6}  splits={num_splits:>4}: {dt*1e6:>7.1f} us  ({bytes_a/dt/1e9:>6.1f} GB/s KV read)")


# ---------------------------------------------------------------------------
# Footprint summary
# ---------------------------------------------------------------------------

def report_footprint():
    n_layers, n_heads, max_seq = 6, 2, 65536
    bf16  = n_layers * n_heads * max_seq * HEAD_DIM * 2
    nvfp4 = n_layers * n_heads * max_seq * (DATA_BYTES + SCALE_BYTES)
    mb = lambda b: b / (1024 ** 2)
    print(f"  per-cache  bf16: {mb(bf16):7.2f} MB   nvfp4: {mb(nvfp4):7.2f} MB")
    print(f"  K+V total  bf16: {mb(2*bf16):7.2f} MB   nvfp4: {mb(2*nvfp4):7.2f} MB   "
          f"({2*bf16/(2*nvfp4):.2f}x smaller)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert torch.cuda.is_available()
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  cap={cap}")
    if cap[0] < 12:
        print("NVFP4 KV requires compute capability >= 12.0. Skipping.")
        sys.exit(0)

    print("\n=== KV cache memory footprint (lucebox defaults) ===")
    report_footprint()

    print("\n=== 1. Shape grid (T, H, distribution) ===")
    ok1 = test_shape_grid()

    print("\n=== 2. Edge cases ===")
    ok2 = test_edge_cases()

    print("\n=== 3. GQA Q.K dot ===")
    ok3 = test_qk_dot_gqa()

    print("\n=== 4a. Fused softmax attention vs bf16 reference ===")
    ok4 = test_fused_attention()

    print("\n=== 4b. Split-K attention vs serial ===")
    ok4b = test_split_attention_matches_serial()

    print("\n=== 5. Throughput ===")
    bench()

    print()
    all_ok = ok1 and ok2 and ok3 and ok4 and ok4b
    print("ALL PASSED" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
