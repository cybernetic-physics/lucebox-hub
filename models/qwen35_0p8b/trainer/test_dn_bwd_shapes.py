"""Wide shape-sweep test for the dn_fwd_save / dn_bwd CUDA kernels.

  - Correctness vs torch autograd reference at S ≤ 1024 (ref is slow above).
  - Bench-only (and NaN check) at S = 2048..32768.

Reports per-S:
  - max|Δ| and cos against autograd ref (where applicable)
  - kernel ms/iter for fwd+save and bwd
  - peak GB of allocated state_history
"""
from __future__ import annotations

import gc
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401
from dn_bwd import dn_forward_recurrence, dn_backward_autograd


H, Dk, Dv = 16, 128, 128


def gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def cos(a, b):
    af, bf = a.to(torch.float32).flatten(), b.to(torch.float32).flatten()
    return F.cosine_similarity(af, bf, dim=0).item()


def run_one(S: int, do_correctness: bool):
    dev = "cuda"
    torch.manual_seed(0)
    q = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    k = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    v = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    beta  = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
    decay = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
    state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()
    dy = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()

    y_cuda    = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
    state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
    delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
    state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)
    sh_bytes = state_history.numel() * 4

    dq = torch.empty_like(q); dk_ = torch.empty_like(k); dv_ = torch.empty_like(v)
    dbeta = torch.empty_like(beta); ddecay = torch.empty_like(decay)
    dstate_init = torch.empty_like(state_init)

    def run_fwd():
        torch.ops.train_megakernel_C.dn_fwd_save(
            q, k, v, beta, decay, state_init,
            y_cuda, state_out, delta_save, state_history)

    def run_bwd():
        torch.ops.train_megakernel_C.dn_bwd(
            q, k, v, beta, decay, state_init, delta_save, dy, state_history,
            dq, dk_, dv_, dbeta, ddecay, dstate_init)

    # Single-call timing (one fwd, one bwd separately, plus a combined).
    for _ in range(2): run_fwd()
    for _ in range(2): run_bwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3): run_fwd()
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0 / 3
    t0 = time.perf_counter()
    for _ in range(3): run_bwd()
    torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - t0) * 1000.0 / 3

    # Sanity: no NaN/Inf in any output.
    bad = []
    for name, t in [("y", y_cuda), ("state_out", state_out), ("delta_save", delta_save),
                    ("dq", dq), ("dk", dk_), ("dv", dv_),
                    ("dbeta", dbeta), ("ddecay", ddecay), ("dstate_init", dstate_init)]:
        if not torch.isfinite(t).all():
            bad.append(name)

    summary = {
        "S": S,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "sh_bytes": sh_bytes,
        "bad": bad,
        "cos_y":   None,
        "cos_dq":  None,
        "cos_dk":  None,
        "cos_dv":  None,
        "cos_db":  None,
        "cos_dd":  None,
        "cos_ds":  None,
    }

    if do_correctness:
        # Correctness vs torch reference.
        y_ref, _ = dn_forward_recurrence(q, k, v, beta, decay, state_init)
        ref = dn_backward_autograd(q, k, v, beta, decay, state_init, dy)
        summary["cos_y"]  = cos(y_cuda, y_ref)
        summary["cos_dq"] = cos(dq,    ref.dq)
        summary["cos_dk"] = cos(dk_,   ref.dk)
        summary["cos_dv"] = cos(dv_,   ref.dv)
        summary["cos_db"] = cos(dbeta,  ref.dbeta)
        summary["cos_dd"] = cos(ddecay, ref.ddecay)
        summary["cos_ds"] = cos(dstate_init, ref.dstate_init)

    # Free.
    del q, k, v, beta, decay, state_init, dy, y_cuda, state_out, delta_save, state_history
    del dq, dk_, dv_, dbeta, ddecay, dstate_init
    if do_correctness:
        del y_ref, ref
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    seq_lens = [32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    correctness_lens = {32, 128, 512, 1024}

    print(f"{'S':>6} | {'fwd ms':>8} | {'bwd ms':>9} | {'fwd+bwd ms':>12} | {'state_hist':>10} | "
          f"{'cos y/dq/dk/dv/dβ/dγ/ds0 or NaN check':<60}")
    print("-" * 130)
    for S in seq_lens:
        try:
            r = run_one(S, S in correctness_lens)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"{S:>6} | OOM/error: {str(e)[:80]}")
            continue
        if r["cos_y"] is not None:
            cs = (f"y={r['cos_y']:.5f} dq={r['cos_dq']:.5f} dk={r['cos_dk']:.5f} "
                  f"dv={r['cos_dv']:.5f} dβ={r['cos_db']:.5f} dγ={r['cos_dd']:.5f} "
                  f"ds0={r['cos_ds']:.5f}")
        else:
            cs = "no NaN/Inf" if not r["bad"] else f"NaN in: {r['bad']}"
        print(f"{r['S']:>6} | {r['fwd_ms']:>8.2f} | {r['bwd_ms']:>9.2f} | "
              f"{r['fwd_ms']+r['bwd_ms']:>12.2f} | {gb(r['sh_bytes']):>10} | {cs}")


if __name__ == "__main__":
    main()
