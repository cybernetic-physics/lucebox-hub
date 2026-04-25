"""Sweep H (head count) at fixed S=512 to expose parallelism behavior.

The kernel launches one block per head. B200 has 132 SMs — H=16 (the
actual Qwen3.5-0.8B config) uses only ~12% of them. Larger H should
keep per-head ms approximately flat until H>132 (then SMs start
serializing two blocks).
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401


Dk, Dv = 128, 128
S = 512


def run(H: int):
    dev = "cuda"
    torch.manual_seed(0)
    q = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    k = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    v = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    beta  = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
    decay = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
    state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()
    dy = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    y_cuda = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
    state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
    delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
    state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)
    dq = torch.empty_like(q); dk_ = torch.empty_like(k); dv_ = torch.empty_like(v)
    dbeta = torch.empty_like(beta); ddecay = torch.empty_like(decay)
    dstate_init = torch.empty_like(state_init)

    def step():
        torch.ops.train_megakernel_C.dn_fwd_save(
            q, k, v, beta, decay, state_init,
            y_cuda, state_out, delta_save, state_history)
        torch.ops.train_megakernel_C.dn_bwd(
            q, k, v, beta, decay, state_init, delta_save, dy, state_history,
            dq, dk_, dv_, dbeta, ddecay, dstate_init)

    for _ in range(3): step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5): step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / 5


def main():
    print(f"S = {S}, Dk = {Dk}, Dv = {Dv} — fwd+bwd ms vs head count")
    print(f"{'H':>5} | {'ms/iter':>9} | {'us/head/step':>14} | notes")
    print("-" * 60)
    for H in (1, 2, 4, 8, 16, 32, 64, 128, 132, 256):
        try:
            ms = run(H)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"{H:>5} | error: {str(e)[:40]}")
            continue
        us_per_head_step = ms * 1000.0 / (H * S)
        note = ""
        if H <= 132:
            note = f"H/132 SMs = {100*H/132:.0f}%"
        else:
            note = f"oversubscribed ({H/132:.1f}x SMs)"
        print(f"{H:>5} | {ms:>9.2f} | {us_per_head_step:>14.3f} | {note}")


if __name__ == "__main__":
    main()
