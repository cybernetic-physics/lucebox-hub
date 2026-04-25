# DN kernel performance — where the cycles go and how to claw them back

Snapshot of where we are and what each optimization buys, by shape regime.

> **Honest baseline note**: every speedup vs "HF" anywhere in this repo
> has two flavors. The HF *fp32 torch fallback* (which we hit when
> fla isn't installed) is much slower than HF + fla (production). Numbers
> below distinguish the two.

## Where the time goes today

Three reference points (all per-layer, isolated kernel calls):

| kernel | S=128 | S=512 | S=2048 | scaling |
|---|---|---|---|---|
| recurrent bwd (ours, scalar fp32) | 3.1 ms | 12.7 ms | 51 ms | linear in S |
| chunked fwd (ours, bf16 WMMA) | 0.27 ms | 1.0 ms | 3.6 ms | linear in S |
| **fla chunk fwd (Triton, production)** | **0.48 ms** | **0.45 ms** | **0.47 ms** | **~constant in S** |

The architectural reason fla wins at long S: it parallelizes *chunks*
across SMs (one block per chunk per head). For S=2048 with H=16, that's
S/64 × H = 32 × 16 = 512 logical work units, so they saturate SMs even
at this S. Our kernel parallelizes *heads* (one block per head, all
chunks done sequentially in that block) → 16 SMs busy, the rest idle,
and the per-chunk cost stacks linearly in S.

Per-step in the recurrent kernel is **31.5 µs constant** at H=16 — and stays
~31 µs at H=132 (full SM occupancy). So we're **latency-bound, not
compute-bound**. The block does very little arithmetic per step (~640 fp32
ops/thread = nanoseconds of actual math) but is gated on:

- ~10 `__syncthreads()` per step (~200 cycles each ≈ 1.2 µs total)
- 2× state_history loads from HBM per step (Dk×Dv×4 = 64 KB each)
- Many small bf16 element-wise passes between matmul-style reductions

## Optimizations ranked by ROI

### Tier 1 — the unlock (do these next)

**A. Chunked tensor-core BACKWARD** — biggest single win.
   - Forward: 1 ms at S=512. Backward analytical reference: same matmul
     count + a few extras. Target: **~2 ms/layer at S=512** (vs 12.7 ms today).
   - End-to-end training step at S=512: 277 ms → 80 ms = **3.5× more**.
   - Eliminates the long-S training regression from the wide-shape bench
     (ratio at S=2048 goes from 1.06× to ~3×).
   - Foundation already shipped: `state_chunks` saved by fwd, validated
     analytical Python reference.

**B. tcgen05 instead of nvcuda::wmma** — Blackwell-native MMA.
   - WMMA on Blackwell maps to legacy mma.sync; tcgen05 is the new
     async-pipelined MMA family with bigger tiles and TMEM-backed
     accumulators. ~2× higher peak FLOPs and lower launch latency.
   - Drops chunked fwd from 1 ms → ~0.5 ms at S=512.
   - Expensive to write (PTX-level, no `nvcuda` wrapper), but each kernel
     gets the multiplier for free.

### Tier 2 — long-S specifics (S ≥ 1K)

**C. Larger chunk size when S is big.**
   - Currently C=64. Per-chunk overhead (load q/k/v, build T, etc.) is
     amortized over 64 steps. At C=128, amortized over 128 steps.
   - Inner T-construction loop is O(C³) = 1M ops at C=128, still fast.
   - Memory: shared mem doubles for `[C,C]` buffers (16 KB → 64 KB), but
     chunk-input buffers also double. Tight at C=128 — would need
     to spill some buffers to HBM. Worth ~1.5× at S ≥ 2K.

**D. Multi-block-per-head for the chunk's matmul phase.**
   - Inside one chunk, T's intra-chunk computation is sequential (the
     row-update loop), but everything *after* T is pure matmuls that
     parallelize cleanly across rows. Split rows across 2-4 blocks per
     head → uses more SMs, drops per-chunk latency.
   - Requires inter-block sync via cooperative groups or atomic
     barriers. At C=64 with 4 blocks, sync cost is ~5 µs which is
     amortized across 60+ µs of compute. Worth ~1.5–2× at long S.

### Tier 3 — universal launch-overhead reduction

**E. Fuse multiple DN layers into one kernel launch.**
   - Currently each of 18 DN layers does a separate kernel launch.
     Each launch is ~5 µs of host overhead × 18 = 90 µs per fwd, plus
     kernel-startup latency on each.
   - A "DN-tower kernel" that loops over layers internally cuts launch
     overhead and keeps weights warm in L2 across layers.
   - Hard part: needs all per-layer LoRA tensors + activation buffers
     bound at launch time. Doable; the existing prefill megakernel does
     this for inference.
   - Worth ~10% at short S (where launches dominate), ~1% at long S.

**F. CUDA Graphs** — capture the fwd or fwd+bwd sequence.
   - Replaces N kernel launches with one graph launch. Saves ~5 µs/launch.
   - Trivial to add to autograd path (`torch.cuda.graph`). PyTorch's
     `make_graphed_callables` works out of the box.
   - Worth ~5–15% at short S; negligible at long S.

### Tier 4 — memory-side wins

**G. TMA (Tensor Memory Accelerator) loads on Blackwell.**
   - Replaces explicit `__shared__` loads with async hardware copies.
   - Hides HBM latency behind compute (chunked fwd's q/k/v loads are
     ~30% of per-chunk time today).
   - Requires sm_100a-specific PTX; non-trivial port.

**H. Reduce shared mem to fit 2 blocks per SM.**
   - Recurrent bwd: 200 KB shared mem → 1 block/SM. Cut to <100 KB
     (drop one state slab, recompute via rollback or replay) → 2 blocks/SM,
     better latency hiding.
   - But we already have 16 heads × 1 block = 16 blocks. With 2 blocks
     per SM, occupancy doesn't help unless we ALSO have more blocks.
     So this is only useful in combination with **D** (multi-block-per-head).

### Tier 5 — algorithm-level wins specific to LoRA training

**I. Per-step gradient accumulation directly into LoRA's A/B grads.**
   - dq, dk, dv from the DN bwd flow back into `q_proj`, `k_proj`,
     `v_proj`'s LoRA. Today: bwd kernel writes dq to a [S,H,Dk] tensor;
     PyTorch then does `dq @ x.T` (for grad_A) and `(dq.T @ A) @ x.T`
     (for grad_B). Two extra matmuls per linear per layer.
   - Fuse: have the DN bwd kernel ALSO accumulate directly into
     `grad_A_q`, `grad_B_q`, etc. Eliminates the extra matmuls.
   - Only worth it if PyTorch's autograd-driven bwd of those linears
     turns out to be slow — wide-shape bench suggests it's not the
     bottleneck for our model. Skip unless profiling says otherwise.

## Shape-regime cheat-sheet

| regime | S | dominant cost | best win | est. speedup |
|---|---|---|---|---|
| short | ≤256 | launch overhead | E + F + chunked bwd | 6→8× e2e |
| medium | 512–1K | per-step DN bwd | **A** (chunked bwd) | 2.5→4× e2e |
| long | 2K+ | DN bwd compute | A + C + D | 1.1→3× e2e |
| inference | any | DN fwd | already 5.6×; B → 8–10× | — |

## What I'd ship next, in order

### Already shipped (this PR series)
- **CUDA Graphs wrap** (F) — `cuda_graph_train.GraphedTrainStep`.
  Validated speedup: 1.16× at S=128, 1.07× at S=256, 1.03× at S=512.
- **16-warp chunked fwd** (partial D) — bumped block size from 256 to
  512 threads. Validated speedup over 8-warp: 1.10–1.19× across S.
  Combined with the chunked fwd → 3.11–3.69× over recurrent.

### Remaining (in shipping order, with realistic effort)

The honest picture has changed since fla got installed properly. Beating
fla requires matching its chunk-parallelism architecture, not just
porting our existing per-head kernel to tensor cores.

1. **Chunk-parallelism rewrite** (NEW, was D) — process chunks across
   SMs in parallel with a persistent kernel + per-chunk producer/consumer
   sync. This is the single biggest architectural change required to
   match fla's long-S performance. Real effort: **~1–2 weeks**. Without
   this, the rest is rearranging deck chairs at long S.

2. **CUDA chunked bwd** (was A) — port the validated Python analytical
   bwd. Real effort: **~3–5 focused days**. Algorithm is in
   `dn_chunked_bwd_proto.py` (cos > 0.99998). Useful even with the
   per-head architecture for short-S use cases. Long-S only matters
   if combined with (1).

3. **C=128 chunk variant** (was C) — needs HBM spilling. **~2 days**.
   Marginal vs (1).

4. **tcgen05 port** (was B) — Blackwell-native MMA. **~2–3 weeks**.
   2× peak throughput across kernels. Most useful AFTER (1) since
   tcgen05 helps per-block compute density which matters more once
   blocks aren't already idle.

### When does our kernel actually beat fla?

| regime | who wins | why |
|---|---|---|
| S ≤ 128, training | **us, ~1.16×** | launch overhead; fla has fixed per-call constants |
| S ≤ 512, inference | **us, ~1.27×** | same, plus our chunked fwd is tighter at small S |
| S ≥ 256, training | fla, 1.4–10× | fla has chunked Triton bwd; ours is scalar recurrent |
| S ≥ 1024, inference | fla, 1.0–1.85× | fla parallelizes chunks across SMs |

So the "we are faster than PyTorch" claim is honest only in the short-S
regime today. For genuinely long-context training, **fla is better, full
stop, until we ship the chunk-parallelism rewrite**.
