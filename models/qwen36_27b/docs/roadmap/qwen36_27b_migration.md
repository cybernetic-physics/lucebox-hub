# Qwen3.6-27B megakernel migration writeup

This is the postmortem for porting the 0.8B megakernel pattern from
`models/qwen35_0p8b/` to dense 27B with hybrid Gated DeltaNet + Gated
Attention. Captured May 2026 once the BF16 path passed C7 (top-1 vs HF
on S=32..256 wikitext).

## What stayed the same

- **Kernel structure.** Persistent cooperative-grid megakernel; one
  decode call per token. Layer dispatcher walks `LayerWeights<Cfg>[]`
  and calls the FA or DN forward.
- **DN core algorithm.** Gated DeltaNet chunked scan, conv1d ring
  buffer, per-V-head state evolution, group-RMSNorm + SiLU(z) output
  gate.
- **FA core.** RMSNorm + Q/K-norm + RoPE per head + online-softmax
  split-K attention + sigmoid gate (not swish, despite the config
  string — C5d).

## What changed

| | 0.8B | 27B |
|---|---:|---:|
| NUM_LAYERS | 24 | 64 |
| HIDDEN | 1024 | 5120 |
| INTERMEDIATE | 3584 | 17408 |
| FA Q heads | 8 | 24 |
| FA KV heads | 2 | 4 |
| FA head_dim | 256 | 256 |
| DN V heads | 16 | 48 |
| DN QK heads | 16 | 16 |
| DN V_PER_QK | 1 | **3** |
| Native context | 32k | 262k |
| RoPE theta | 1e6 | **1e7** |

The single real architectural change is **DN V/QK split** — 27B
shares Q and K across 3 V heads (GQA-style within DN). That meant a
new `Cfg::DN_V_PER_QK` constant plus rewriting the V-head loop in
`dn_layer.cuh` so each V head looks up its parent QK head:

```cpp
int v_head  = block_id;
int qk_head = v_head / V_PER_QK;
```

The recurrence state still lives per V head (`dn_state[v_head]`) but
the Q/K projections are shared across the V_PER_QK siblings.

## What surprised us

### 1. FA layer was missing the MLP entirely (C5d)

The FA struct declared `post_attn_layernorm_weight`, `gate_proj_weight`,
`up_proj_weight`, `down_proj_weight` — but the forward function ended
at O-proj + residual. We were skipping the standard SwiGLU MLP every
4th layer (16 layers total). Cumulative drift across 16 missing MLPs
explained the catastrophic divergence the C3 layer-diff test caught.

Fix: append the same post-attn norm + SwiGLU + down + residual block
that the DN layer had. After the fix: cos jumped from 0.87 to
0.9996 and top-1 matched HF.

### 2. Gate activation is sigmoid, not swish (C5)

`config.json` reports `output_gate_type="swish"` but applying SiLU
to the gate at the O-proj input gave cos=0.81 (vs sigmoid at 0.997).
The reference runtime must use plain sigmoid despite the config
string. Logged in `fa_layer.cuh:303` for future-us.

### 3. Q+gate layout is interleaved, not split (C5)

We tried both per-head split (Q half, gate half) and interleaved (Q
half, gate half within each head's pair). Split made cos=0.948 at
layer 3 vs 0.997 with interleaved. Kept interleaved. Confirmed
matches HF's `q_proj` output layout.

### 4. MRoPE: text-only needs all 3 axes set to pos_t (C6 pre-fix)

When pos_h and pos_w were 0, only the temporal section of the
rotary_dim rotated and h/w sections stayed static — which is NOT
equivalent to standard RoPE. Fixed at the runtime call site:
`(int)tok, pos, pos, pos`.

### 5. State-dict prefix is `model.layers.X.*`, not `model.language_model.layers.X.*`

The safetensors index uses the `model.language_model.*` namespace but
HF's `AutoModelForCausalLM` flattens that wrapper at load time. The
state_dict() returned to Python uses the flat prefix. Took an hour to
notice that our packer was looking up nothing.

### 6. Pack stride bug took the kernel out (S1c)

When we extended `LayerWeights<Cfg>` to hold the NVFP4 union members
(192 B), Python's `PACK_STRUCT = ((8 + 21*8 + 15)//16)*16 = 176` was
silently 16 bytes too small. The kernel read every layer past 0 from
the wrong offset and crashed in FA gate/up matvec with an OOB load.
Fix: hardcode `PACK_STRUCT = 192` + asserts. Lesson: when the C++
struct size depends on `_force_size[N]`, the Python stride must be
derived from N directly, not from a max-ptr count.

## What we did NOT have to change

- **Megakernel control flow** (cooperative grid + grid-sync between
  sections). Same launch model as 0.8B.
- **FA attention scan.** Split-K online softmax, same code path.
- **DN conv1d ring buffer.** Identical to 0.8B.
- **YaRN scaling.** Already implemented (0.8B never needed it, the
  scaffold was there).

## Debug strategy that worked

1. **C1.** Just call decode on real weights with a single token.
   Crash → fix; runs → log argmax.
2. **C2/C3.** Add a per-layer hidden-state capture buffer to the
   kernel (optional `g_layer_outputs` arg). HF side: forward with
   `output_hidden_states=True`. Diff each layer's cos / max-abs.
3. **C4.** First-divergent layer points at the responsible function
   (FA vs DN). Read that function side-by-side with HF until the
   delta makes sense.

This caught the FA-MLP-missing bug at layer 3 cleanly — the first
divergence was the first FA layer, and the drift pattern was "FA
adds linear noise per pass" which fingered the missing residual
update.

## Open questions

- The cos drift at S=64 (0.966 vs 0.997 at S=32) — top-1 still
  matches, but the distribution widens. Probably DN state
  accumulation across the host-loop prefill. Will be answered by S2
  (parallel-S kernel) which replaces the host loop with a chunked
  scan that matches HF's algorithm exactly.
- NVFP4 numeric parity vs HF (S1e). The dispatch path works
  (test_nvfp4_dispatch.py); correctness vs HF still needs a test.

## See also

- [docs/results/qwen36_27b_gb10.md](../results/qwen36_27b_gb10.md)
  current numbers
- [TODO.md](../../TODO.md) — actionable backlog
- [SESSION_NOTES.md](../../SESSION_NOTES.md) — investigation log
