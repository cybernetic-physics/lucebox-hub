# Training Megakernel (LoRA forward — starting point)

Fork of `../prefill_megakernel` that adds LoRA residuals into the
forward pass inside the single cooperative dispatch. First layer covered:
**q_proj on all 6 Full-Attention layers**. The rest of the trainable
modules (k/v/o/gate/up/down, LoRA over DeltaNet projections, lm_head)
follow the same pattern and are TODO.

## What's shipped

- LoRA forward on q_proj (all 6 FA layers), rank configurable (default 16).
- Two fused phases `phase_lora_h` / `phase_lora_b_add` live inside the
  megakernel; no new launches.
- Kernel accepts `lora_a_q_all`, `lora_b_q_all` pointers plus `use_lora`
  bool. With `use_lora=False` or zero weights, the base forward is
  bit-identical to `prefill_megakernel_C.prefill_bf16`.

## Sanity tests (`test_lora_forward.py`)

```
baseline prefill_megakernel  next_token = 159560
train_mega use_lora=False    next_token = 159560  ✓ (matches baseline)
train_mega use_lora=True A=B=0 next_token = 159560  ✓ (zero LoRA is a no-op)
train_mega use_lora=True A,B≠0 next_token = 159510  ✓ (diverges when LoRA≠0)
OK
```

## Files

- `kernel.cu` — prefill megakernel + LoRA phases
- `torch_bindings.cpp` — `torch.ops.train_megakernel_C.train_mega_forward`
- `setup.py` — CUDAExtension (inherits `sm_{major}{minor}` from current device)
- `test_lora_forward.py` — three-way diff (baseline / LoRA-off / LoRA-on)

## Roadmap (this directory)

1. ✅ LoRA forward on q_proj (all FA layers).
2. ⬜ LoRA forward on k/v/o, gate/up/down (FA path).
3. ⬜ LoRA forward on DeltaNet QKV/Z/out projections (DN path).
4. ⬜ Save per-layer activations during forward so backward can consume them.
5. ⬜ Backward megakernel:
   - CE loss + LM head backward.
   - Final-norm backward.
   - Per-layer reverse: MLP bwd → post-norm bwd → attn/DN bwd → QKV bwd → input-norm bwd.
   - Flash-attention-style bwd for FA; BPTT for DN recurrence.
   - Emit LoRA grads at each trainable linear.
6. ⬜ Fused AdamW phase updating every LoRA A/B pair in one kernel launch.
7. ⬜ End-to-end correctness vs pure-torch Qwen 3.5-0.8B + HF LoRA.

Steps 1–2 are straightforward extensions of the existing phase pattern.
Steps 3–4 are mechanical but voluminous. Step 5's DeltaNet backward
(BPTT through `S_t = (I − β k kᵀ) S_{t−1} + β k vᵀ`) is genuinely
novel — needs the same care as the flash-attention-1 backward work did.

## Build

```
cd train_megakernel
python3 setup.py build_ext --inplace
python3 test_lora_forward.py
```
