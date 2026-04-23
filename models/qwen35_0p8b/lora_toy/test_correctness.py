"""Correctness check: fused megakernel vs pure-torch reference.

Runs the same training recipe through both implementations and diffs:
- selected log-probs (forward output)
- loss
- post-step LoRA parameters

Shapes are small so the test is fast; the kernel path is exercised as a
drop-in replacement for a single training step.
"""

from __future__ import annotations

import sys
import torch

from model import AdamConfig, LoRATrainStepKernel, _seed_from_name


class TorchReference:
    """Pure torch implementation with the exact same math as the kernel."""

    def __init__(self, *, base_model: str, vocab_size: int, hidden_size: int, lora_rank: int, device: torch.device):
        gen = torch.Generator(device="cpu").manual_seed(_seed_from_name(base_model))
        emb = (torch.randn(vocab_size, hidden_size, generator=gen) * 0.05).to(device=device, dtype=torch.bfloat16)
        out = (torch.randn(vocab_size, hidden_size, generator=gen) * 0.05).to(device=device, dtype=torch.bfloat16)
        a = (torch.randn(hidden_size, lora_rank, generator=gen) * 0.01).to(device=device, dtype=torch.bfloat16)
        b = torch.zeros(lora_rank, vocab_size, device=device, dtype=torch.bfloat16)

        # Frozen.
        self.embedding = emb
        self.output_weight = out
        # Trainable.
        self.lora_a = torch.nn.Parameter(a.float())
        self.lora_b = torch.nn.Parameter(b.float())
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank

        self.opt = torch.optim.AdamW([self.lora_a, self.lora_b], lr=1e-3)
        self.step = 0

    def forward_loss(self, context_tokens, target_tokens, token_weights):
        # Emulate the kernel's math in fp32 for reference.
        ctx = context_tokens.to(self.device, dtype=torch.long)
        tgt = target_tokens.to(self.device, dtype=torch.long)
        w = token_weights.to(self.device, dtype=torch.float32)

        hidden = self.embedding[ctx].float()  # [T, H] in fp32 (mirrors kernel which accumulates in fp32)
        base_logits = hidden @ self.output_weight.float().T  # [T, V]
        lora_h = hidden @ self.lora_a  # [T, R]
        lora_logits = lora_h @ self.lora_b  # [T, V]
        logits = base_logits + lora_logits

        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
        loss = -(selected * w).mean()
        return loss, selected

    def train_step(self, context_tokens, target_tokens, token_weights, adam: AdamConfig):
        for g in self.opt.param_groups:
            g["lr"] = adam.lr
            g["betas"] = (adam.beta1, adam.beta2)
            g["eps"] = adam.eps
            g["weight_decay"] = adam.weight_decay
        self.opt.zero_grad(set_to_none=True)
        loss, selected = self.forward_loss(context_tokens, target_tokens, token_weights)
        loss.backward()
        self.opt.step()
        self.step += 1
        return float(loss.detach()), selected.detach()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Start with the rl/backend's actual config so we stress the exact shapes it uses.
    shapes = [
        {"base": "meta-llama/Llama-3.2-1B", "V": 256, "H": 64, "R": 8, "T": 8},
        {"base": "Qwen/Qwen2.5-0.5B", "V": 1024, "H": 128, "R": 16, "T": 16},
        {"base": "bigger", "V": 4096, "H": 256, "R": 16, "T": 32},
    ]

    fail = False
    for cfg in shapes:
        print(f"\n=== shape V={cfg['V']} H={cfg['H']} R={cfg['R']} T={cfg['T']} ===")

        ref = TorchReference(
            base_model=cfg["base"], vocab_size=cfg["V"], hidden_size=cfg["H"],
            lora_rank=cfg["R"], device=device,
        )
        kernel = LoRATrainStepKernel.from_base_model(
            base_model=cfg["base"], lora_rank=cfg["R"],
            vocab_size=cfg["V"], hidden_size=cfg["H"],
            max_seq_len=max(64, cfg["T"]), device=device,
        )

        # Sanity: initial params identical.
        d_a0 = _max_abs(ref.lora_a.detach(), kernel.lora_a.float())
        d_b0 = _max_abs(ref.lora_b.detach(), kernel.lora_b.float())
        print(f"  initial lora_a max|Δ|: {d_a0:.2e}  lora_b: {d_b0:.2e}")

        adam = AdamConfig(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)

        for step in range(3):
            T = cfg["T"]
            ctx = torch.randint(0, cfg["V"], (T,), device=device, dtype=torch.int32)
            tgt = torch.randint(0, cfg["V"], (T,), device=device, dtype=torch.int32)
            w = torch.ones(T, device=device, dtype=torch.float32)

            ref_loss, ref_selected = ref.train_step(ctx, tgt, w, adam)
            out = kernel.train_step(context_tokens=ctx, target_tokens=tgt, token_weights=w, adam=adam)
            k_loss = out["loss"]
            k_selected = out["selected_log_probs"]

            # bf16 matmul over moderate H introduces rounding; allow a few hundredths.
            d_sel = _max_abs(ref_selected, k_selected)
            d_loss = abs(ref_loss - k_loss)
            d_a = _max_abs(ref.lora_a.detach(), kernel.lora_a.float())
            d_b = _max_abs(ref.lora_b.detach(), kernel.lora_b.float())

            print(f"  step {step}: loss ref={ref_loss:.5f} k={k_loss:.5f} Δ={d_loss:.2e}")
            print(f"           selected max|Δ|={d_sel:.2e}  lora_a Δ={d_a:.2e}  lora_b Δ={d_b:.2e}")

            # Tolerances: bf16 weights + fp32 accumulation; scale with H.
            tol_sel = 5e-2 if cfg["H"] <= 64 else 2e-1
            tol_loss = 1e-2
            tol_param = 1e-2

            if d_sel > tol_sel or d_loss > tol_loss or d_a > tol_param or d_b > tol_param:
                print("  FAILED tolerance")
                fail = True

    if fail:
        print("\nCORRECTNESS FAIL")
        sys.exit(1)
    print("\nCORRECTNESS OK")


if __name__ == "__main__":
    main()
