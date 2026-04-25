"""Numerically check that cuda_chunk_gated_delta_rule produces the same
output (within bf16 noise) as HF's torch_chunk_gated_delta_rule on the
same inputs, including the l2norm + scale + log-decay preprocessing.

We can't go via 'from transformers.models.qwen3_next...' because the
fla import in this transformers install raises an InvalidVersion error.
So we inline the HF reference function below.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from dn_hf_patch import cuda_chunk_gated_delta_rule


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                                  initial_state=None, output_final_state=False,
                                  use_qk_l2norm_in_kernel=False):
    """Verbatim copy of HF transformers' Qwen3Next reference."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key   = F.pad(key,   (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta  = F.pad(beta,  (0, pad_size))
    g     = F.pad(g,     (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key   * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_in = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        core_attn_out[:, :, i] = attn_inter + attn_in @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state: last_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_state


def diff(name, a, b):
    af, bf = a.to(torch.float32), b.to(torch.float32)
    mx = (af - bf).abs().max().item()
    cos = F.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
    print(f"  {name:<14} max|Δ|={mx:.4e}  cos={cos:.6f}")
    return cos


def main():
    torch.manual_seed(0)
    H, Dk, Dv = 16, 128, 128
    for S in (64, 128, 512):
        # HF expects S divisible by chunk_size=64; pick from {64, 128, 512}.
        dev = "cuda"
        q = (torch.randn(1, S, H, Dk, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        k = (torch.randn(1, S, H, Dk, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        v = (torch.randn(1, S, H, Dv, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        beta = torch.sigmoid(torch.randn(1, S, H, dtype=torch.float32, device=dev))
        g = -torch.rand(1, S, H, dtype=torch.float32, device=dev) * 0.2

        # Forward parity (l2norm on, like HF default).
        y_ref, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta,
                                                use_qk_l2norm_in_kernel=True)
        y_ours, _ = cuda_chunk_gated_delta_rule(q, k, v, g, beta,
                                                use_qk_l2norm_in_kernel=True)
        print(f"--- S={S} (l2norm=True) ---")
        cy = diff("y", y_ours, y_ref)
        print(f"  -> {'PASS' if cy > 0.99 else 'FAIL'}")


if __name__ == "__main__":
    main()
