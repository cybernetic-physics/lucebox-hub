"""Correctness test: bwd_ce_lm_head + bwd_rmsnorm vs torch autograd."""

import sys
import torch
import train_megakernel_C  # noqa: F401


HIDDEN = 1024
VOCAB = 248320


def test_bwd_ce_lm_head():
    torch.manual_seed(0)
    final_normed_fp = torch.randn(HIDDEN, device="cuda") * 0.5
    lm_head_w_fp = torch.randn(VOCAB, HIDDEN, device="cuda") * 0.02
    target = 42

    # Reference via torch autograd
    fn_ref = final_normed_fp.detach().clone().requires_grad_(True)
    w_ref = lm_head_w_fp.detach().clone()
    logits = fn_ref @ w_ref.T
    log_probs = torch.log_softmax(logits, dim=-1)
    loss_ref = -log_probs[target]
    loss_ref.backward()
    grad_ref = fn_ref.grad.detach()

    # My kernel
    fn_bf16 = final_normed_fp.to(torch.bfloat16).contiguous()
    w_bf16 = lm_head_w_fp.to(torch.bfloat16).contiguous()
    grad_out = torch.zeros(HIDDEN, device="cuda", dtype=torch.float32)
    loss_out = torch.zeros(1, device="cuda", dtype=torch.float32)
    torch.ops.train_megakernel_C.bwd_ce_lm_head(
        fn_bf16, w_bf16, target, grad_out, loss_out,
    )
    torch.cuda.synchronize()

    # Compare
    ref_loss = loss_ref.item()
    my_loss = loss_out.item()
    diff = (grad_out - grad_ref).abs()
    print(f"loss  ref={ref_loss:.4f}  mine={my_loss:.4f}  Δ={abs(ref_loss - my_loss):.2e}")
    print(f"grad  max|Δ|={diff.max().item():.2e}  mean|Δ|={diff.mean().item():.2e}")
    if abs(ref_loss - my_loss) > 5e-2 or diff.max().item() > 0.5:
        print("CE+LM HEAD BWD FAIL"); return False
    print("CE+LM HEAD BWD OK")
    return True


def test_bwd_rmsnorm():
    torch.manual_seed(1)
    S = 4
    H = HIDDEN
    eps = 1e-6

    x_fp = torch.randn(S, H, device="cuda") * 0.5
    w_fp = torch.randn(H, device="cuda") * 0.1

    # torch reference
    x_ref = x_fp.detach().clone().requires_grad_(True)
    w_ref = w_fp.detach().clone()
    ms = (x_ref * x_ref).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    y = x_ref * rstd * (1.0 + w_ref)
    dy_fixed = torch.randn_like(y) * 0.1
    y.backward(dy_fixed)
    grad_ref = x_ref.grad.detach()

    # My kernel (inputs bf16)
    x_bf16 = x_fp.to(torch.bfloat16).contiguous()
    w_bf16 = w_fp.to(torch.bfloat16).contiguous()
    dy_f32 = dy_fixed.detach().float().contiguous()
    dx_out = torch.zeros(S, H, device="cuda", dtype=torch.float32)
    torch.ops.train_megakernel_C.bwd_rmsnorm(x_bf16, w_bf16, dy_f32, dx_out, S, H, eps)
    torch.cuda.synchronize()

    diff = (dx_out - grad_ref).abs()
    rel = diff / (grad_ref.abs() + 1e-6)
    print(f"rmsnorm-bwd  max|Δ|={diff.max().item():.2e}  max rel={rel.max().item():.2e}  mean|Δ|={diff.mean().item():.2e}")
    if rel.max().item() > 0.1:
        print("RMSNORM BWD FAIL"); return False
    print("RMSNORM BWD OK")
    return True


def main():
    ok = True
    ok &= test_bwd_ce_lm_head()
    ok &= test_bwd_rmsnorm()
    print("ALL OK" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
