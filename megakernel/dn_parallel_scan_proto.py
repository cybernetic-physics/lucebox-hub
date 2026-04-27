"""Parallel-scan reformulation of the gated delta rule — math reference.

The DeltaNet recurrence has a strict serial dependency over t:

    state_{t+1} = decay_t * state_t
                  + beta_t * k_t @ (v_t - decay_t * (k_t.T @ state_t)).T

This module shows that the recurrence is, in fact, a *linear* update on
the state, i.e.

    state_{t+1} = A_t @ state_t + B_t

with

    A_t = decay_t * (I - beta_t * k_t @ k_t.T)         [Dk × Dk]
    B_t = beta_t * k_t @ v_t.T                          [Dk × Dv]

and the operator composition

    (A2, B2) ∘ (A1, B1) = (A2 @ A1,  A2 @ B1 + B2)

is associative — exactly the form a parallel scan handles. The scan
takes a strictly serial dependency of depth T into a tree-shaped
dependency of depth log2(T).

This file:
  1. Implements the existing serial recurrence (sequential reference).
  2. Implements the linear-recurrence form (sanity check that A/B are
     correct).
  3. Implements a Hillis-Steele parallel scan over (A, B) operators.
  4. Validates step 3 against step 1 on a randomized input — to within
     fp32 numerical tolerance.

Once the math is locked, a CUDA kernel can implement the parallel
scan with tensor-core matmuls for the (A, B) compositions. Each
composition is a [Dk, Dk] @ [Dk, Dk] and a [Dk, Dk] @ [Dk, Dv] +
[Dk, Dv] — perfectly sized for WMMA / tcgen05 tiles.

Run:
    python3 dn_parallel_scan_proto.py
"""

import math
import torch


def dn_serial(q, k, v, beta, g, state_0):
    """Sequential reference. Mirrors what `pf_deltanet_recurrence_vsplit_prepped`
    in prefill.cu computes per token.

      decay = exp(g)
      kv    = state.T @ k                     # uses old state (size Dv)
      delta = (v - decay * kv) * beta
      state = decay * state + outer(k, delta) # update
      y     = state.T @ q                     # uses new state (size Dv)

    q/k: [T, Dk]; v: [T, Dv]; beta/g: [T]; state_0: [Dk, Dv]
    Returns (y [T, Dv], state_final [Dk, Dv]).
    """
    T = q.shape[0]
    Dv = v.shape[1]
    state = state_0.clone()
    y = torch.zeros((T, Dv), dtype=state.dtype, device=state.device)
    for t in range(T):
        decay = math.exp(g[t].item())
        kv = state.T @ k[t]                              # [Dv]
        delta = (v[t] - decay * kv) * beta[t]            # [Dv]
        state = decay * state + torch.outer(k[t], delta)
        y[t] = state.T @ q[t]                            # [Dv]
    return y, state


def make_AB(k, v, beta, g):
    """Construct the per-step linear-recurrence operators (A_t, B_t)
    from the post-pre-pass q/k/v/beta/g tensors.

      state_{t+1} = A_t @ state_t + B_t
      A_t = decay_t * (I - beta_t * k_t @ k_t.T)
      B_t = beta_t * outer(k_t, v_t)
    """
    T, Dk = k.shape
    Dv = v.shape[1]
    device, dtype = k.device, k.dtype
    decays = torch.exp(g)                                # [T]
    eye = torch.eye(Dk, device=device, dtype=dtype).expand(T, Dk, Dk)
    kk_T = torch.einsum('ti,tj->tij', k, k)              # [T, Dk, Dk]
    A = decays.view(T, 1, 1) * (eye - beta.view(T, 1, 1) * kk_T)
    B = beta.view(T, 1, 1) * torch.einsum('ti,tj->tij', k, v)  # [T, Dk, Dv]
    return A, B


def dn_linear_serial(q, k, v, beta, g, state_0):
    """Same algorithm as dn_serial, but using the (A, B) operators
    directly. If this matches dn_serial we know the (A, B) construction
    is correct."""
    T = q.shape[0]
    A, B = make_AB(k, v, beta, g)
    state = state_0.clone()
    y = torch.zeros((T, v.shape[1]), dtype=state.dtype, device=state.device)
    for t in range(T):
        state = A[t] @ state + B[t]
        y[t] = state.T @ q[t]
    return y, state


def parallel_scan_compose(A, B):
    """Hillis-Steele parallel prefix scan over the operator (A, B) with
    composition

        (A2, B2) ∘ (A1, B1) = (A2 @ A1, A2 @ B1 + B2).

    Each Hillis-Steele stage takes T compositions in parallel, with
    log2(T) stages total. Returns:

      A_prefix[t] = A[t] @ A[t-1] @ ... @ A[0]
      B_prefix[t] = the composed B reaching position t

    so that state_{t+1} = A_prefix[t] @ state_0 + B_prefix[t].

    A: [T, Dk, Dk]; B: [T, Dk, Dv]. Returns (A_prefix, B_prefix).
    """
    T = A.shape[0]
    A_out = A.clone()
    B_out = B.clone()
    d = 1
    while d < T:
        # Stage d. Position t (for t >= d) absorbs the operator from
        # position t-d. Read both inputs first, then write — no aliasing.
        A_in = A_out.clone()
        B_in = B_out.clone()
        # New_A[t] = A_in[t] @ A_in[t-d]  for t >= d
        # New_B[t] = A_in[t] @ B_in[t-d] + B_in[t]  for t >= d
        A_out[d:] = torch.einsum('tij,tjk->tik', A_in[d:], A_in[:T-d])
        B_out[d:] = torch.einsum('tij,tjk->tik', A_in[d:], B_in[:T-d]) + B_in[d:]
        d *= 2
    return A_out, B_out


def dn_parallel_scan(q, k, v, beta, g, state_0):
    """Parallel-scan implementation of the DN forward.

    Steps:
      1. Compute per-step linear operators (A_t, B_t) — fully parallel.
      2. Parallel prefix scan over (A, B) — log2(T) depth.
      3. state_{t+1} = A_prefix[t] @ state_0 + B_prefix[t] — fully parallel.
      4. y_t = state_{t+1}.T @ q_t — fully parallel.

    """
    A, B = make_AB(k, v, beta, g)
    A_prefix, B_prefix = parallel_scan_compose(A, B)
    # state_{t+1} for each t:
    states_post = torch.einsum('tij,jk->tik', A_prefix, state_0) + B_prefix  # [T, Dk, Dv]
    # y_t = state_{t+1}.T @ q_t  =  einsum tij,ti->tj
    y = torch.einsum('tij,ti->tj', states_post, q)
    state_final = states_post[-1]
    return y, state_final


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64  # use fp64 to isolate algorithmic equivalence from numerical noise

    # Realistic-but-tiny shapes for a quick math check.
    T  = 64
    Dk = 16
    Dv = 16

    q = torch.randn(T, Dk, dtype=dtype, device=device)
    k = torch.randn(T, Dk, dtype=dtype, device=device) * 0.3
    v = torch.randn(T, Dv, dtype=dtype, device=device)
    # beta in (0, 1) like sigmoid; g negative like -exp(a_log)*softplus(...)
    beta = torch.sigmoid(torch.randn(T, dtype=dtype, device=device))
    g = -torch.exp(torch.randn(T, dtype=dtype, device=device)) * 0.1
    state_0 = torch.randn(Dk, Dv, dtype=dtype, device=device) * 0.01

    # 1) sequential reference
    y_serial, state_serial = dn_serial(q, k, v, beta, g, state_0)

    # 2) linear-form serial reference (validates A/B construction)
    y_linear, state_linear = dn_linear_serial(q, k, v, beta, g, state_0)
    diff_y_linear = (y_serial - y_linear).abs().max().item()
    diff_state_linear = (state_serial - state_linear).abs().max().item()
    print(f"Linear-form serial:  max(y diff)     = {diff_y_linear:.3e}")
    print(f"                     max(state diff) = {diff_state_linear:.3e}")

    # 3) parallel-scan implementation
    y_parallel, state_parallel = dn_parallel_scan(q, k, v, beta, g, state_0)
    diff_y_parallel = (y_serial - y_parallel).abs().max().item()
    diff_state_parallel = (state_serial - state_parallel).abs().max().item()
    print(f"Parallel scan:       max(y diff)     = {diff_y_parallel:.3e}")
    print(f"                     max(state diff) = {diff_state_parallel:.3e}")

    # 4) tolerance check
    tol = 1e-9 if dtype == torch.float64 else 1e-3
    assert diff_y_linear < tol and diff_state_linear < tol, \
        "(A, B) construction is wrong"
    assert diff_y_parallel < tol and diff_state_parallel < tol, \
        "parallel scan diverges from serial reference"
    print("OK — parallel-scan reformulation matches the sequential recurrence.")


if __name__ == "__main__":
    main()
