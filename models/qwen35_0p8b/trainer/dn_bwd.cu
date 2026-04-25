/**
 * DeltaNet BPTT — first-pass CUDA kernel.
 *
 * Algorithm (Qwen3.5-0.8B linear-attention, per head, S timesteps):
 *
 *   Forward:
 *     state_t  = decay_t * state_{t-1} + k_t ⊗ delta_t
 *     delta_t  = beta_t * (v_t - state_{t-1} . k_t)
 *     out_t    = state_t . q_t
 *
 *   Backward (given dy_t):
 *     d(state_t)_via_out      = q_t ⊗ dy_t                       — outer product
 *     dq_t                    = state_t.T . dy_t                  — [Dk]
 *     accumulated dstate_t    = (forward decay flow from t+1) + d(state_t)_via_out
 *
 *     d_delta_t               = dstate_t.T . k_t                  — [Dv]
 *     dv_t                    = beta_t * d_delta_t
 *     dbeta_t                 = sum(d_delta_t * (v_t - state_{t-1} . k_t))
 *     # Through delta_t = beta * (v - state_{t-1} . k):
 *     #   d(state_{t-1})_via_delta_via_k = -beta_t * (k_t ⊗ d_delta_t).T  [Dk, Dv]
 *     # Through state_t = decay * state_{t-1} + k ⊗ delta:
 *     #   d(state_{t-1})_via_decay = decay_t * dstate_t
 *     # Combined:
 *     dstate_{t-1}            = decay_t * dstate_t  +  d(state_{t-1})_via_delta_via_k
 *
 *     dk_t                    = (dstate_t . delta_t)              — from k ⊗ delta
 *                              + (-beta_t) * (state_{t-1}.T . d_delta_t)  — from state_{t-1} . k_t in delta
 *     ddecay_t                = sum(state_{t-1} * dstate_t)
 *
 * Block strategy (this first pass):
 *   Grid: one block per HEAD. H_HEADS blocks total.
 *   Block: 256 threads collaborating on the [Dk=128, Dv=128] state matrix.
 *   Each thread owns Dk*Dv / blockDim.x = 64 state cells.
 *
 * Forward saves delta_t to a global [S, H, Dv] bf16 slab so backward
 * can recompute state_{t-1} on the fly without storing the full state
 * sequence. Memory cost: S * H * Dv * 2 bytes — at S=512 H=16 Dv=128
 * that's 2 MB total. Fine.
 *
 * Backward then walks t = S-1 .. 0 using:
 *   state_t computed forward inside the kernel from saved
 *   chunk_states_in (pre-chunk-boundary state) + saved deltas. We keep
 *   only ONE [Dk, Dv] state in shared memory and roll it forward as we
 *   need; then the backward sweep uses dstate (also [Dk, Dv] in shared).
 *
 * THIS is the simplest design that compiles. Next iterations can:
 *   1. Use bf16 WGMMA for the outer products + matmuls instead of
 *      thread-level fp32 ops (target ~5 ms at S=512 vs ~50 ms scalar).
 *   2. Process multiple heads per SM via persistent threads.
 *   3. Save k @ delta partial matmuls to amortize across decay propagation.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>


// Hardcoded for Qwen3.5-0.8B DeltaNet shapes — the kernel templates on
// these via #defines so the compiler can unroll and place state in
// registers / shared cleanly. Generalizing later is one #define swap.
#ifndef DN_HEADS_DEFAULT
#define DN_HEADS_DEFAULT 16
#endif
#ifndef DN_KEY_DEFAULT
#define DN_KEY_DEFAULT 128
#endif
#ifndef DN_VAL_DEFAULT
#define DN_VAL_DEFAULT 128
#endif


// Block-reduce-sum via warp shuffles. `scratch` must be at least
// (blockDim.x + 31) / 32 floats. Returns the meaningful sum on tid==0
// (and undefined elsewhere). Issues exactly one __syncthreads().
__device__ __forceinline__ float block_reduce_sum(float val, float *scratch,
                                                  int tid, int nt) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) scratch[warp] = val;
    __syncthreads();
    int n_warps = (nt + 31) >> 5;
    if (warp == 0) {
        val = (tid < n_warps) ? scratch[tid] : 0.0f;
        val += __shfl_xor_sync(0xffffffff, val, 16);
        val += __shfl_xor_sync(0xffffffff, val, 8);
        val += __shfl_xor_sync(0xffffffff, val, 4);
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
    }
    return val;
}


// ===================== forward + delta save kernel =====================
// One block per head. Walks t = 0 .. S-1, accumulating state in shared
// memory (256 threads × 64 cells = 16384 = Dk * Dv). Writes per-step
// delta to delta_save[t, h, :] for the backward to consume.
//
// Inputs (all per-head views into the larger tensor — outer indexing is
// done by the caller via head pointer arithmetic):
//   q, k     : [S, Dk]   bf16   (one head's Q and K post conv/silu/norm)
//   v        : [S, Dv]   bf16
//   beta     : [S]       fp32   (sigmoid'd)
//   decay    : [S]       fp32
//   state_in : [Dk, Dv]  fp32   (initial state, often zero)
// Outputs:
//   y          : [S, Dv]  bf16  (recurrence output for this head)
//   state_out  : [Dk, Dv] fp32  (final state)
//   delta_save : [S, Dv]  bf16  (per-step delta_t for use in bwd)
//
// Block: 256 threads. Each thread owns DK*DV/256 = 64 state cells.
__global__ void dn_fwd_with_delta_save_kernel(
    const __nv_bfloat16 *__restrict__ q_base,
    const __nv_bfloat16 *__restrict__ k_base,
    const __nv_bfloat16 *__restrict__ v_base,
    const float *__restrict__ beta_base,
    const float *__restrict__ decay_base,
    const float *__restrict__ state_in_base,
    __nv_bfloat16 *__restrict__ y_base,
    float *__restrict__ state_out_base,
    __nv_bfloat16 *__restrict__ delta_save_base,
    float *__restrict__ state_history_base,   // [H, S+1, Dk, Dv]
    int S,
    int qkd_pos_stride,
    int v_pos_stride,
    int bd_pos_stride,
    int state_hist_stride_t)              // Dk*Dv
{
    constexpr int Dk = DN_KEY_DEFAULT;
    constexpr int Dv = DN_VAL_DEFAULT;
    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int h   = blockIdx.x;

    // Per-head views.
    const __nv_bfloat16 *q     = q_base + h * Dk;
    const __nv_bfloat16 *k     = k_base + h * Dk;
    const __nv_bfloat16 *v     = v_base + h * Dv;
    const float *beta          = beta_base  + h;
    const float *decay         = decay_base + h;
    const float *state_in      = state_in_base + h * Dk * Dv;
    __nv_bfloat16 *y           = y_base + h * Dv;
    float *state_out           = state_out_base ? state_out_base + h * Dk * Dv : nullptr;
    __nv_bfloat16 *delta_save  = delta_save_base + h * Dv;
    float *state_history       = state_history_base
        ? state_history_base + h * (S + 1) * (size_t)Dk * Dv : nullptr;

    // ----- shared layout -----
    // state[Dk, Dv] fp32, sk[Dv] fp32 reduction scratch, sk_cache[Dv] fp32,
    // s_q[Dk] fp32, s_k[Dk] fp32, s_v[Dv] fp32
    extern __shared__ float smem[];
    float *state    = smem;                  // Dk*Dv
    float *s_q      = state + Dk * Dv;       // Dk
    float *s_k      = s_q + Dk;               // Dk
    float *s_v      = s_k + Dk;               // Dv
    float *s_sk     = s_v + Dv;               // Dv  (intermediate state.k)
    float *s_delta  = s_sk + Dv;              // Dv

    // Load initial state.
    for (int i = tid; i < Dk * Dv; i += nt) state[i] = state_in[i];
    __syncthreads();

    // Snapshot state at t=0 (state_{-1} aka state_init) into history[0].
    if (state_history) {
        for (int i = tid; i < Dk * Dv; i += nt)
            state_history[0 * state_hist_stride_t + i] = state[i];
    }

    for (int t = 0; t < S; t++) {
        // ----- load q, k, v for this step into shared -----
        const __nv_bfloat16 *q_t = q + t * qkd_pos_stride;
        const __nv_bfloat16 *k_t = k + t * qkd_pos_stride;
        const __nv_bfloat16 *v_t = v + t * v_pos_stride;
        for (int i = tid; i < Dk; i += nt) s_q[i] = __bfloat162float(q_t[i]);
        for (int i = tid; i < Dk; i += nt) s_k[i] = __bfloat162float(k_t[i]);
        for (int i = tid; i < Dv; i += nt) s_v[i] = __bfloat162float(v_t[i]);
        float beta_t  = beta [t * bd_pos_stride];
        float decay_t = decay[t * bd_pos_stride];
        __syncthreads();

        // ----- step 1: sk[j] = sum_i state[i, j] * k[i]   (fp32) -----
        // Layout: state[i, j] at state[i * Dv + j]. Each thread handles
        // a subset of j.
        for (int j = tid; j < Dv; j += nt) {
            float acc = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < Dk; i++) {
                acc += state[i * Dv + j] * s_k[i];
            }
            s_sk[j] = acc;
        }
        __syncthreads();

        // ----- step 2: delta[j] = beta_t * (v[j] - sk[j]) -----
        for (int j = tid; j < Dv; j += nt) {
            float d = beta_t * (s_v[j] - s_sk[j]);
            s_delta[j] = d;
            if (delta_save) delta_save[t * v_pos_stride + j] = __float2bfloat16(d);
        }
        __syncthreads();

        // ----- step 3: state[i, j] = decay * state[i, j] + k[i] * delta[j] -----
        // Then attn[j] = sum_i state[i, j] * q[i].
        // We fuse: update state first, then per-(i,j) accumulate.
        for (int idx = tid; idx < Dk * Dv; idx += nt) {
            int i = idx / Dv;
            int j = idx - i * Dv;
            float new_state = decay_t * state[idx] + s_k[i] * s_delta[j];
            state[idx] = new_state;
        }
        __syncthreads();

        // ----- step 4: attn[j] = sum_i state[i, j] * q[i] -----
        for (int j = tid; j < Dv; j += nt) {
            float a = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < Dk; i++) {
                a += state[i * Dv + j] * s_q[i];
            }
            y[t * v_pos_stride + j] = __float2bfloat16(a);
        }
        // Snapshot post-step state into history[t+1].
        if (state_history) {
            for (int i = tid; i < Dk * Dv; i += nt)
                state_history[(t + 1) * state_hist_stride_t + i] = state[i];
        }
        __syncthreads();
    }

    // Final state writeback.
    if (state_out) {
        for (int i = tid; i < Dk * Dv; i += nt) state_out[i] = state[i];
    }
}


// ===================== backward kernel =====================
// Walks t = S-1 .. 0 using state replay from delta_save + initial state.
// One block per head. Same shared-memory budget as the forward kernel.
//
// Inputs:
//   q, k         : [S, Dk]   bf16   (post conv/silu/norm — same as fwd input)
//   v            : [S, Dv]   bf16
//   beta         : [S]       fp32
//   decay        : [S]       fp32
//   state_in     : [Dk, Dv]  fp32   (initial state from forward)
//   delta_save   : [S, Dv]   bf16   (saved per-step delta from forward)
//   dy           : [S, Dv]   bf16   (grad flowing back from out)
// Outputs:
//   dq, dk       : [S, Dk]   bf16
//   dv           : [S, Dv]   bf16
//   dbeta, ddecay: [S]       fp32
//   dstate_init  : [Dk, Dv]  fp32   (grad of initial state — for incremental prefill)
//
// Block: 256 threads.
__global__ void dn_bwd_kernel(
    const __nv_bfloat16 *__restrict__ q_base,
    const __nv_bfloat16 *__restrict__ k_base,
    const __nv_bfloat16 *__restrict__ v_base,
    const float *__restrict__ beta_base,
    const float *__restrict__ decay_base,
    const float *__restrict__ state_in_base,
    const __nv_bfloat16 *__restrict__ delta_save_base,
    const __nv_bfloat16 *__restrict__ dy_base,
    const float *__restrict__ state_history_base,    // [H, S+1, Dk, Dv]
    __nv_bfloat16 *__restrict__ dq_base,
    __nv_bfloat16 *__restrict__ dk_base,
    __nv_bfloat16 *__restrict__ dv_base,
    float *__restrict__ dbeta_base,
    float *__restrict__ ddecay_base,
    float *__restrict__ dstate_init_base,
    int S,
    int qkd_pos_stride,
    int v_pos_stride,
    int bd_pos_stride,
    int state_hist_stride_t)
{
    constexpr int Dk = DN_KEY_DEFAULT;
    constexpr int Dv = DN_VAL_DEFAULT;
    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int h   = blockIdx.x;

    // Per-head views.
    const __nv_bfloat16 *q          = q_base          + h * Dk;
    const __nv_bfloat16 *k          = k_base          + h * Dk;
    const __nv_bfloat16 *v          = v_base          + h * Dv;
    const float *beta               = beta_base       + h;
    const float *decay              = decay_base      + h;
    (void)state_in_base;  // unused — backward replays state from state_history.
    const __nv_bfloat16 *delta_save = delta_save_base + h * Dv;
    const __nv_bfloat16 *dy         = dy_base         + h * Dv;
    const float *state_history      = state_history_base + h * (S + 1) * (size_t)Dk * Dv;
    __nv_bfloat16 *dq               = dq_base         + h * Dk;
    __nv_bfloat16 *dk               = dk_base         + h * Dk;
    __nv_bfloat16 *dv               = dv_base         + h * Dv;
    float *dbeta                    = dbeta_base      + h;
    float *ddecay                   = ddecay_base     + h;
    float *dstate_init              = dstate_init_base
        ? dstate_init_base + h * Dk * Dv : nullptr;

    extern __shared__ float smem[];
    float *state       = smem;                        // [Dk, Dv] current state_t (loaded from history)
    float *prev_state  = state       + Dk * Dv;       // [Dk, Dv] state_{t-1} (also from history)
    float *dstate      = prev_state  + Dk * Dv;       // [Dk, Dv] accumulator
    float *s_q         = dstate      + Dk * Dv;       // [Dk]
    float *s_k         = s_q         + Dk;             // [Dk]
    float *s_v         = s_k         + Dk;             // [Dv]
    float *s_dy        = s_v         + Dv;             // [Dv]
    float *s_delta     = s_dy        + Dv;             // [Dv]
    float *s_d_delta   = s_delta     + Dv;             // [Dv]
    float *s_red       = s_d_delta   + Dv;             // [blockDim.x]

    // Initialize dstate to zero.
    for (int i = tid; i < Dk * Dv; i += nt) dstate[i] = 0.0f;
    __syncthreads();

    // ---- backward walk t = S-1 .. 0 ----
    for (int t = S - 1; t >= 0; t--) {
        // Load per-step quantities into shared.
        const __nv_bfloat16 *q_t  = q  + t * qkd_pos_stride;
        const __nv_bfloat16 *k_t  = k  + t * qkd_pos_stride;
        const __nv_bfloat16 *v_t  = v  + t * v_pos_stride;
        const __nv_bfloat16 *dy_t = dy + t * v_pos_stride;
        for (int i = tid; i < Dk; i += nt) s_q[i] = __bfloat162float(q_t[i]);
        for (int i = tid; i < Dk; i += nt) s_k[i] = __bfloat162float(k_t[i]);
        for (int j = tid; j < Dv; j += nt) s_v [j] = __bfloat162float(v_t [j]);
        for (int j = tid; j < Dv; j += nt) s_dy[j] = __bfloat162float(dy_t[j]);
        for (int j = tid; j < Dv; j += nt)
            s_delta[j] = __bfloat162float(delta_save[t * v_pos_stride + j]);
        float beta_t  = beta [t * bd_pos_stride];
        float decay_t = decay[t * bd_pos_stride];
        __syncthreads();

        // ----- load state_t from history[t+1] and state_{t-1} from history[t] -----
        for (int idx = tid; idx < Dk * Dv; idx += nt) {
            state     [idx] = state_history[(t + 1) * state_hist_stride_t + idx];
            prev_state[idx] = state_history[(t    ) * state_hist_stride_t + idx];
        }
        __syncthreads();

        // ----- dstate += dy_t ⊗ q_t  (because out_t = state_t.q_t) -----
        // Element (i, j): dstate[i, j] += s_q[i] * s_dy[j].
        // No sync after this: dq below reads `state` (unchanged) and `s_dy`
        // (unchanged), so it's safe to start before all dstate writes drain;
        // the sync at the end of dq ensures dstate is visible before d_delta.
        for (int idx = tid; idx < Dk * Dv; idx += nt) {
            int i = idx / Dv, j = idx - i * Dv;
            dstate[idx] += s_q[i] * s_dy[j];
        }

        // ----- dq[i] = sum_j state_t[i, j] * dy[j] -----
        for (int i = tid; i < Dk; i += nt) {
            float acc = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < Dv; j++) acc += state[i * Dv + j] * s_dy[j];
            dq[t * qkd_pos_stride + i] = __float2bfloat16(acc);
        }
        __syncthreads();

        // ----- d_delta[j] = sum_i k[i] * dstate[i, j] -----
        // (delta enters state via k ⊗ delta).
        for (int j = tid; j < Dv; j += nt) {
            float acc = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < Dk; i++) acc += s_k[i] * dstate[i * Dv + j];
            s_d_delta[j] = acc;
        }
        __syncthreads();

        // ----- dv[j] = beta_t * d_delta[j] -----
        for (int j = tid; j < Dv; j += nt) {
            dv[t * v_pos_stride + j] = __float2bfloat16(beta_t * s_d_delta[j]);
        }
        // ----- dbeta_t = sum_j d_delta[j] * (v[j] - prev_state.k_t)[j] -----
        // pk[j] = sum_i prev_state[i, j] * s_k[i]. Per-thread partial,
        // then block-reduce via shared mem.
        float my_partial = 0.0f;
        for (int j = tid; j < Dv; j += nt) {
            float pk = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < Dk; i++) pk += prev_state[i * Dv + j] * s_k[i];
            my_partial += s_d_delta[j] * (s_v[j] - pk);
        }
        float beta_sum = block_reduce_sum(my_partial, s_red, tid, nt);
        if (tid == 0) dbeta[t * bd_pos_stride] = beta_sum;
        __syncthreads();

        // ----- dk: two pieces -----
        //  (a) dk_a[i] = sum_j dstate[i, j] * delta[j]   from k ⊗ delta in state update
        //  (b) dk_b[i] = -beta_t * sum_j prev_state[i, j] * d_delta[j]   from sk = state_{t-1}.k inside delta
        for (int i = tid; i < Dk; i += nt) {
            float a = 0.0f, b = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < Dv; j++) {
                a += dstate[i * Dv + j] * s_delta[j];
                b += prev_state[i * Dv + j] * s_d_delta[j];
            }
            float dk_t = a - beta_t * b;
            dk[t * qkd_pos_stride + i] = __float2bfloat16(dk_t);
        }
        __syncthreads();

        // ----- ddecay_t = sum_(i,j) prev_state[i, j] * dstate[i, j] -----
        float my_dec = 0.0f;
        for (int idx = tid; idx < Dk * Dv; idx += nt) {
            my_dec += prev_state[idx] * dstate[idx];
        }
        float dec_sum = block_reduce_sum(my_dec, s_red, tid, nt);
        if (tid == 0) ddecay[t * bd_pos_stride] = dec_sum;
        __syncthreads();

        // ----- update dstate for next (earlier) step -----
        // dstate_{t-1} = decay_t * dstate_t  +  d(state_{t-1})_via_delta
        // The delta path: delta_t = beta * (v_t - sk_t) where sk_t = state_{t-1} . k_t.
        // So d(state_{t-1})[i, j] -= beta_t * k_t[i] * d_delta_t[j].
        for (int idx = tid; idx < Dk * Dv; idx += nt) {
            int i = idx / Dv, j = idx - i * Dv;
            dstate[idx] = decay_t * dstate[idx] - beta_t * s_k[i] * s_d_delta[j];
        }
        __syncthreads();

        // No rollback needed — next iteration loads state from history.
    }

    // Final dstate is dstate_init.
    if (dstate_init) {
        for (int i = tid; i < Dk * Dv; i += nt) dstate_init[i] = dstate[i];
    }
}


// ===================== launchers =====================
// Multi-head dispatch: grid = (H,), block = (256,). Per-head pointers
// are pulled from the input tensors via stride arithmetic.

extern "C" cudaError_t launch_dn_fwd_with_delta_save(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *decay, const float *state_in,
    __nv_bfloat16 *y, float *state_out, __nv_bfloat16 *delta_save,
    float *state_history,                  // [H, S+1, Dk, Dv] or null
    int S, int H, cudaStream_t stream)
{
    constexpr int Dk = DN_KEY_DEFAULT;
    constexpr int Dv = DN_VAL_DEFAULT;
    size_t smem = (Dk * Dv + 2 * Dk + 3 * Dv) * sizeof(float);
    int threads = 256;
    cudaFuncSetAttribute(dn_fwd_with_delta_save_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int qkd_stride = H * Dk;
    int v_stride   = H * Dv;
    int bd_stride  = H;
    int hist_stride_t = Dk * Dv;
    dn_fwd_with_delta_save_kernel<<<H, threads, smem, stream>>>(
        q, k, v, beta, decay, state_in,
        y, state_out, delta_save, state_history,
        S, qkd_stride, v_stride, bd_stride, hist_stride_t);
    return cudaGetLastError();
}


extern "C" cudaError_t launch_dn_bwd(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *decay,
    const float *state_in,
    const __nv_bfloat16 *delta_save,
    const __nv_bfloat16 *dy,
    const float *state_history,           // [H, S+1, Dk, Dv]
    __nv_bfloat16 *dq, __nv_bfloat16 *dk, __nv_bfloat16 *dv,
    float *dbeta, float *ddecay, float *dstate_init,
    int S, int H, cudaStream_t stream)
{
    constexpr int Dk = DN_KEY_DEFAULT;
    constexpr int Dv = DN_VAL_DEFAULT;
    // 3 state slabs + 2 Dk-sized (s_q,s_k) +
    // 4 Dv-sized (s_v,s_dy,s_delta,s_d_delta) + 256-thread reduction.
    size_t smem = (3 * Dk * Dv + 2 * Dk + 4 * Dv + 256) * sizeof(float);
    int threads = 256;
    cudaFuncSetAttribute(dn_bwd_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int qkd_stride = H * Dk;
    int v_stride   = H * Dv;
    int bd_stride  = H;
    int hist_stride_t = Dk * Dv;
    dn_bwd_kernel<<<H, threads, smem, stream>>>(
        q, k, v, beta, decay, state_in,
        delta_save, dy, state_history,
        dq, dk, dv, dbeta, ddecay, dstate_init,
        S, qkd_stride, v_stride, bd_stride, hist_stride_t);
    return cudaGetLastError();
}
