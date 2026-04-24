/**
 * Chunked DeltaNet backward (BPTT) for Qwen3.5-0.8B linear-attention
 * layers. DeltaNet is not in CUTLASS — this kernel is custom.
 *
 * Algorithm (matches the fla library's chunked gated-delta-net):
 *   Partition S into chunks of C tokens (C = 64 by default).
 *   Within a chunk, represent the recurrence as a matrix equation so
 *     the Dk × Dk × C inner workload maps to a bf16 WGMMA.
 *   Between chunks, a prefix-scan over [H, Dk, Dv] states propagates
 *     history. On backward, walk chunks in reverse.
 *
 * Per-chunk work (bf16 WGMMA):
 *   Forward, within chunk c spanning t ∈ [c*C, c*C+C):
 *     attn_inchunk = K_c @ state + TRIL(K_c @ K_c.T, diag = -inf)
 *                                          (intra-chunk causal)
 *     where K_c, V_c ∈ [C, Dk], [C, Dv], and state is carried from c-1.
 *
 * This file is a SCAFFOLD — the kernel body is a TODO. The launcher
 * signature is stable so the Python trainer can bind against it.
 *
 * TODO:
 *   1. Port fla's `chunked_delta_bwd` (Triton) to native CUDA C++
 *      with WGMMA on bf16 (sm_100).
 *   2. Save intra-chunk states during forward (~ S/C states of size
 *      H*Dk*Dv each; ~MB per layer).
 *   3. Reverse-walk chunks computing dQ, dK, dV, dbeta, ddecay.
 *   4. Expose as extern "C" for torch binding.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" cudaError_t cutlass_deltanet_bwd_sm100(
    const __nv_bfloat16 *q,         // [S, H, Dk] post conv/silu/norm
    const __nv_bfloat16 *k,         // [S, H, Dk]
    const __nv_bfloat16 *v,         // [S, H, Dv]
    const float *beta,              // [S, H] sigmoid
    const float *decay,             // [S, H]
    const float *state_saves,       // [N_chunks, H, Dk, Dv] fp32 saved intra-chunk states
    const __nv_bfloat16 *dy,        // [S, H, Dv] grad of recurrence output
    __nv_bfloat16 *dq,              // [S, H, Dk]
    __nv_bfloat16 *dk,              // [S, H, Dk]
    __nv_bfloat16 *dv,              // [S, H, Dv]
    float *dbeta,                   // [S, H]
    float *ddecay,                  // [S, H]
    float *dstate_init,             // [H, Dk, Dv]
    int S, int H, int Dk, int Dv,
    int chunk_size,                 // C
    cudaStream_t stream)
{
    // TODO: chunked BPTT with bf16 WGMMA per chunk.
    (void)q; (void)k; (void)v; (void)beta; (void)decay; (void)state_saves;
    (void)dy; (void)dq; (void)dk; (void)dv; (void)dbeta; (void)ddecay;
    (void)dstate_init;
    (void)S; (void)H; (void)Dk; (void)Dv; (void)chunk_size; (void)stream;
    return cudaErrorNotSupported;
}
