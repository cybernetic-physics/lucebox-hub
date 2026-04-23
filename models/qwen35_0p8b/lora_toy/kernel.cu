/**
 * Fused single-kernel LoRA training step.
 *
 * One CUDA dispatch does: embedding lookup -> hidden @ output_weight.T
 * + LoRA residual -> log-softmax / CE loss -> backward to LoRA grads ->
 * AdamW update on lora_a and lora_b. Frozen embedding and output_weight
 * stay in device memory between steps.
 *
 * bf16 weights and hidden, fp32 accumulation, fp32 optimizer state.
 * cg::this_grid().sync() separates phases; the grid is cooperative.
 *
 * Target: NVIDIA B200 (sm_100). Runs on sm_80+ as well.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

#ifndef LORA_BLOCK_SIZE
#define LORA_BLOCK_SIZE 256
#endif

#ifndef LORA_NUM_BLOCKS
#define LORA_NUM_BLOCKS 128
#endif

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = LORA_BLOCK_SIZE / WARP_SIZE;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int off = WARP_SIZE / 2; off > 0; off /= 2)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int off = WARP_SIZE / 2; off > 0; off /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float *smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float *smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < NUM_WARPS) ? smem[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// ==============================================================================
// Phase A: embedding lookup. Each row t in [0, T) gets copied from
// embedding[context_tokens[t], :] into hidden[t, :].
// Blocks stride over rows; threads stride over H.
// ==============================================================================
__device__ void embed_rows(
    const __nv_bfloat16 *__restrict__ embedding,  // [V_emb, H]
    const int *__restrict__ context_tokens,        // [T]
    __nv_bfloat16 *__restrict__ hidden,            // [T, H]
    int T, int H)
{
    for (int t = blockIdx.x; t < T; t += gridDim.x) {
        int tok = context_tokens[t];
        const __nv_bfloat16 *src = embedding + (size_t)tok * H;
        __nv_bfloat16 *dst = hidden + (size_t)t * H;
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            dst[h] = src[h];
        }
    }
}

// ==============================================================================
// Phase B: lora_h[T, R] = hidden[T, H] @ lora_a[H, R].
// One block per row t. Each block loops threads across R output columns;
// for each column, reduces dot over H.
// ==============================================================================
__device__ void forward_lora_h(
    const __nv_bfloat16 *__restrict__ hidden,   // [T, H]
    const __nv_bfloat16 *__restrict__ lora_a,    // [H, R]
    float *__restrict__ lora_h,                   // [T, R]  (fp32)
    int T, int H, int R)
{
    __shared__ float smem[NUM_WARPS];
    for (int t = blockIdx.x; t < T; t += gridDim.x) {
        const __nv_bfloat16 *hrow = hidden + (size_t)t * H;
        for (int r = 0; r < R; r++) {
            float acc = 0.0f;
            for (int h = threadIdx.x; h < H; h += blockDim.x) {
                float x = __bfloat162float(hrow[h]);
                float a = __bfloat162float(lora_a[(size_t)h * R + r]);
                acc += x * a;
            }
            float sum = block_reduce_sum(acc, smem);
            if (threadIdx.x == 0) lora_h[(size_t)t * R + r] = sum;
            __syncthreads();
        }
    }
}

// ==============================================================================
// Phase C: per-row logits, log-softmax, CE loss, and grad_logits.
// Each block handles one row t. Computes:
//   base_logits[v] = sum_h hidden[t,h] * output_weight[v,h]         (frozen path)
//   lora_logits[v] = sum_r lora_h[t,r] * lora_b[r,v]                (LoRA path)
//   logits[v] = base + lora
//   lse = logsumexp(logits)
//   selected[t] = logits[target] - lse
//   loss_accum += -selected[t] * w[t]
//   grad_logits[t, v] = (w[t] / T) * (softmax[v] - onehot(v == target))
//
// Materialized buffers: logits_buf[T, V] (fp32) and grad_logits[T, V] (fp32).
// V can be large; we stream in tiles but write back every element since
// backward needs grad_logits. Per-row logits_buf not strictly needed beyond
// this phase — kept as a convenience for softmax pass.
// ==============================================================================
__device__ void forward_softmax_backward_setup(
    const __nv_bfloat16 *__restrict__ hidden,       // [T, H]
    const float *__restrict__ lora_h,                // [T, R]
    const __nv_bfloat16 *__restrict__ output_weight, // [V, H]
    const __nv_bfloat16 *__restrict__ lora_b,        // [R, V]
    const int *__restrict__ target_tokens,            // [T]
    const float *__restrict__ weights,                // [T]
    float *__restrict__ logits_buf,                   // [T, V]
    float *__restrict__ grad_logits,                  // [T, V]
    float *__restrict__ selected_out,                 // [T]
    float *__restrict__ loss_accum,                   // [1]  atomic
    int T, int H, int V, int R)
{
    __shared__ float smem[NUM_WARPS];

    for (int t = blockIdx.x; t < T; t += gridDim.x) {
        const __nv_bfloat16 *hrow = hidden + (size_t)t * H;
        const float *lrow = lora_h + (size_t)t * R;
        int target = target_tokens[t];
        float w_t = weights[t];
        float inv_T = 1.0f / (float)T;

        // Pass 1: compute logits and row max.
        float row_max = -INFINITY;
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            // Base logit: hidden[t] . output_weight[v]
            float base = 0.0f;
            const __nv_bfloat16 *wrow = output_weight + (size_t)v * H;
            for (int h = 0; h < H; h++) {
                base += __bfloat162float(hrow[h]) * __bfloat162float(wrow[h]);
            }
            // LoRA logit: lora_h[t] . lora_b[:, v]
            float lora = 0.0f;
            for (int r = 0; r < R; r++) {
                lora += lrow[r] * __bfloat162float(lora_b[(size_t)r * V + v]);
            }
            float z = base + lora;
            logits_buf[(size_t)t * V + v] = z;
            if (z > row_max) row_max = z;
        }
        row_max = block_reduce_max(row_max, smem);

        // Pass 2: compute denominator of softmax.
        float denom = 0.0f;
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            float z = logits_buf[(size_t)t * V + v];
            denom += __expf(z - row_max);
        }
        denom = block_reduce_sum(denom, smem);
        float lse = row_max + __logf(denom);
        float inv_denom = 1.0f / denom;

        // Pass 3: write grad_logits and grab selected log-prob.
        // grad = (w/T) * (softmax - onehot(target))
        float scale = w_t * inv_T;
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            float z = logits_buf[(size_t)t * V + v];
            float p = __expf(z - row_max) * inv_denom;
            float g = scale * p;
            if (v == target) g -= scale;
            grad_logits[(size_t)t * V + v] = g;
        }

        if (threadIdx.x == 0) {
            float sel = logits_buf[(size_t)t * V + target] - lse;
            selected_out[t] = sel;
            atomicAdd(loss_accum, -sel * w_t * inv_T);
        }
        __syncthreads();
    }
}

// ==============================================================================
// Phase D: grad_lora_b[r, v] = sum_t lora_h[t, r] * grad_logits[t, v].
// Parallelize over (r, v). Each thread owns one output element and loops
// over T. Blocks tile over the R*V output grid.
// ==============================================================================
__device__ void backward_lora_b(
    const float *__restrict__ lora_h,       // [T, R]
    const float *__restrict__ grad_logits,   // [T, V]
    float *__restrict__ grad_lora_b,          // [R, V]
    int T, int R, int V)
{
    int total = R * V;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int r = idx / V;
        int v = idx % V;
        float acc = 0.0f;
        for (int t = 0; t < T; t++) {
            acc += lora_h[(size_t)t * R + r] * grad_logits[(size_t)t * V + v];
        }
        grad_lora_b[(size_t)r * V + v] = acc;
    }
}

// ==============================================================================
// Phase E: grad_lora_h[t, r] = sum_v grad_logits[t, v] * lora_b[r, v].
// One block per (t, r_tile). Small output [T, R].
// ==============================================================================
__device__ void backward_lora_h(
    const float *__restrict__ grad_logits,      // [T, V]
    const __nv_bfloat16 *__restrict__ lora_b,    // [R, V]
    float *__restrict__ grad_lora_h,              // [T, R]
    int T, int V, int R)
{
    __shared__ float smem[NUM_WARPS];
    int total_rows = T * R;
    for (int idx = blockIdx.x; idx < total_rows; idx += gridDim.x) {
        int t = idx / R;
        int r = idx % R;
        const float *grow = grad_logits + (size_t)t * V;
        const __nv_bfloat16 *brow = lora_b + (size_t)r * V;
        float acc = 0.0f;
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            acc += grow[v] * __bfloat162float(brow[v]);
        }
        float sum = block_reduce_sum(acc, smem);
        if (threadIdx.x == 0) grad_lora_h[(size_t)t * R + r] = sum;
        __syncthreads();
    }
}

// ==============================================================================
// Phase F: grad_lora_a[h, r] = sum_t hidden[t, h] * grad_lora_h[t, r].
// Parallelize over (h, r). Each thread owns one output element, loops over T.
// ==============================================================================
__device__ void backward_lora_a(
    const __nv_bfloat16 *__restrict__ hidden,   // [T, H]
    const float *__restrict__ grad_lora_h,       // [T, R]
    float *__restrict__ grad_lora_a,              // [H, R]
    int T, int H, int R)
{
    int total = H * R;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int h = idx / R;
        int r = idx % R;
        float acc = 0.0f;
        for (int t = 0; t < T; t++) {
            acc += __bfloat162float(hidden[(size_t)t * H + h]) * grad_lora_h[(size_t)t * R + r];
        }
        grad_lora_a[(size_t)h * R + r] = acc;
    }
}

// ==============================================================================
// Phase G: AdamW update. Element-wise over lora_a [H, R] and lora_b [R, V].
// m = beta1*m + (1-beta1)*g
// v = beta2*v + (1-beta2)*g^2
// m_hat = m / (1 - beta1^step);  v_hat = v / (1 - beta2^step)
// p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
// Note: `step` is the 1-indexed Adam step *after* increment.
// ==============================================================================
__device__ void adamw_update(
    __nv_bfloat16 *__restrict__ param,    // bf16 parameter (updated in place)
    float *__restrict__ m,                 // fp32 state
    float *__restrict__ v,                 // fp32 state
    const float *__restrict__ grad,        // fp32 grad
    int numel,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < numel; i += stride) {
        float g = grad[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;
        float p = __bfloat162float(param[i]);
        p -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * p);
        param[i] = __float2bfloat16(p);
    }
}

// ==============================================================================
// Megakernel.
// ==============================================================================
extern "C" __global__ void lora_train_kernel(
    // Tokens / weights
    const int *__restrict__ context_tokens,     // [T]
    const int *__restrict__ target_tokens,       // [T]
    const float *__restrict__ token_weights,     // [T]
    // Frozen params
    const __nv_bfloat16 *__restrict__ embedding,     // [V_emb, H]
    const __nv_bfloat16 *__restrict__ output_weight, // [V, H]
    // Trainable params
    __nv_bfloat16 *__restrict__ lora_a,   // [H, R]
    __nv_bfloat16 *__restrict__ lora_b,   // [R, V]
    // Adam state
    float *__restrict__ m_a, float *__restrict__ v_a,   // [H, R]
    float *__restrict__ m_b, float *__restrict__ v_b,   // [R, V]
    // Workspace
    __nv_bfloat16 *__restrict__ hidden,   // [T, H]
    float *__restrict__ lora_h,           // [T, R]
    float *__restrict__ logits_buf,       // [T, V]
    float *__restrict__ grad_logits,      // [T, V]
    float *__restrict__ grad_lora_a,      // [H, R]
    float *__restrict__ grad_lora_b,      // [R, V]
    float *__restrict__ grad_lora_h,      // [T, R]
    // Outputs
    float *__restrict__ selected_out,     // [T]
    float *__restrict__ loss_out,         // [1]
    // Shapes
    int T, int H, int V, int R,
    // Optimizer hyperparams
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    int do_update)
{
    cg::grid_group grid = cg::this_grid();

    // Zero loss accumulator in block 0.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *loss_out = 0.0f;
    }
    grid.sync();

    embed_rows(embedding, context_tokens, hidden, T, H);
    grid.sync();

    forward_lora_h(hidden, lora_a, lora_h, T, H, R);
    grid.sync();

    forward_softmax_backward_setup(
        hidden, lora_h, output_weight, lora_b,
        target_tokens, token_weights,
        logits_buf, grad_logits, selected_out, loss_out,
        T, H, V, R);
    grid.sync();

    backward_lora_b(lora_h, grad_logits, grad_lora_b, T, R, V);
    grid.sync();

    backward_lora_h(grad_logits, lora_b, grad_lora_h, T, V, R);
    grid.sync();

    backward_lora_a(hidden, grad_lora_h, grad_lora_a, T, H, R);
    grid.sync();

    if (do_update) {
        adamw_update(lora_a, m_a, v_a, grad_lora_a, H * R,
                     lr, beta1, beta2, eps, wd,
                     bias_correction1, bias_correction2);
        adamw_update(lora_b, m_b, v_b, grad_lora_b, R * V,
                     lr, beta1, beta2, eps, wd,
                     bias_correction1, bias_correction2);
    }
}

// ==============================================================================
// Host-side launch wrapper.
// ==============================================================================
extern "C" void launch_lora_train(
    const int *context_tokens, const int *target_tokens, const float *token_weights,
    const void *embedding, const void *output_weight,
    void *lora_a, void *lora_b,
    float *m_a, float *v_a, float *m_b, float *v_b,
    void *hidden, float *lora_h, float *logits_buf, float *grad_logits,
    float *grad_lora_a, float *grad_lora_b, float *grad_lora_h,
    float *selected_out, float *loss_out,
    int T, int H, int V, int R,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    int do_update,
    cudaStream_t stream)
{
    dim3 grid(LORA_NUM_BLOCKS);
    dim3 block(LORA_BLOCK_SIZE);

    void *args[] = {
        (void *)&context_tokens, (void *)&target_tokens, (void *)&token_weights,
        (void *)&embedding, (void *)&output_weight,
        (void *)&lora_a, (void *)&lora_b,
        (void *)&m_a, (void *)&v_a, (void *)&m_b, (void *)&v_b,
        (void *)&hidden, (void *)&lora_h, (void *)&logits_buf, (void *)&grad_logits,
        (void *)&grad_lora_a, (void *)&grad_lora_b, (void *)&grad_lora_h,
        (void *)&selected_out, (void *)&loss_out,
        (void *)&T, (void *)&H, (void *)&V, (void *)&R,
        (void *)&lr, (void *)&beta1, (void *)&beta2, (void *)&eps, (void *)&wd,
        (void *)&bias_correction1, (void *)&bias_correction2,
        (void *)&do_update,
    };

    cudaLaunchCooperativeKernel(
        (void *)lora_train_kernel, grid, block, args, 0, stream);
}
