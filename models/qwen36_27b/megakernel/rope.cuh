/**
 * RoPE with YaRN scaling + MRoPE-interleaved sections.
 *
 * Standard RoPE: rotate halves of each head_dim/2 pair by `position * freq_i`
 * where freq_i = theta^(-2i / rotary_dim).
 *
 * YaRN (Qwen3.6 extension): for context > native, blend two frequency
 * regimes:
 *   - Low frequencies (small i, "long-wavelength") get extrapolated, so
 *     wavelengths > native_context get rescaled by scale_factor.
 *   - High frequencies (large i, "short-wavelength") get interpolated
 *     (compressed to fit the extended range).
 *   - A NTK-aware ramp blends between the two over the freq spectrum.
 *
 * MRoPE-interleaved (Qwen3.6 multimodal): the per-token position is a
 * 3-tuple (t, h, w) for (temporal, height, width). The rotary_dim is
 * split into 3 sections, one per axis. For text-only path the model
 * sets h = w = 0 so the contribution from the height/width sections
 * is zero, but the SECTION BOUNDARIES still split the angle vector.
 *
 * For Qwen3.6-27B:
 *   FA_ROTARY_DIM = 64    (head_dim 256, partial rotary 0.25)
 *   MRoPE sections = [11, 11, 10]   sums to 32 = rotary_dim/2 freq pairs
 *
 * This header provides:
 *   compute_rope_freqs<Cfg>(out_inv_freq) -- write base inv_freqs to shmem
 *   apply_yarn_scaling<Cfg>(freqs, position, ctx_len) -- blend
 *   rope_apply<Cfg>(qk_head, position, freqs)  -- rotate one head in-place
 */
#pragma once

#include "Cfg.cuh"
#include "helpers.cuh"

namespace lucebox::qwen3x {

#ifndef WARP_SIZE_DEFINED
#define WARP_SIZE_DEFINED
constexpr int WARP_SIZE_ROPE = 32;
#endif

// MRoPE sections per the Qwen3.6 config — [t, h, w]. Sum must equal
// FA_ROTARY_DIM / 2 (32 for both 0.8B and 27B). 0.8B doesn't use MRoPE
// (text-only model), but giving it the same section layout doesn't
// change text-only behavior because h = w = 0.
struct MRopeSections {
    int t, h, w;
    constexpr int sum() const { return t + h + w; }
};
constexpr MRopeSections MROPE_QWEN36 = {11, 11, 10};   // 11+11+10 = 32

// Compute the inv_freq table for the model. Writes
// `Cfg::FA_ROTARY_DIM / 2` floats into `out_inv_freq[]`.
// Standard formula: inv_freq[i] = theta^(-2i / rotary_dim).
template<typename Cfg>
__device__ void compute_base_inv_freq(float *__restrict__ out_inv_freq) {
    constexpr int R   = Cfg::FA_ROTARY_DIM;
    constexpr int N   = R / 2;
    constexpr float T = Cfg::FA_ROPE_THETA;
    int tid = threadIdx.x;
    if (tid < N) {
        float exp_i = float(2 * tid) / float(R);
        float v;
        // theta^(-exp_i) = exp(-exp_i * ln(theta))
        // Use double for the exponent then truncate; happens once per
        // launch so we don't care about throughput.
        out_inv_freq[tid] = powf(T, -exp_i);
    }
}

// YaRN scaling parameters. Qwen3.6 default config exposes
// `rope_scaling.factor` (e.g. 4.0 for 4x context extension) and
// optional alpha/beta ramps. This struct captures the math; in
// practice the loader populates it from rope_scaling in config.json.
struct YarnParams {
    float scale_factor;       // e.g. 4.0 for native 262k -> 1M effective
    float beta_fast;          // freq threshold above which we extrapolate (default 32)
    float beta_slow;          // freq threshold below which we interpolate (default 1)
    int   original_ctx_len;   // 262144 for 27B
    bool  enabled;
    constexpr YarnParams()
        : scale_factor(1.0f), beta_fast(32.0f), beta_slow(1.0f),
          original_ctx_len(262144), enabled(false) {}
};

// Apply YaRN to a single inv_freq value. Returns the blended frequency
// for position `pos`. See NTK-aware "by parts" derivation; we follow
// the implementation in HF transformers' modeling_qwen3.py.
__device__ __forceinline__ float yarn_blend(
    float inv_freq, float pos, const YarnParams &yp)
{
    if (!yp.enabled || yp.scale_factor <= 1.0f) return inv_freq * pos;

    // Wavelength of this frequency:
    //   lambda_i = 2*pi / inv_freq_i
    // Extrapolation factor = 1 (use as-is).
    // Interpolation factor = 1 / scale_factor.
    // Ramp between them across [beta_slow, beta_fast] in (orig_ctx /
    // (2*pi)) wavelengths.
    float lam = 6.283185307179586f / inv_freq;
    float r   = float(yp.original_ctx_len) / lam;
    float t   = (r - yp.beta_slow) / (yp.beta_fast - yp.beta_slow);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    // Extrapolated: pos * inv_freq.
    // Interpolated: pos * inv_freq / scale_factor.
    // Blend: t * extrap + (1 - t) * interp.
    float extrap = inv_freq;
    float interp = inv_freq / yp.scale_factor;
    float blended = t * extrap + (1.0f - t) * interp;
    return blended * pos;
}

// Apply RoPE rotation in-place to one head in fp32.
//   head: [rotary_dim] f32, gets rotated; remaining elements untouched
//   pos: integer position
//   inv_freq: precomputed inv_freq table of size rotary_dim/2
//   yp: YaRN params (disabled => standard RoPE)
//   sections: MRoPE [t, h, w] sizes summing to rotary_dim/2; pos_h/pos_w
//             are 0 for text-only.
template<typename Cfg>
__device__ void rope_apply(
    float *__restrict__ head,
    int pos_t, int pos_h, int pos_w,
    const float *__restrict__ inv_freq,
    const YarnParams &yp,
    const MRopeSections &sections,
    int lane_id)
{
    constexpr int R    = Cfg::FA_ROTARY_DIM;
    constexpr int HALF = R / 2;

    // For pair index i (0..HALF-1), pick the position dimension based on
    // which MRoPE section it lives in.
    for (int i = lane_id; i < HALF; i += WARP_SIZE_ROPE) {
        int p;
        if      (i < sections.t)                          p = pos_t;
        else if (i < sections.t + sections.h)             p = pos_h;
        else                                              p = pos_w;
        float phase = yarn_blend(inv_freq[i], float(p), yp);
        float c, s;
        __sincosf(phase, &s, &c);
        // Rotate pair (head[i], head[i + HALF]):
        //   a' = a * c - b * s
        //   b' = a * s + b * c
        float a = head[i];
        float b = head[i + HALF];
        head[i]        = a * c - b * s;
        head[i + HALF] = a * s + b * c;
    }
}

}  // namespace lucebox::qwen3x
