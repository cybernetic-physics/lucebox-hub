# S3 — NVFP4 KV cache wireup

The 0.8B already has working NVFP4 KV helpers at
`models/qwen35_0p8b/nvfp4_kv.cuh`. Head_dim is hardcoded to 256 there
— **same value as 27B**, so the same helpers apply. This page outlines
the wireup for 27B.

## Storage layout (re-uses the 0.8B layout exactly)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| k_cache_data | [N_FA, KV_H, max_seq, HEAD_DIM/2 = 128] | uint8 | packed FP4x2 |
| k_cache_scales | [N_FA, KV_H, max_seq, HEAD_DIM/16 = 16] | uint8 | E4M3 FP8 |
| v_cache_data | same | uint8 | |
| v_cache_scales | same | uint8 | |

Block size 16. At max_seq=32k, FA layers=16, KV heads=4:
- bf16:  16 * 4 * 32768 * 256 * 2 = **1.0 GB per cache**, K+V = 2.0 GB
- nvfp4: 16 * 4 * 32768 * (128 + 16) = **0.28 GB per cache**, K+V = 0.56 GB (3.5×)

At max_seq=262144 (native):
- bf16: **8 GB per cache**, K+V = 16 GB
- nvfp4: **2.3 GB per cache**, K+V = 4.6 GB (3.5×)

## Kernel-side change (fa_layer.cuh)

Add a `Cfg::USE_NVFP4_KV` trait (or `bool USE_NVFP4_KV` template on
decode_kernel_impl). In the FA layer's K/V cache write site:

```cpp
// Currently (bf16 path):
__nv_bfloat16 *kc = k_cache + (size_t)h * max_seq * D + (size_t)position * D;
for (int i = lane_id; i < D; i += WARP_SIZE) {
    kc[i] = __float2bfloat16(kh[i]);
}

// New (nvfp4 path):
uint8_t *kc_data   = k_cache_data   + (size_t)h * max_seq * (D/2)  + (size_t)position * (D/2);
uint8_t *kc_scales = k_cache_scales + (size_t)h * max_seq * (D/16) + (size_t)position * (D/16);
nvfp4_kv::pack_to_nvfp4(kh, kc_data, kc_scales);  // 0.8B helper, head_dim=256
```

In the FA scan loop's K/V read site:

```cpp
// Currently:
const __nv_bfloat16 *k_p = k_cache + ...;
score += q_local[e] * __bfloat162float(__ldg(k_p + lane_id * EPL + e));

// New:
float k_local[D];  // dequant target — or unpack into shmem
nvfp4_kv::unpack_to_float(kc_data, kc_scales, k_local);
score += q_local[e] * k_local[lane_id * EPL + e];
```

The scan structure stays the same; just the per-position read costs an
unpack.

## Python-side change (weight_packer.py)

Add an `alloc_nvfp4_kv_scratch` that returns the 4 packed buffers
instead of 2 bf16 caches. Wire as alternative `Scratch` fields.

## Acceptance

- Reuse C7 wikitext sweep with `--backend nvfp4-kv` (new flag). Top-1
  match vs HF on S ≤ 256.
- KV memory at max_seq=32768 drops from 2 GB → 0.56 GB.
- Wall-clock per decode step within 1.2× of the bf16-KV path (the
  pack/unpack overhead is small vs HBM-bound BF16 reads).

## Effort

~2-3 days (kernel write/unpack at two sites + new launchers +
runtime swap + correctness test). Smaller than S2 because the math
doesn't change.

## See also

- 0.8B helpers: `models/qwen35_0p8b/nvfp4_kv.cuh`
- 0.8B reference roundtrip: `models/qwen35_0p8b/trainer/test_nvfp4_kv_roundtrip.py`
