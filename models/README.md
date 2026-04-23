# models/

One subdirectory per model pair we support.

| Dir                  | Model             | Backends today                         |
|----------------------|-------------------|----------------------------------------|
| [`qwen35_0p8b/`](qwen35_0p8b/) | Qwen3.5-0.8B      | BF16 (sm_86, sm_100), NVFP4 (sm_121a/sm_100) |
| [`qwen35_27b/`](qwen35_27b/)   | Qwen3.5-27B       | DFlash+DDTree Q4_K_M (sm_86, sm_100)   |

Layout convention (work in progress — today both models keep their
flat layout from the earlier `megakernel/` / `dflash/` dirs):

```
models/<model>/
├── README.md              · model-specific narrative
├── RESULTS.md             · original (3090-era) results
├── backends/              · one dir per (arch, backend) pair (planned)
│   ├── sm_86_bf16/
│   ├── sm_100_bf16/
│   └── sm_121_nvfp4/
├── model.py               · Python entry point (picks the backend)
└── setup.py               · build the local device's backend
```

Per-arch benchmark reports live under `docs/results/`, one MD per
(model, arch) cell. Per-arch roadmaps under `docs/roadmap/`.

Each model/arch directory is its **own compilation unit**. No
`#include` across backends. No `#ifdef __CUDA_ARCH__` branches between
arch variants. This is the trick that lets the RTX 3090 BF16 kernel
and the B200 V-split kernel coexist without one edit breaking the
other.
