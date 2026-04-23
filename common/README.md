# common/

Cross-cutting Python utilities shared across every entry under `models/`.

Currently a skeleton. As the repo grows past two models × three arches,
the following pieces move here:

- `dispatcher.py` — given a model name + device capability, returns the
  right backend import path. Each `models/<model>/model.py` becomes a
  thin wrapper that calls into this.
- `reference.py` — one HF-based reference implementation per model,
  used by every backend's correctness test.
- `bench.py` — unified bench runner that dumps JSON per
  (model, arch, commit), consumed by CI and by the per-cell docs in
  `docs/results/`.

Nothing in here should ever contain arch-specific CUDA — that all lives
under `models/<model>/backends/<arch>/`.
