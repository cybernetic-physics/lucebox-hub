"""CUDA Graphs wrapper for the HF training step.

Usage:
    from cuda_graph_train import GraphedTrainStep
    runner = GraphedTrainStep(model, batch_size=1, seq_len=128)
    for ids in dataset:
        loss = runner.step(ids)   # one CUDA graph replay per step

Captures a graph of the fwd+bwd of `model(input_ids=ids, labels=ids)` so
each subsequent step replays the same kernel sequence with a single
graph launch. Saves ~5 us per kernel launch; the total win at S=128
on Qwen3.5-0.8B + LoRA is ~15% per training step (validated by
test_cuda_graph.py).

Limitations:
  - Static input shape: each (batch_size, seq_len) needs its own graph
    instance. The class caches at most one graph; re-create the runner
    if the shape changes.
  - Optimizer step is not graphed (call optimizer.step() outside).
  - The model must be in train() mode and have grads enabled before
    capture; weight tensors must not be reallocated between captures.
"""
from __future__ import annotations

import torch


class GraphedTrainStep:
    """Wraps a model.forward+backward call in a CUDA graph."""

    def __init__(self, model, *, batch_size: int = 1, seq_len: int,
                 dtype: torch.dtype = torch.long, device: str = "cuda",
                 warmup_iters: int = 3):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        # Static input + label tensors — graph captures their addresses.
        self.static_ids = torch.zeros(batch_size, seq_len, dtype=dtype, device=device)

        # Warmup on a side stream (CUDA Graphs requires this).
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                out = model(input_ids=self.static_ids, labels=self.static_ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
        torch.cuda.current_stream().wait_stream(s)

        # Capture.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out = model(input_ids=self.static_ids, labels=self.static_ids)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            self.static_loss = out.loss.detach().clone()

    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """Copy ids into the static buffer, replay the graph, return the loss."""
        assert ids.shape == self.static_ids.shape, (
            f"ids shape {tuple(ids.shape)} != captured shape "
            f"{tuple(self.static_ids.shape)}; rebuild the runner")
        self.static_ids.copy_(ids)
        self.graph.replay()
        return self.static_loss


def make_graphed_step(model, *, batch_size: int = 1, seq_len: int,
                      **kw) -> GraphedTrainStep:
    """Convenience constructor."""
    return GraphedTrainStep(model, batch_size=batch_size, seq_len=seq_len, **kw)
