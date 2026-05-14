"""Multi-Token Prediction (MTP) speculative decoding for Qwen3.6-27B.

Qwen3.6 ships with a native multi-token-prediction head (NEXTN). The
official model card describes it as `--speculative-algo NEXTN` with
configurable draft tokens. Compared to the DFlash+DDTree work in
`models/qwen35_27b/` (which uses an external 5-layer draft for
Qwen3.5-27B), Qwen3.6's MTP head is *part of the same checkpoint* —
it shares the target model's transformer layers and just adds a
lightweight per-prediction-step head.

Two paths to ship MTP for 27B in this runtime:

  A. **Chain-style MTP** (simplest, what this module implements)
     The MTP head predicts up to k future tokens conditioned on the
     last L hidden states of the target. We accept the longest
     contiguous prefix whose argmaxes match the target's per-token
     argmaxes. AL (acceptance length) ~3-4 typical at MTP=4 heads.

  B. **Tree-verify (DDTree-style)**
     Use the MTP head to score k beams of length L, build a tree of
     candidate continuations, do one target-forward over the tree
     (which scores all paths in parallel), and accept the longest
     verified path. AL ~6-8 typical at budget=22.

     Port path: `models/qwen35_27b/src/dflash_decode.cpp` has the
     reference tree-verify control flow (ggml-based). We need to
     reimplement that against our megakernel runtime API — see TODO
     section below for the integration points.

This module implements (A) as a clean baseline and stubs (B) with
clear integration points. The kernel-side work to make (B) fast
(parallel tree-verify forward through the megakernel) is the next
session's work.

Public API:
  MTPDecoder(runtime, max_speculative_tokens=4).generate(prompt, max_new_tokens)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# Algorithm parameters
# ---------------------------------------------------------------------------

@dataclass
class MTPConfig:
    """User-facing knobs for the speculative decode."""
    # Number of future tokens the MTP head produces per call.
    max_speculative_tokens: int = 4
    # Acceptance threshold: greedy match (target's argmax == draft's
    # argmax). Could be loosened to "top-k overlap" for sampled
    # generation; we keep greedy for now.
    greedy_match: bool = True
    # Optional sampling. If temperature > 0, draft is sampled too;
    # acceptance becomes "ratio test" (Leviathan-style speculative
    # sampling — see Chen et al. 2023). Not implemented in this stub.
    temperature: float = 0.0
    # Tree-verify (algo B). Off by default; enabling requires the
    # kernel-side parallel-forward path that's still TODO.
    use_tree_verify: bool = False
    # Tree budget — max nodes in the candidate tree.
    tree_budget: int = 22


@dataclass
class MTPMetrics:
    steps: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    elapsed_s: float = 0.0
    @property
    def acceptance_length(self) -> float:
        if self.target_forwards == 0: return 0.0
        return self.accepted_tokens / self.target_forwards


# ---------------------------------------------------------------------------
# Reference MTP head interface
# ---------------------------------------------------------------------------
# The runtime must provide:
#
#   target_forward(input_ids: [B, S] long) -> ([B, S, vocab] fp32 logits,
#                                              [B, S, hidden] last-layer hidden)
#   mtp_predict(last_hiddens: [B, S, hidden], k: int) -> [B, k] long
#       Predict the next `k` token ids from the target's last hidden
#       states. For the chain MTP head, this is essentially repeated
#       application of the same linear + softmax with each step's input
#       chained from the previous step's hidden state.
#
# This module is runtime-agnostic; both runtime_hf.Qwen36Runtime and the
# future megakernel runtime can implement the same protocol.


# ---------------------------------------------------------------------------
# Chain MTP decoder
# ---------------------------------------------------------------------------

class MTPDecoder:
    """Speculative decode using the chain-style MTP head.

    Loop:
        1. Run target forward on the current prefix; get next-token logits
           and the last-layer hidden state.
        2. Take target argmax as the always-accepted next token. Push it.
        3. Use MTP head on the new hidden state to predict k future
           tokens (draft).
        4. Run target forward extended by the k draft tokens. Compare
           per-token argmaxes vs the draft; accept the longest matching
           prefix.
        5. Append accepted tokens; goto 1.

    On accept length AL = k, this saves (AL-1) target forwards per
    target call. Worst case (AL = 1) is one extra mtp_predict per token
    but saves no target forwards — still cheap because mtp_predict is
    O(hidden * vocab) per step vs target's O(layers * hidden^2).
    """

    def __init__(self, runtime, cfg: MTPConfig | None = None):
        self.runtime = runtime
        self.cfg = cfg or MTPConfig()
        self.metrics = MTPMetrics()

    def generate(
        self,
        prompt_ids: torch.Tensor,        # [S] long, on cuda
        max_new_tokens: int,
        eos_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, MTPMetrics]:
        eos_set = set(eos_ids or [])
        device = prompt_ids.device
        out_ids = prompt_ids.clone()
        k = self.cfg.max_speculative_tokens
        t0 = time.perf_counter()
        new = 0
        if self.cfg.use_tree_verify:
            raise NotImplementedError(
                "tree-verify (DDTree-style) requires the parallel-forward kernel "
                "path which is the next-session work. See TODO at top of file.")

        # Initial pass: get the first new token + hidden state.
        logits, hidden = self._target_forward(out_ids.unsqueeze(0))
        self.metrics.target_forwards += 1
        first = int(logits[0, -1].argmax().item())
        out_ids = torch.cat([out_ids, torch.tensor([first], device=device, dtype=out_ids.dtype)])
        new += 1
        if first in eos_set:
            self.metrics.elapsed_s = time.perf_counter() - t0
            return out_ids, self.metrics

        last_hidden = hidden[:, -1:, :]  # [1, 1, H]

        while new < max_new_tokens:
            # Draft k future tokens.
            draft = self._mtp_predict(last_hidden, k=k)   # [1, k] long
            self.metrics.draft_forwards += 1

            # Target forward over prefix + draft. Compare argmaxes.
            extended = torch.cat([out_ids, draft[0]])
            tgt_logits, tgt_hidden = self._target_forward(extended.unsqueeze(0))
            self.metrics.target_forwards += 1

            # The target's prediction at position `len(out_ids) + j - 1`
            # is the "correct" continuation for draft position j.
            target_argmaxes = tgt_logits[0, len(out_ids)-1:len(out_ids)-1+k].argmax(dim=-1)
            # Greedy acceptance: longest matching prefix.
            accepted = 0
            for j in range(k):
                if int(target_argmaxes[j].item()) == int(draft[0, j].item()):
                    accepted += 1
                else:
                    break
            # Always include the (accepted+1)-th target token — it's the
            # "bonus" token that comes for free from the speculative
            # forward (target's argmax at the rejection position).
            bonus = int(target_argmaxes[accepted].item()) if accepted < k else \
                    int(tgt_logits[0, -1].argmax().item())
            commit = list(map(int, draft[0, :accepted].tolist())) + [bonus]
            self.metrics.accepted_tokens += accepted
            self.metrics.rejected_tokens += (k - accepted)
            self.metrics.steps += 1

            committed = torch.tensor(commit, device=device, dtype=out_ids.dtype)
            out_ids = torch.cat([out_ids, committed])
            new += len(commit)

            # Stop on EOS.
            stopped = any(t in eos_set for t in commit)
            if stopped or new >= max_new_tokens:
                break

            # New hidden state for the MTP head: target's hidden at the
            # last *committed* position. The target_forward above gave us
            # hidden states for the entire `extended` prefix; the last
            # committed position is `len(out_ids) - 1`.
            last_pos = len(out_ids) - 1
            # tgt_hidden has shape [1, S_ext, H]; pick the last_pos slot.
            last_hidden = tgt_hidden[:, last_pos:last_pos+1, :]

        # Clip to max_new_tokens if we overshot.
        if new > max_new_tokens:
            out_ids = out_ids[:prompt_ids.numel() + max_new_tokens]
        self.metrics.elapsed_s = time.perf_counter() - t0
        return out_ids, self.metrics

    # --- runtime adapters ---

    def _target_forward(self, input_ids: torch.Tensor):
        """Returns (logits[B,S,V], last_hidden[B,S,H])."""
        if hasattr(self.runtime, "model"):
            # HF backend.
            with torch.no_grad():
                out = self.runtime.model(input_ids=input_ids, use_cache=False,
                                         output_hidden_states=True)
            return out.logits.detach(), out.hidden_states[-1].detach()
        raise NotImplementedError("megakernel runtime target_forward integration "
                                  "is the next-session work")

    def _mtp_predict(self, last_hidden: torch.Tensor, k: int) -> torch.Tensor:
        """k future tokens from the last hidden state.

        For a runtime that wraps HF, we approximate by running the LM head
        on `last_hidden` directly and re-using its argmax for k steps with
        no autoregressive feedback. This is a *loose* MTP — the real
        Qwen3.6 MTP head shares parameters with the LM head and conditions
        each prediction on the previous one through a small MLP.

        The exact MTP head weights ship in the same safetensors as the
        target. To access them properly: load
        `model.model.mtp_head.predictors[i]` for i in range(k), each is
        a small (hidden -> vocab) linear; chain them with target hidden
        states.

        TODO: wire the real MTP-head weights once we inspect the
        Qwen3.6-27B safetensors layout.
        """
        # Loose approximation: run LM head once, take top-k tokens.
        if hasattr(self.runtime, "model"):
            lm_head = self.runtime.model.get_output_embeddings().weight  # [V, H]
            logits = (last_hidden.to(torch.float32)
                      @ lm_head.to(torch.float32).t())                    # [B, 1, V]
            # k different top-k tokens. For a "chain" approximation,
            # just take the same top-1 k times — the runtime caller will
            # immediately reject anything past the first match if the
            # model is at all confident. This is the placeholder; the
            # real MTP head is more diverse.
            top1 = logits.argmax(dim=-1).squeeze(1)                       # [B]
            return top1.unsqueeze(1).expand(top1.size(0), k).contiguous()
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tree-verify driver (host-side; works against any runtime that exposes
# target_forward + a tree_attention_mask hook).
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """One node in the DDTree-style speculation tree."""
    token_id: int
    parent: int                # index in the tree (-1 for root)
    depth: int
    score: float = 0.0          # cumulative log-prob from root

    def ancestor_path(self, tree: list["TreeNode"]) -> list[int]:
        """Return the path of token ids from the root's parent down to
        this node (inclusive). Used to construct the verify forward's
        input sequence."""
        path = [self.token_id]
        cur = self.parent
        while cur >= 0:
            path.append(tree[cur].token_id)
            cur = tree[cur].parent
        return list(reversed(path))


class TreeBuilder:
    """Build a DDTree from the MTP draft head's top-k predictions.

    Strategy: from each "best" leaf, expand top_k children and add them
    to the tree until budget is reached. Score nodes by cumulative
    log-prob; expand greedily from the best score.
    """
    def __init__(self, top_k: int = 4, max_depth: int = 6, budget: int = 22):
        self.top_k = top_k
        self.max_depth = max_depth
        self.budget = budget

    def build(self, draft_fn: Callable[[int, list[int]], list[tuple[int, float]]],
              root_token: int) -> list[TreeNode]:
        """Build the tree. `draft_fn(node_idx, path)` returns up to top_k
        (token_id, log_prob) candidates for the children of `path`."""
        tree: list[TreeNode] = [TreeNode(root_token, -1, 0, 0.0)]
        # Frontier: (score, node_idx) heap. Use a list with sorting since
        # builds are small (budget ~22).
        frontier = [(0.0, 0)]
        while frontier and len(tree) < self.budget:
            frontier.sort(reverse=True)
            score, node_idx = frontier.pop(0)
            depth = tree[node_idx].depth
            if depth >= self.max_depth: continue
            path = tree[node_idx].ancestor_path(tree)
            children = draft_fn(node_idx, path)[: self.top_k]
            for tok, lp in children:
                if len(tree) >= self.budget: break
                tree.append(TreeNode(tok, node_idx, depth + 1, score + lp))
                frontier.append((score + lp, len(tree) - 1))
        return tree


class TreeVerifyDriver:
    """Host-side tree-verify driver. Uses runtime.target_forward to
    score every node in one pass (per chain — the parallel tree-mode
    forward needs the kernel-side `prefill_megakernel_tree<Cfg>` which
    is the next-session work)."""

    def __init__(self, runtime, cfg: MTPConfig | None = None):
        self.runtime = runtime
        self.cfg = cfg or MTPConfig(use_tree_verify=True)
        self.builder = TreeBuilder(top_k=cfg.max_speculative_tokens if cfg else 4,
                                    max_depth=6, budget=self.cfg.tree_budget)

    def verify(self, prefix_ids: torch.Tensor, tree: list[TreeNode]
              ) -> tuple[list[int], int]:
        """Score every leaf->root path in the tree against the target.

        Returns (accepted_path_tokens, accepted_length). For the host-
        side reference, we just run one target forward per leaf. The
        fast path (one forward over the whole tree with tree-attention)
        needs the kernel-side primitive in `megakernel/tree_verify.cuh`.
        """
        leaves = [i for i in range(len(tree))
                  if all(n.parent != i for n in tree)]
        best: list[int] = []
        # Score each leaf's path: count how many of the path's argmaxes
        # match the target's predictions at each position.
        for leaf_idx in leaves:
            path = tree[leaf_idx].ancestor_path(tree)
            # Path includes the already-committed root_token; we only
            # speculate from path[1:] onward.
            spec = path[1:]
            if not spec: continue
            ids = torch.cat([prefix_ids,
                             torch.tensor(spec, device=prefix_ids.device,
                                          dtype=prefix_ids.dtype)])
            with torch.no_grad():
                out = self.runtime.model(input_ids=ids.unsqueeze(0), use_cache=False)
            logits = out.logits[0, len(prefix_ids)-1 : len(prefix_ids)-1 + len(spec)]
            argmaxes = logits.argmax(dim=-1).tolist()
            # Longest matching prefix.
            accepted: list[int] = []
            for j, tok in enumerate(spec):
                if argmaxes[j] == tok: accepted.append(tok)
                else: break
            if len(accepted) > len(best):
                best = accepted
        return best, len(best)


# ---------------------------------------------------------------------------
# Where the kernel-side speculative tree-verify will plug in
# ---------------------------------------------------------------------------
#
# Tree-verify (algo B) on the megakernel runtime needs ONE new kernel:
#
#   prefill_megakernel_tree<Cfg>(...)
#       Takes a `tree_descriptor` (parent_idx[], position[] per node)
#       and runs the same hybrid DN+FA stack over the tree, broadcasting
#       state from each parent to its children. Reuses dn_states /
#       fa_k_cache as a "tree-mode" cache — see the DDTree paper, sec 4.
#
# The host-side accept logic (longest verified path) is identical to
# what `models/qwen35_27b/src/dflash_decode.cpp` already implements;
# we port it to Python sitting on top of the kernel above.
#
# Estimated effort: ~3 days kernel + 1 day host + 2 days tuning.
# Acceptance length target: 6-8 tokens at budget=22 on HumanEval
# (matches the qwen35_27b results).
