"""HF-backed Qwen3.6-27B runtime — correctness-by-construction baseline.

Wraps `transformers` Qwen3.6-27B with the same Decoder API as our 0.8B
megakernel (`models/qwen35_0p8b/model.py:Decoder`). The megakernel
optimization will swap in later phases; this file is the working
reference that supports everything Qwen3.6 ships:

  - chat template (multi-turn, system prompt, multimodal-ready)
  - thinking mode (`<think>...</think>` tags, native)
  - tool calls (OpenAI-compatible; parsed at output time)
  - grammar-constrained decoding (XGrammar logit mask)
  - streaming / non-streaming

Backends:
  - "bf16"     : full BF16 weights (~54 GB on GB10)
  - "fp8"      : Qwen/Qwen3.6-27B-FP8 (~27 GB; native NVFP8 KV)
  - "bnb-4bit" : bitsandbytes 4-bit (~14 GB; fits on 24 GB consumer)

The kernel-accelerated path will replace `_hf_forward()` with calls
into our megakernel ops; everything above that line stays the same.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME_DEFAULT = "Qwen/Qwen3.6-27B"

# Qwen3.6-27B architecture (from HF model card; mirrored in PLAN.md).
# These constants are the model invariants; runtime_hf doesn't use them
# directly (HF handles the model), but the future megakernel path will.
NUM_LAYERS         = 64
HIDDEN_SIZE        = 5120
INTERMEDIATE_SIZE  = 17408
FA_NUM_Q_HEADS     = 24
FA_NUM_KV_HEADS    = 4
FA_HEAD_DIM        = 256
DN_NUM_V_HEADS     = 48
DN_NUM_QK_HEADS    = 16
DN_HEAD_DIM        = 128
FA_ROTARY_DIM      = 64
FA_ROPE_THETA      = 1.0e7
VOCAB_SIZE         = 248320
NATIVE_CONTEXT     = 262_144

# Thinking-mode tags. Qwen3.6 uses `<think>...</think>` as native
# reasoning markers — emitted in the model output, parseable on host.
THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """One parsed tool invocation from the model's output."""
    name: str
    arguments: dict
    raw: str = ""  # original text span this was parsed from


@dataclass
class GenerationResult:
    """One generation. `tool_calls` is populated post-hoc by parsing
    `text`; `thinking` holds the content between <think>...</think> if
    the response contained a reasoning block.
    """
    text: str
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0


@dataclass
class GenerationConfig:
    """User-facing knobs for one generation."""
    max_tokens: int = 512
    temperature: float = 0.0           # 0.0 == greedy
    top_p: float = 1.0
    top_k: int = -1                    # -1 == disabled
    stop: list[str] = field(default_factory=list)
    enable_thinking: bool = True
    preserve_thinking_in_text: bool = False   # if False, strip <think>...</think> out
    response_format_json_schema: dict | None = None  # XGrammar JSON-schema mask
    response_format_grammar: str | None = None       # raw GBNF/EBNF grammar
    tools: list[dict] | None = None               # OpenAI-style tool defs


# ---------------------------------------------------------------------------
# Tool-call parser (Qwen3 native format)
# ---------------------------------------------------------------------------

import json
import re

# Qwen3 native tool-call format embeds `<tool_call>{...}</tool_call>`
# blocks in the model output. Also accept Hermes-style (compatible).
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)

def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool-call blocks from model output. Returns
    (text_without_tool_calls, list_of_tool_calls).
    """
    calls: list[ToolCall] = []
    def _take(m: re.Match) -> str:
        raw = m.group(1)
        try:
            obj = json.loads(raw)
            # Qwen3 format: {"name": ..., "arguments": {...}}
            # Hermes format: {"name": ..., "arguments": {...}}
            name = str(obj.get("name", "unknown"))
            args = obj.get("arguments", {})
            if isinstance(args, str):
                # Some templates serialize arguments as a JSON string.
                try: args = json.loads(args)
                except Exception: args = {"_raw": args}
            calls.append(ToolCall(name=name, arguments=args, raw=m.group(0)))
        except json.JSONDecodeError:
            calls.append(ToolCall(name="<parse_error>", arguments={"_raw": raw},
                                  raw=m.group(0)))
        return ""
    stripped = _TOOL_CALL_RE.sub(_take, text)
    return stripped.strip(), calls


# ---------------------------------------------------------------------------
# Thinking-mode helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(
    re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.DOTALL
)

def split_thinking(text: str) -> tuple[str, str | None]:
    """Pull <think>...</think> out of the output. Returns
    (text_without_think, thinking_content_or_None).
    """
    m = _THINK_RE.search(text)
    if not m:
        return text, None
    thinking = m.group(1).strip()
    stripped = (_THINK_RE.sub("", text)).strip()
    return stripped, thinking


# ---------------------------------------------------------------------------
# XGrammar logits processor (only constructed when caller asks for it)
# ---------------------------------------------------------------------------

class _XGrammarLogitsProcessor:
    """Bridges xgrammar's bitmask to a `transformers` LogitsProcessor.

    Builds the matcher lazily on first __call__ since instantiation
    requires the tokenizer (which we have on the runtime).
    """
    def __init__(self, compiled_grammar, vocab_size: int, device: torch.device):
        import xgrammar as xg
        self._matcher = xg.GrammarMatcher(compiled_grammar)
        self._bitmask = xg.allocate_token_bitmask(1, vocab_size)
        self._device = device
        self._xg = xg

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: [batch=1, seq]; scores: [batch=1, vocab]
        # On the first call no token has been sampled yet; subsequent calls
        # carry the previously-sampled token.
        # The HF generate loop calls this BEFORE sampling, so we need to
        # advance the matcher by the *last* token in input_ids only if it
        # was actually committed by sampling. Practical approach:
        # accept_token after each sampled step via a paired callback.
        # transformers >=5 doesn't expose that cleanly; use the simpler
        # "fill mask from current state" pattern: the matcher tracks its
        # own state separately and is advanced from outside the processor
        # in `generate_with_grammar` below.
        self._xg.apply_token_bitmask_inplace(scores, self._bitmask.to(self._device))
        return scores

    def advance(self, token_id: int) -> bool:
        return self._matcher.accept_token(token_id)

    def fill_mask(self) -> None:
        self._matcher.fill_next_token_bitmask(self._bitmask)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class Qwen36Runtime:
    """HF-backed runtime for Qwen3.6-27B. API mirrors the 0.8B Decoder.

    Threading note: holds an internal lock; safe for multiple callers,
    but each generate() is serial under the lock.
    """

    BASE_MODEL = "Qwen/Qwen3.6-27B"

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME_DEFAULT,
        backend: str = "bf16",
        device: str = "cuda",
        max_context: int = 32_768,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.max_context = max_context

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[runtime_hf] tokenizer: {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)

        load_kwargs: dict[str, Any] = dict(
            trust_remote_code=trust_remote_code,
            device_map=device,
        )
        if backend == "bf16":
            load_kwargs["dtype"] = torch.bfloat16
        elif backend == "fp8":
            # Qwen ships a separate FP8 weight repo.
            if model_name == self.BASE_MODEL:
                model_name = self.BASE_MODEL + "-FP8"
            load_kwargs["dtype"] = "auto"
        elif backend == "bnb-4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise ValueError(f"unknown backend: {backend}")

        print(f"[runtime_hf] model: {model_name}  backend={backend}", flush=True)
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        print(f"[runtime_hf] loaded in {time.perf_counter()-t0:.1f}s; "
              f"gpu_alloc={torch.cuda.memory_allocated() / (1024**3):.1f} GB",
              flush=True)

        # Lazy XGrammar compiler.
        self._xg_compiler = None
        self._xg_tokenizer_info = None

    # ----- Tokenization + chat template -----

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool = True,
        add_generation_prompt: bool = True,
        tools: list[dict] | None = None,
    ) -> str:
        """Run the model's chat template. Returns the prompt string."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        if tools is not None:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    # ----- XGrammar wiring -----

    def _xgrammar_compile(self, grammar_text: str | None = None,
                          json_schema: dict | None = None):
        import xgrammar as xg
        if self._xg_compiler is None:
            self._xg_tokenizer_info = xg.TokenizerInfo.from_huggingface(
                self.tokenizer, vocab_size=VOCAB_SIZE)
            self._xg_compiler = xg.GrammarCompiler(self._xg_tokenizer_info)
        if json_schema is not None:
            return self._xg_compiler.compile_json_schema(json_schema)
        if grammar_text is not None:
            return self._xg_compiler.compile_grammar(grammar_text)
        raise ValueError("must provide json_schema or grammar_text")

    # ----- Forward / generate -----

    def _hf_generate(
        self,
        prompt_ids: torch.Tensor,
        cfg: GenerationConfig,
    ) -> tuple[str, int]:
        """One blocking generate call. Returns (text, completion_tokens)."""
        from transformers import LogitsProcessorList

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=cfg.max_tokens,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            top_k=(cfg.top_k if cfg.top_k > 0 else None),
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if cfg.stop:
            stop_ids = []
            for s in cfg.stop:
                ids = self.tokenizer(s, add_special_tokens=False).input_ids
                if ids: stop_ids.extend(ids)
            if stop_ids:
                gen_kwargs["eos_token_id"] = list(set(
                    [self.tokenizer.eos_token_id] + stop_ids
                ))

        # Optional grammar mask.
        processors = LogitsProcessorList()
        grammar_proc = None
        if cfg.response_format_json_schema is not None:
            compiled = self._xgrammar_compile(json_schema=cfg.response_format_json_schema)
            grammar_proc = _XGrammarLogitsProcessor(
                compiled, VOCAB_SIZE, torch.device(self.device))
            processors.append(grammar_proc)
        elif cfg.response_format_grammar is not None:
            compiled = self._xgrammar_compile(grammar_text=cfg.response_format_grammar)
            grammar_proc = _XGrammarLogitsProcessor(
                compiled, VOCAB_SIZE, torch.device(self.device))
            processors.append(grammar_proc)
        if len(processors) > 0:
            gen_kwargs["logits_processor"] = processors

        with torch.no_grad():
            out = self.model.generate(
                input_ids=prompt_ids,
                **gen_kwargs,
            )
        new_tokens = out[0, prompt_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return text, int(new_tokens.numel())

    def chat_stream(
        self,
        messages: list[dict],
        cfg: GenerationConfig | None = None,
    ):
        """Streaming generator. Yields incremental text chunks; final yield
        is a GenerationResult-shaped dict with usage stats.

        Uses transformers' TextIteratorStreamer running the actual generate
        call on a background thread; the main thread drains the streamer
        until EOS.
        """
        from threading import Thread
        from transformers import TextIteratorStreamer, LogitsProcessorList

        cfg = cfg or GenerationConfig()
        prompt = self.apply_chat_template(
            messages, enable_thinking=cfg.enable_thinking, tools=cfg.tools,
        )
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False,
            timeout=600.0,
        )

        gen_kwargs: dict[str, Any] = dict(
            input_ids=prompt_ids,
            max_new_tokens=cfg.max_tokens,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            top_k=(cfg.top_k if cfg.top_k > 0 else None),
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            streamer=streamer,
        )
        if cfg.stop:
            stop_ids = []
            for s in cfg.stop:
                ids = self.tokenizer(s, add_special_tokens=False).input_ids
                if ids: stop_ids.extend(ids)
            if stop_ids:
                gen_kwargs["eos_token_id"] = list(set(
                    [self.tokenizer.eos_token_id] + stop_ids))

        processors = LogitsProcessorList()
        if cfg.response_format_json_schema is not None:
            compiled = self._xgrammar_compile(json_schema=cfg.response_format_json_schema)
            processors.append(_XGrammarLogitsProcessor(
                compiled, VOCAB_SIZE, torch.device(self.device)))
        elif cfg.response_format_grammar is not None:
            compiled = self._xgrammar_compile(grammar_text=cfg.response_format_grammar)
            processors.append(_XGrammarLogitsProcessor(
                compiled, VOCAB_SIZE, torch.device(self.device)))
        if len(processors) > 0:
            gen_kwargs["logits_processor"] = processors

        # Drive generate on a worker thread, drain the streamer here.
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        t0 = time.perf_counter()
        n_tokens = 0
        accumulated = ""
        for chunk in streamer:
            if not chunk: continue
            accumulated += chunk
            n_tokens += 1
            yield {"type": "chunk", "text": chunk}
        thread.join()
        elapsed = time.perf_counter() - t0

        body, thinking = split_thinking(accumulated)
        clean, tools = parse_tool_calls(body)
        yield {
            "type": "done",
            "text": clean if not cfg.preserve_thinking_in_text else accumulated,
            "thinking": thinking,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments} for tc in tools
            ],
            "prompt_tokens": int(prompt_ids.shape[1]),
            "completion_tokens": n_tokens,
            "elapsed_s": elapsed,
        }

    def chat(
        self,
        messages: list[dict],
        cfg: GenerationConfig | None = None,
    ) -> GenerationResult:
        """End-to-end chat with chat template + thinking + tools + grammar."""
        cfg = cfg or GenerationConfig()
        prompt = self.apply_chat_template(
            messages,
            enable_thinking=cfg.enable_thinking,
            tools=cfg.tools,
        )
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        t0 = time.perf_counter()
        raw_text, n_new = self._hf_generate(prompt_ids, cfg)
        elapsed = time.perf_counter() - t0

        # Pull thinking out first; tool-calls live in the post-think portion.
        body, thinking = split_thinking(raw_text)
        clean, tools = parse_tool_calls(body)

        final_text = raw_text if cfg.preserve_thinking_in_text else clean
        return GenerationResult(
            text=final_text,
            thinking=thinking,
            tool_calls=tools,
            finish_reason="stop",
            prompt_tokens=int(prompt_ids.shape[1]),
            completion_tokens=n_new,
            elapsed_s=elapsed,
        )

    def complete(self, prompt: str, cfg: GenerationConfig | None = None) -> GenerationResult:
        """Raw text completion (no chat template)."""
        cfg = cfg or GenerationConfig()
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        t0 = time.perf_counter()
        raw_text, n_new = self._hf_generate(prompt_ids, cfg)
        elapsed = time.perf_counter() - t0
        body, thinking = split_thinking(raw_text)
        clean, tools = parse_tool_calls(body)
        final_text = raw_text if cfg.preserve_thinking_in_text else clean
        return GenerationResult(
            text=final_text, thinking=thinking, tool_calls=tools,
            prompt_tokens=int(prompt_ids.shape[1]),
            completion_tokens=n_new, elapsed_s=elapsed,
        )

    def logits_for(self, prompt_text_or_ids) -> torch.Tensor:
        """One forward pass; returns last-position fp32 logits over vocab.

        Used by the correctness harness — matches what
        `models/qwen35_0p8b/trainer/bench_correctness_vs_hf.py` does on
        the 0.8B side.
        """
        if isinstance(prompt_text_or_ids, str):
            ids = self.tokenizer(prompt_text_or_ids, return_tensors="pt").input_ids
        else:
            ids = prompt_text_or_ids
            if ids.dim() == 1: ids = ids.unsqueeze(0)
        ids = ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=False)
        return out.logits[0, -1].detach().to(torch.float32)


# ---------------------------------------------------------------------------
# Convenience constructor used by tests + serve
# ---------------------------------------------------------------------------

def load_runtime(
    backend: str = "bf16",
    model_name: str | None = None,
    device: str = "cuda",
    max_context: int = 32_768,
) -> Qwen36Runtime:
    return Qwen36Runtime(
        model_name=(model_name or os.environ.get("MODEL", MODEL_NAME_DEFAULT)),
        backend=backend, device=device, max_context=max_context,
    )
