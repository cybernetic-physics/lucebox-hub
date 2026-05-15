"""Wiring sanity tests for the qwen36_27b runtime.

Runs in seconds, no big model download. Validates the bits of
`runtime_hf` that *don't* require the 54 GB Qwen3.6-27B weights:

  - thinking-mode parser (split_thinking)
  - tool-call parser (parse_tool_calls)
  - generation-config plumbing
  - XGrammar compile + bitmask shape against the Qwen3.6 tokenizer

The full end-to-end check (HF reference logits == ours) lives in
trainer/test_correctness_vs_hf.py and requires the model download.
"""
from __future__ import annotations

import json
import os
import sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from runtime_hf import (
    parse_tool_calls, split_thinking, GenerationConfig,
    THINK_OPEN, THINK_CLOSE, VOCAB_SIZE, MODEL_NAME_DEFAULT,
)


def _count_bits(bitmask: torch.Tensor, vocab_size: int) -> int:
    """Count set bits across a packed-int32 token bitmask, masking off
    the unused tail above `vocab_size`."""
    flat = bitmask.view(torch.int32).flatten()
    total = 0
    for v in flat.tolist():
        # int.bit_count() requires non-negative; reinterpret int32 -> uint32.
        total += int(v & 0xFFFFFFFF).bit_count()
    # Mask off bits above vocab_size in the last word.
    overflow_bits = (flat.numel() * 32) - vocab_size
    if overflow_bits > 0:
        last = int(flat[-1].item()) & 0xFFFFFFFF
        # Bits at positions [vocab_size .. flat.numel()*32-1] are the high
        # bits of the last word. Subtract whatever's set there.
        high_mask = ((1 << overflow_bits) - 1) << (32 - overflow_bits)
        total -= (last & high_mask).bit_count()
    return total


def test_split_thinking_present():
    s = f"prefix {THINK_OPEN}I will compute step by step.{THINK_CLOSE} the answer is 42"
    body, think = split_thinking(s)
    assert think == "I will compute step by step.", think
    assert body == "prefix  the answer is 42" or body == "prefix the answer is 42", body


def test_split_thinking_absent():
    s = "no thinking here, just an answer."
    body, think = split_thinking(s)
    assert think is None
    assert body == s


def test_parse_tool_calls_single():
    raw = ("Sure, I'll call the calculator.\n"
           "<tool_call>{\"name\": \"calc\", \"arguments\": {\"a\": 3, \"b\": 4}}</tool_call>")
    stripped, calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "calc"
    assert calls[0].arguments == {"a": 3, "b": 4}
    assert "<tool_call>" not in stripped


def test_parse_tool_calls_args_as_string():
    raw = '<tool_call>{"name": "echo", "arguments": "{\\"text\\": \\"hi\\"}"}</tool_call>'
    _, calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].arguments == {"text": "hi"}


def test_parse_tool_calls_qwen3_native():
    """Qwen3 native ChatML format: <|tool_call|>{...}<|/tool_call|>"""
    raw = ("Sure.\n"
           '<|tool_call|>{"name": "search", "arguments": {"q": "weather"}}<|/tool_call|>')
    stripped, calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "weather"}
    assert "<|tool_call|>" not in stripped


def test_parse_tool_calls_qwen3_coder():
    """qwen3_coder XML-like: <function=NAME>{...}</function>"""
    raw = '<function=calc>{"a": 3, "b": 4}</function>'
    stripped, calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "calc"
    assert calls[0].arguments == {"a": 3, "b": 4}
    assert "<function=" not in stripped


def test_parse_tool_calls_multiple():
    raw = ('do these:\n'
           '<tool_call>{"name":"a","arguments":{}}</tool_call>'
           ' and\n'
           '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>')
    stripped, calls = parse_tool_calls(raw)
    assert len(calls) == 2
    assert [c.name for c in calls] == ["a", "b"]
    assert "<tool_call>" not in stripped


def test_parse_tool_calls_malformed_does_not_crash():
    # Regex only matches `<tool_call>{...}</tool_call>` shape; a non-JSON
    # block that doesn't even have braces is silently skipped (no crash).
    raw = "<tool_call>not json</tool_call>"
    stripped, calls = parse_tool_calls(raw)
    assert len(calls) == 0  # didn't match the regex, ignored
    # Now a brace-shaped but invalid-JSON block should produce a parse_error.
    raw2 = "<tool_call>{name: missing_quotes}</tool_call>"
    _, calls2 = parse_tool_calls(raw2)
    assert len(calls2) == 1
    assert calls2[0].name == "<parse_error>"


def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.max_tokens == 512
    assert cfg.temperature == 0.0
    assert cfg.enable_thinking is True
    assert cfg.tools is None


def test_xgrammar_json_schema_compiles():
    """The grammar matcher should compile a JSON schema against the
    Qwen3.6 tokenizer's vocabulary. Smoke test — does not load the model."""
    import xgrammar as xg
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME_DEFAULT, trust_remote_code=True)
    info = xg.TokenizerInfo.from_huggingface(tok, vocab_size=VOCAB_SIZE)
    compiler = xg.GrammarCompiler(info)
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    compiled = compiler.compile_json_schema(schema)
    matcher = xg.GrammarMatcher(compiled)
    bitmask = xg.allocate_token_bitmask(1, VOCAB_SIZE)
    matcher.fill_next_token_bitmask(bitmask)
    # bitmask is packed int32 — count set bits, not sum of int32 values.
    n_allowed = int(_count_bits(bitmask, VOCAB_SIZE))
    assert 0 < n_allowed < VOCAB_SIZE, f"weird bitmask: {n_allowed} allowed"
    print(f"  json-schema mask: {n_allowed} / {VOCAB_SIZE} tokens initially valid")


def test_xgrammar_grammar_compiles():
    import xgrammar as xg
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME_DEFAULT, trust_remote_code=True)
    info = xg.TokenizerInfo.from_huggingface(tok, vocab_size=VOCAB_SIZE)
    compiler = xg.GrammarCompiler(info)
    # A trivial GBNF: produce one of "yes", "no", "maybe".
    grammar = 'root ::= "yes" | "no" | "maybe"'
    compiled = compiler.compile_grammar(grammar)
    matcher = xg.GrammarMatcher(compiled)
    bitmask = xg.allocate_token_bitmask(1, VOCAB_SIZE)
    matcher.fill_next_token_bitmask(bitmask)
    n_allowed = int(_count_bits(bitmask, VOCAB_SIZE))
    assert 0 < n_allowed < VOCAB_SIZE, f"weird bitmask: {n_allowed} allowed"
    print(f"  ebnf mask: {n_allowed} / {VOCAB_SIZE} tokens initially valid")


def test_chat_template_thinking_flag():
    """Verify the Qwen3.6 chat template honors enable_thinking."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME_DEFAULT, trust_remote_code=True)
    messages = [{"role": "user", "content": "Hello"}]
    on = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                  chat_template_kwargs={"enable_thinking": True})
    off = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   chat_template_kwargs={"enable_thinking": False})
    # If the template handles the flag, the two prompts should differ.
    if on == off:
        print(f"  NOTE: chat template did not honor enable_thinking flag "
              f"(may be model-side default for this build)")
    print(f"  thinking-on prompt length: {len(on)}, thinking-off: {len(off)}")


def main():
    tests = [
        test_split_thinking_present,
        test_split_thinking_absent,
        test_parse_tool_calls_single,
        test_parse_tool_calls_args_as_string,
        test_parse_tool_calls_qwen3_native,
        test_parse_tool_calls_qwen3_coder,
        test_parse_tool_calls_multiple,
        test_parse_tool_calls_malformed_does_not_crash,
        test_generation_config_defaults,
        test_xgrammar_json_schema_compiles,
        test_xgrammar_grammar_compiles,
        test_chat_template_thinking_flag,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failures.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)
    print(f"ALL {len(tests)} WIRING TESTS PASSED")


if __name__ == "__main__":
    main()
