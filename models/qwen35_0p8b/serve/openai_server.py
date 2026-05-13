"""OpenAI-compatible HTTP server backed by the qwen35_0p8b megakernel.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions   (greedy only; temperature/top_p ignored)
  POST /v1/completions        (raw prompt → completion)
  GET  /healthz

The megakernel decoder is greedy-only (argmax inside the kernel), so the
server reports an explicit warning if a non-default temperature/top_p is
requested.

Run from repo root:
  source /home/sparkz/rl/.venv/bin/activate
  export HF_HOME=/home/sparkz/rl/.hf_cache
  PYTHONPATH=models/qwen35_0p8b python -m serve.openai_server \\
      --backend bf16 --port 8765
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importing the C extension and Decoder requires PYTHONPATH=models/qwen35_0p8b.
import qwen35_megakernel_bf16_C  # noqa: F401
from model import Decoder, MAX_SEQ_LEN
from transformers import AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen35_0p8b-megakernel"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=MAX_SEQ_LEN)
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: dict


class CompletionRequest(BaseModel):
    model: str = "qwen35_0p8b-megakernel"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=MAX_SEQ_LEN)
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: dict


class Engine:
    """Thread-safe wrapper around a single Decoder instance."""

    def __init__(self, model_name: str, backend: str, verbose: bool = False):
        self.model_name = model_name
        self.backend = backend
        print(f"[engine] loading Decoder({model_name}, backend={backend})...",
              flush=True)
        self.decoder = Decoder(model_name=model_name, backend=backend,
                               verbose=verbose)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lock = threading.Lock()
        # Resolve im_end token id for early-stop.
        ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_id = ids[0] if len(ids) == 1 else None
        print(f"[engine] backend={self.decoder.backend_label} "
              f"eos={self.tokenizer.eos_token_id} im_end={self.im_end_id}",
              flush=True)

    def render_chat(self, messages: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        stop_token_ids: list[int],
        stop_strings: Optional[list[str]] = None,
    ):
        with self.lock:
            self.decoder.reset()
            if self.backend == "bf16":
                first = self.decoder.prefill(prompt_ids)
            else:
                # NVFP4 has no fused prefill; step through.
                for tid in prompt_ids[:-1]:
                    self.decoder.step(int(tid))
                first = self.decoder.step(int(prompt_ids[-1]))

            out_ids = []
            finish = "length"
            cur = int(first)
            for _ in range(max_tokens):
                if cur in stop_token_ids:
                    finish = "stop"
                    break
                out_ids.append(cur)
                # Optional string-level stop.
                if stop_strings:
                    cur_text = self.tokenizer.decode(
                        out_ids, skip_special_tokens=False
                    )
                    if any(s in cur_text for s in stop_strings):
                        finish = "stop"
                        break
                cur = int(self.decoder.step(cur))

            text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return text, out_ids, finish


def _maybe_warn_sampling(req):
    if req.temperature not in (0.0, 1.0) or req.top_p != 1.0:
        print(f"[warn] megakernel is greedy-only; ignoring "
              f"temperature={req.temperature} top_p={req.top_p}", flush=True)


def build_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="qwen35_0p8b megakernel OpenAI server")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "backend": engine.decoder.backend_label,
                "model": engine.model_name}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "qwen35_0p8b-megakernel",
                "object": "model",
                "owned_by": "lucebox-hub",
                "backend": engine.decoder.backend_label,
                "base_model": engine.model_name,
            }],
        }

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    def chat(req: ChatRequest):
        _maybe_warn_sampling(req)
        messages = [m.model_dump() for m in req.messages]
        prompt_ids = engine.render_chat(messages)
        if len(prompt_ids) + req.max_tokens >= MAX_SEQ_LEN:
            raise HTTPException(
                400,
                f"prompt({len(prompt_ids)}) + max_tokens({req.max_tokens}) "
                f">= MAX_SEQ_LEN({MAX_SEQ_LEN})",
            )
        stop_ids = []
        if engine.im_end_id is not None:
            stop_ids.append(engine.im_end_id)
        if engine.tokenizer.eos_token_id is not None:
            stop_ids.append(engine.tokenizer.eos_token_id)

        t0 = time.perf_counter()
        text, out_ids, finish = engine.generate(
            prompt_ids, req.max_tokens, stop_ids, req.stop
        )
        dt = time.perf_counter() - t0
        tps = len(out_ids) / dt if dt > 0 else 0.0
        print(f"[chat] prompt={len(prompt_ids)} gen={len(out_ids)} "
              f"finish={finish} {tps:.1f} tok/s", flush=True)

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=req.model,
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish,
            )],
            usage={
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
            },
        )

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(req: CompletionRequest):
        _maybe_warn_sampling(req)
        prompt_ids = engine.tokenizer.encode(
            req.prompt, add_special_tokens=False
        )
        stop_ids = []
        if engine.tokenizer.eos_token_id is not None:
            stop_ids.append(engine.tokenizer.eos_token_id)
        t0 = time.perf_counter()
        text, out_ids, finish = engine.generate(
            prompt_ids, req.max_tokens, stop_ids, req.stop
        )
        dt = time.perf_counter() - t0
        print(f"[cmpl] prompt={len(prompt_ids)} gen={len(out_ids)} "
              f"finish={finish} {len(out_ids)/max(dt,1e-9):.1f} tok/s",
              flush=True)
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=req.model,
            choices=[CompletionChoice(text=text, finish_reason=finish)],
            usage={
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
            },
        )

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--backend", default="bf16",
                    choices=("auto", "bf16", "nvfp4"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    engine = Engine(args.model_name, args.backend)
    app = build_app(engine)

    # Self-test on startup so a broken weight path fails fast.
    test_prompt = engine.render_chat([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "ping"},
    ])
    text, ids, finish = engine.generate(test_prompt, 8, [
        engine.im_end_id, engine.tokenizer.eos_token_id
    ] if engine.im_end_id else [engine.tokenizer.eos_token_id], None)
    print(f"[startup-selftest] generated {len(ids)} tokens "
          f"finish={finish}: {text[:80]!r}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
