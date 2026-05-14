"""OpenAI-compatible HTTP server for Qwen3.6-27B.

Supports:
  - /v1/chat/completions  (with thinking-mode toggle, tools=, response_format={"type":"json_schema",...})
  - /v1/completions
  - /v1/models
  - /healthz

Wraps `runtime_hf.Qwen36Runtime`. The runtime currently uses HF
transformers — correctness baseline. The megakernel-accelerated runtime
will be a drop-in replacement (Phase 1+ of PLAN.md).

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 -m \\
        models.qwen36_27b.serve.openai_server \\
        --backend bf16 --port 8765
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys, os
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
from runtime_hf import (
    Qwen36Runtime, GenerationConfig, MODEL_NAME_DEFAULT,
    parse_tool_calls, split_thinking,
)

# ----------------------------- pydantic types -----------------------------

class ChatMessage(BaseModel):
    role: str
    content: str | list

class ResponseFormat(BaseModel):
    type: str = "text"
    json_schema: dict | None = None
    grammar: str | None = None

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] | None = None
    tools: list[dict] | None = None
    enable_thinking: bool = True
    preserve_thinking: bool = False
    response_format: ResponseFormat | None = None
    stream: bool = False  # streaming not implemented in this baseline

class CompletionRequest(BaseModel):
    prompt: str
    model: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] | None = None
    response_format: ResponseFormat | None = None
    enable_thinking: bool = True
    preserve_thinking: bool = False


# ----------------------------- runtime holder -----------------------------

class _State:
    runtime: Qwen36Runtime | None = None
    backend: str = "bf16"
    model_name: str = MODEL_NAME_DEFAULT

state = _State()


def _gen_config_from(req: ChatRequest | CompletionRequest) -> GenerationConfig:
    cfg = GenerationConfig(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop or [],
        enable_thinking=getattr(req, "enable_thinking", True),
        preserve_thinking_in_text=getattr(req, "preserve_thinking", False),
        tools=getattr(req, "tools", None),
    )
    rf = getattr(req, "response_format", None)
    if rf is not None:
        if rf.type == "json_schema":
            cfg.response_format_json_schema = rf.json_schema
        elif rf.type == "grammar":
            cfg.response_format_grammar = rf.grammar
    return cfg


# ----------------------------- app -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.runtime is None:
        print("[openai_server] loading runtime in lifespan...", flush=True)
        state.runtime = Qwen36Runtime(model_name=state.model_name, backend=state.backend)
    yield
    state.runtime = None

app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {
        "ok": state.runtime is not None,
        "model": state.model_name,
        "backend": state.backend,
        "gpu_alloc_gb": (torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
    }


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": state.model_name, "object": "model", "owned_by": "qwen"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if state.runtime is None:
        raise HTTPException(status_code=503, detail="runtime not loaded")
    if req.stream:
        raise HTTPException(status_code=400, detail="streaming not implemented in baseline runtime")
    cfg = _gen_config_from(req)
    messages = [m.model_dump() for m in req.messages]
    # Serialize the runtime call to avoid concurrent HF generation on one GPU.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: state.runtime.chat(messages, cfg))

    # Build OpenAI-style response.
    msg: dict[str, Any] = {"role": "assistant", "content": result.text or None}
    if result.thinking is not None:
        # Qwen3 thinking exposed under a non-standard field; OpenAI tools that
        # don't know about it still see content.
        msg["reasoning_content"] = result.thinking
    if result.tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in result.tool_calls
        ]
        msg["content"] = result.text or None
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": state.model_name,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": "tool_calls" if result.tool_calls else result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        "_lucebox": {"elapsed_s": result.elapsed_s, "backend": state.backend},
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if state.runtime is None:
        raise HTTPException(status_code=503, detail="runtime not loaded")
    cfg = _gen_config_from(req)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: state.runtime.complete(req.prompt, cfg))
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": state.model_name,
        "choices": [{"index": 0, "text": result.text, "finish_reason": result.finish_reason}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        "_lucebox": {"elapsed_s": result.elapsed_s},
    }


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="bf16", choices=["bf16", "fp8", "bnb-4bit"])
    ap.add_argument("--model", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    state.backend = args.backend
    state.model_name = args.model
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
