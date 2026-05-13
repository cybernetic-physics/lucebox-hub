"""Hermes-style tool-calling agent over the megakernel server.

Uses the standard NousResearch Hermes function-calling prompt: the system
turn declares available tools in JSON-schema form inside <tools> tags;
the model is expected to emit one or more <tool_call> JSON objects when
it wants to call a tool, and a plain assistant turn otherwise.

The agent loop:
  user -> assistant
    if <tool_call> tags present:
        parse each, dispatch to the registered tool
        return tool results as role="tool" messages
        re-query the assistant
    else:
        emit assistant text, return to user

Run from repo root:
  source /home/sparkz/rl/.venv/bin/activate
  PYTHONPATH=models/qwen35_0p8b python -m serve.hermes_agent \\
      --base http://127.0.0.1:8765 --task "What is 17 * 23?"

  # or interactive REPL:
  python -m serve.hermes_agent --base http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
from typing import Any, Callable

import requests


# ----- Tool registry ----- #

ToolFn = Callable[[dict], dict]


def tool_calculator(args: dict) -> dict:
    expr = args.get("expression", "")
    # Sandbox: only allow digit/operator characters and a tiny math name set.
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)%^,a-zA-Z_]+", expr):
        return {"error": "expression contains disallowed characters"}
    safe_env = {n: getattr(math, n) for n in (
        "pi", "e", "sqrt", "sin", "cos", "tan", "log", "log10", "exp",
        "floor", "ceil", "fabs", "pow"
    )}
    try:
        val = eval(expr, {"__builtins__": {}}, safe_env)  # noqa: S307
    except Exception as exc:
        return {"error": f"eval failed: {exc!r}"}
    return {"result": val}


def tool_get_time(args: dict) -> dict:
    tz = args.get("timezone", "UTC")
    now = _dt.datetime.now(_dt.timezone.utc)
    return {"iso8601": now.isoformat(), "requested_timezone": tz}


def tool_echo(args: dict) -> dict:
    return {"echoed": args.get("text", "")}


TOOLS: dict[str, dict] = {
    "calculator": {
        "fn": tool_calculator,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": ("Evaluate a Python arithmetic expression. "
                                "Supports +, -, *, /, **, parentheses, and "
                                "sqrt/sin/cos/log/exp/pi/e from math."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string",
                                       "description": "e.g. '17 * 23'"},
                    },
                    "required": ["expression"],
                },
            },
        },
    },
    "get_time": {
        "fn": tool_get_time,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current wall-clock time in ISO-8601.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string",
                                     "description": "IANA tz name; default UTC"},
                    },
                },
            },
        },
    },
    "echo": {
        "fn": tool_echo,
        "schema": {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a string back. For wiring sanity.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    },
}


# ----- Hermes prompt scaffolding ----- #

HERMES_SYSTEM_TEMPLATE = """You are a function-calling AI assistant.
Within <tools></tools> XML tags you have access to the following tools:
<tools>
{tools_json}
</tools>

For each function call, return a JSON object with the function name and \
arguments within <tool_call></tool_call> tags:
<tool_call>
{{"name": "<function-name>", "arguments": <args-as-json-object>}}
</tool_call>

If multiple tool calls are needed, emit each inside its own <tool_call> \
block. When you have the final answer, reply in plain text without any \
<tool_call> tags."""


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def build_system(tools: dict[str, dict]) -> str:
    schemas = [t["schema"] for t in tools.values()]
    return HERMES_SYSTEM_TEMPLATE.format(
        tools_json=json.dumps(schemas, indent=2)
    )


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError as exc:
            calls.append({"_parse_error": str(exc), "_raw": m.group(1)})
    return calls


# ----- Agent loop ----- #

class HermesAgent:
    def __init__(self, base_url: str, model: str = "qwen35_0p8b-megakernel",
                 max_steps: int = 4, max_tokens: int = 256,
                 tools: dict[str, dict] = TOOLS, verbose: bool = True):
        self.base = base_url.rstrip("/")
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.tools = tools
        self.verbose = verbose
        self.system = build_system(tools)

    def _chat(self, messages: list[dict]) -> str:
        r = requests.post(
            f"{self.base}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def run(self, user_msg: str) -> dict:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_msg},
        ]
        trace = []
        for step in range(self.max_steps):
            if self.verbose:
                print(f"\n[step {step+1}] -> assistant", flush=True)
            assistant_text = self._chat(messages)
            trace.append({"role": "assistant", "content": assistant_text})
            if self.verbose:
                print(f"  raw: {assistant_text[:300]}", flush=True)

            calls = parse_tool_calls(assistant_text)
            if not calls:
                return {
                    "final": assistant_text.strip(),
                    "trace": trace,
                    "steps": step + 1,
                    "stop_reason": "no_tool_call",
                }

            messages.append({"role": "assistant", "content": assistant_text})

            tool_block_parts = []
            for call in calls:
                name = call.get("name")
                args = call.get("arguments", {}) or {}
                if name not in self.tools:
                    result = {"error": f"unknown tool {name!r}"}
                else:
                    if self.verbose:
                        print(f"  tool: {name}({args})", flush=True)
                    try:
                        result = self.tools[name]["fn"](args)
                    except Exception as exc:
                        result = {"error": f"tool raised: {exc!r}"}
                if self.verbose:
                    print(f"  result: {result}", flush=True)
                tool_block_parts.append(
                    f"<tool_response>\n"
                    f"{json.dumps({'name': name, 'content': result})}\n"
                    f"</tool_response>"
                )
                trace.append({"role": "tool", "name": name,
                              "arguments": args, "result": result})

            # Hermes convention: tool results go back as a single user
            # message containing all tool_response blocks.
            messages.append({"role": "user",
                             "content": "\n".join(tool_block_parts)})

        return {
            "final": "(max steps reached without final answer)",
            "trace": trace,
            "steps": self.max_steps,
            "stop_reason": "max_steps",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765",
                    help="URL of the openai_server")
    ap.add_argument("--model", default="qwen35_0p8b-megakernel")
    ap.add_argument("--task", default=None,
                    help="One-shot task. If omitted, runs an interactive REPL.")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="Emit the full trace as JSON at the end.")
    args = ap.parse_args()

    agent = HermesAgent(
        base_url=args.base, model=args.model,
        max_steps=args.max_steps, max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )

    # Quick server reachability check.
    try:
        h = requests.get(f"{agent.base}/healthz", timeout=10).json()
        print(f"[server] {h}", flush=True)
    except Exception as exc:
        print(f"ERROR: server at {agent.base} unreachable: {exc}",
              file=sys.stderr)
        sys.exit(2)

    if args.task is not None:
        out = agent.run(args.task)
        print("\n=== FINAL ===")
        print(out["final"])
        if args.json:
            print("\n=== TRACE (JSON) ===")
            print(json.dumps(out, indent=2))
        return

    print("Interactive agent. Ctrl-D to exit. Tools: "
          + ", ".join(TOOLS.keys()), flush=True)
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        out = agent.run(user)
        print(f"\nagent> {out['final']}")
        if args.json:
            print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
