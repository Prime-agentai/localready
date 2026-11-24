"""A configurable fake local-model server that reproduces real engine behaviours and failure modes.

Used by the test suite, and live: drive a real coding agent against it to capture the agent's
request shapes without paying for a model.

Behaviours (``Behaviour`` fields):
  n_ctx               context window in tokens (1 token ~= 4 chars here, deterministic)
  overflow            "error"     -> HTTP 400 like llama.cpp (loud, good)
                      "truncate"  -> keep the LAST n_ctx tokens and answer 200 (silent, like a
                                     default Ollama num_ctx)
  anthropic           serve /v1/messages
  openai              serve /v1/chat/completions
  responses           serve /v1/responses
  tool_calls          model emits a tool call when tools are offered and the last turn is a user
  stream_tool_args    "ok" | "missing_start" (no content_block_start for tool_use) |
                      "bad_json" (input_json_delta does not assemble to JSON)
  drop_system_blocks  ignore list-form system prompts (a real bug class in proxies)
  reject_cache_control 400 when cache_control appears anywhere
  prefill_tps / decode_tps   simulated speed (sleep), 0 = instant
  cache_prefix        simulate a prefix KV cache: only uncached tokens cost prefill time
  engine              "llama.cpp" | "ollama" | "vllm" | "lmstudio" | "unknown" (discovery endpoints)
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler

from localready.proxy import FastHTTPServer as ThreadingHTTPServer
from typing import Any

CHARS_PER_TOKEN = 4


def ntok(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass
class Behaviour:
    n_ctx: int = 32768
    overflow: str = "error"
    anthropic: bool = True
    openai: bool = True
    responses: bool = True
    count_tokens: bool = True
    tool_calls: bool = True
    stream_tool_args: str = "ok"
    drop_system_blocks: bool = False
    reject_cache_control: bool = False
    reject_over_budget: bool = False  # vLLM-style: prompt + max_tokens must fit in n_ctx
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    cache_prefix: bool = True
    engine: str = "llama.cpp"
    model: str = "fake-coder"
    requests: list[dict[str, Any]] = field(default_factory=list)


def _flatten(body: dict[str, Any], drop_system_blocks: bool) -> str:
    parts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list) and not drop_system_blocks:
        parts.extend(str(b.get("text", "")) for b in system if isinstance(b, dict))
    if isinstance(body.get("instructions"), str):
        parts.append(body["instructions"])
    for tool in body.get("tools") or []:
        parts.append(json.dumps(tool))
    convo = body.get("messages", body.get("input"))
    if isinstance(convo, str):
        parts.append(convo)
    for msg in convo if isinstance(convo, list) else []:
        content = msg.get("content", msg) if isinstance(msg, dict) else msg
        parts.append(_render(content))
    return "\n".join(parts)


def _render(content: Any) -> str:
    """Render like a chat template: text blocks become their text; other blocks become JSON."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_render(c) for c in content)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return str(content["text"])
        if content.get("type") == "tool_result":
            return "RESULT: " + _render(content.get("content", ""))
        return json.dumps({k: v for k, v in content.items() if k != "cache_control"}, sort_keys=True)
    return json.dumps(content)


NEEDLE = re.compile(r"secret code is ([A-Z0-9-]{4,})")


class FakeServer:
    def __init__(self, behaviour: Behaviour | None = None) -> None:
        self.b = behaviour or Behaviour()
        self._cache_text = ""
        self._lock = threading.Lock()
        handler = type("H", (_Handler,), {"fake": self})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> "FakeServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # --- model simulation -------------------------------------------------------------------
    def run_prompt(self, body: dict[str, Any]) -> tuple[str | None, int, int, str]:
        """Return (error, prompt_tokens_processed, cached_tokens, visible_text)."""
        if self.b.reject_cache_control and "cache_control" in json.dumps(body):
            return "unknown field: cache_control", 0, 0, ""
        text = _flatten(body, self.b.drop_system_blocks)
        tokens = ntok(text)
        want = body.get("max_tokens") or body.get("max_output_tokens") or 0
        if self.b.reject_over_budget and tokens + int(want) > self.b.n_ctx:
            return (f"This model's maximum context length is {self.b.n_ctx} tokens. However, you requested "
                    f"{tokens + int(want)} tokens ({tokens} in the messages, {want} in the completion)."), tokens, 0, ""
        if tokens > self.b.n_ctx:
            if self.b.overflow == "error":
                return (f"request ({tokens} tokens) exceeds the available context size "
                        f"({self.b.n_ctx} tokens), try increasing it"), tokens, 0, ""
            text = text[-self.b.n_ctx * CHARS_PER_TOKEN:]
            tokens = ntok(text)
        with self._lock:
            common = 0
            if self.b.cache_prefix:
                limit = min(len(text), len(self._cache_text))
                while common < limit and text[common] == self._cache_text[common]:
                    common += 1
            self._cache_text = text
        cached = common // CHARS_PER_TOKEN
        if self.b.prefill_tps:
            time.sleep(max(0, tokens - cached) / self.b.prefill_tps)
        return None, tokens, cached, text

    def answer(self, visible: str, last_user: str) -> str:
        found = NEEDLE.findall(visible)
        if "secret code" in last_user and "?" in last_user:
            return f"The secret code is {found[0]}." if found else "I do not see any secret code."
        if "status code" in last_user.lower():
            m = re.search(r"status code: (\S+)", visible)
            return f"{m.group(1)}" if m else "I do not know."
        if "tool result" in last_user.lower():
            return "Done."
        return "OK"

    def sleep_decode(self, n: int) -> None:
        if self.b.decode_tps:
            time.sleep(n / self.b.decode_tps)


def _last_user_text(body: dict[str, Any]) -> str:
    convo = body.get("messages", body.get("input"))
    if isinstance(convo, str):
        return convo
    for msg in reversed(convo or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "function_call_output":
            return "tool result " + str(msg.get("output"))
        if msg.get("role") in ("user", "tool"):
            c = msg.get("content")
            if isinstance(c, str):
                return ("tool result " + c) if msg.get("role") == "tool" else c
            if isinstance(c, list):
                texts = []
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        texts.append("tool result " + json.dumps(blk.get("content")))
                    elif isinstance(blk, dict) and isinstance(blk.get("text"), str):
                        texts.append(blk["text"])
                return " ".join(texts)
    return ""


def _wants_tool(body: dict[str, Any]) -> bool:
    """Call a tool when tools are offered and no tool result is in the conversation yet."""
    if not body.get("tools"):
        return False
    convo = json.dumps(body.get("messages", body.get("input")))
    return not any(m in convo for m in ('"tool_result"', '"role": "tool"', '"function_call_output"'))


class _Handler(BaseHTTPRequestHandler):
    fake: FakeServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _json(self, status: int, data: Any) -> None:
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    # --- discovery ------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        b = self.fake.b
        p = self.path.split("?")[0]
        if p == "/v1/models":
            entry: dict[str, Any] = {"id": b.model, "object": "model", "owned_by": "fake"}
            if b.engine == "llama.cpp":
                entry["owned_by"] = "llamacpp"
                entry["meta"] = {"n_ctx": b.n_ctx, "n_ctx_train": 131072}
            if b.engine == "vllm":
                entry["owned_by"] = "vllm"
                entry["max_model_len"] = b.n_ctx
            if b.engine == "ollama":
                entry["owned_by"] = "library"
            return self._json(200, {"object": "list", "data": [entry]})
        if p == "/props" and b.engine == "llama.cpp":
            return self._json(200, {"build_info": "b9999-fake", "default_generation_settings": {"n_ctx": b.n_ctx},
                                    "chat_template_caps": {"supports_tool_calls": True}})
        if p == "/api/version" and b.engine == "ollama":
            return self._json(200, {"version": "0.99.0"})
        if p == "/api/ps" and b.engine == "ollama":
            return self._json(200, {"models": [{"name": b.model, "context_length": b.n_ctx}]})
        if p == "/version" and b.engine == "vllm":
            return self._json(200, {"version": "0.99.0"})
        if p == "/api/v0/models" and b.engine == "lmstudio":
            return self._json(200, {"data": [{"id": b.model, "state": "loaded",
                                              "loaded_context_length": b.n_ctx, "max_context_length": 131072}]})
        if p == "/health":
            return self._json(200, {"status": "ok"})
        return self._json(404, {"error": {"message": f"not found: {p}"}})

    def do_POST(self) -> None:  # noqa: N802
        b = self.fake.b
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        p = self.path.split("?")[0]
        b.requests.append({"path": p, "body": body})
        if p == "/v1/messages/count_tokens" and b.anthropic and b.count_tokens:
            return self._json(200, {"input_tokens": ntok(_flatten(body, b.drop_system_blocks))})
        if p == "/v1/messages" and b.anthropic:
            return self._anthropic(body)
        if p == "/v1/chat/completions" and b.openai:
            return self._openai(body)
        if p == "/v1/responses" and b.responses:
            return self._responses(body)
        if p == "/api/show" and b.engine == "ollama":
            return self._json(200, {"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 131072},
                                    "capabilities": ["completion", "tools"]})
        return self._json(404, {"error": {"message": f"not found: {p}"}})

    # --- anthropic ----------------------------------------------------------------------
    def _anthropic(self, body: dict[str, Any]) -> None:
        f = self.fake
        err, ptok, cached, visible = f.run_prompt(body)
        if err:
            return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": err}})
        use_tool = f.b.tool_calls and _wants_tool(body)
        tool = _pick_tool(body["tools"]) if use_tool else None
        text = "" if use_tool else f.answer(visible, _last_user_text(body))
        out_tokens = 12 if use_tool else ntok(text)
        f.sleep_decode(out_tokens)
        tool_input = _example_input(tool) if tool else {}
        usage = {"input_tokens": ptok - cached, "cache_read_input_tokens": cached, "output_tokens": out_tokens}
        mid = "msg_" + uuid.uuid4().hex[:12]
        if not body.get("stream"):
            content: list[dict[str, Any]] = []
            if tool:
                content.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:12],
                                "name": tool["name"], "input": tool_input})
            else:
                content.append({"type": "text", "text": text})
            return self._json(200, {"id": mid, "type": "message", "role": "assistant", "model": body.get("model"),
                                    "content": content, "stop_reason": "tool_use" if tool else "end_turn",
                                    "stop_sequence": None, "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def ev(name: str, data: dict[str, Any]) -> None:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        ev("message_start", {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": body.get("model"), "content": [],
            "stop_reason": None, "usage": {**usage, "output_tokens": 0}}})
        if tool:
            if f.b.stream_tool_args != "missing_start":
                ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
                    "type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:12], "name": tool["name"], "input": {}}})
            js = json.dumps(tool_input)
            if f.b.stream_tool_args == "bad_json":
                js = js[:-1] + ",}"
            for i in range(0, len(js), 5):
                ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "input_json_delta", "partial_json": js[i:i + 5]}})
        else:
            ev("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}})
            for i in range(0, len(text), 8):
                ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "text_delta", "text": text[i:i + 8]}})
        ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use" if tool else "end_turn",
                                                                "stop_sequence": None},
                             "usage": {"output_tokens": out_tokens}})
        ev("message_stop", {"type": "message_stop"})

    # --- openai chat ----------------------------------------------------------------------
    def _openai(self, body: dict[str, Any]) -> None:
        f = self.fake
        err, ptok, cached, visible = f.run_prompt(body)
        if err:
            return self._json(400, {"error": {"message": err, "type": "exceed_context_size_error", "code": 400}})
        use_tool = f.b.tool_calls and _wants_tool(body)
        tool = _pick_tool(body["tools"])["function"] if use_tool else None
        text = "" if use_tool else f.answer(visible, _last_user_text(body))
        out_tokens = 12 if use_tool else ntok(text)
        f.sleep_decode(out_tokens)
        usage = {"prompt_tokens": ptok, "completion_tokens": out_tokens,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        if not body.get("stream"):
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool:
                msg["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
                    "name": tool["name"], "arguments": json.dumps(_example_input({"input_schema": tool.get("parameters")}))}}]
            return self._json(200, {"id": "chatcmpl-1", "object": "chat.completion", "model": body.get("model"),
                                    "choices": [{"index": 0, "message": msg,
                                                 "finish_reason": "tool_calls" if tool else "stop"}],
                                    "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta: dict[str, Any], finish: str | None = None, with_usage: bool = False) -> None:
            data: dict[str, Any] = {"id": "chatcmpl-1", "object": "chat.completion.chunk",
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if with_usage:
                data["usage"] = usage
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        if tool:
            args = json.dumps(_example_input({"input_schema": tool.get("parameters")}))
            if f.b.stream_tool_args == "bad_json":
                args = args[:-1] + ",}"
            first = {"index": 0, "id": "call_1", "type": "function", "function": {"name": tool["name"], "arguments": ""}}
            if f.b.stream_tool_args == "missing_start":
                first = {"index": 0, "function": {"arguments": ""}}
            chunk({"role": "assistant", "tool_calls": [first]})
            for i in range(0, len(args), 5):
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + 5]}}]})
            chunk({}, "tool_calls", True)
        else:
            chunk({"role": "assistant", "content": ""})
            for i in range(0, len(text), 8):
                chunk({"content": text[i:i + 8]})
            chunk({}, "stop", True)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # --- openai responses -----------------------------------------------------------------
    def _responses(self, body: dict[str, Any]) -> None:
        f = self.fake
        err, ptok, cached, visible = f.run_prompt(body)
        if err:
            return self._json(400, {"error": {"message": err, "type": "invalid_request_error"}})
        use_tool = f.b.tool_calls and _wants_tool(body)
        tool = _pick_tool(body["tools"]) if use_tool else None
        text = "" if use_tool else f.answer(visible, _last_user_text(body))
        usage = {"input_tokens": ptok, "output_tokens": 12, "input_tokens_details": {"cached_tokens": cached}}
        output: list[dict[str, Any]] = []
        if tool:
            args = json.dumps(_example_input({"input_schema": tool.get("parameters")}))
            if f.b.stream_tool_args == "bad_json":
                args = args[:-1] + ",}"
            output.append({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": tool.get("name"),
                           "arguments": args, "status": "completed"})
        else:
            output.append({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": text, "annotations": []}]})
        resp = {"id": "resp_1", "object": "response", "status": "completed", "model": body.get("model"),
                "output": output, "usage": usage}
        if not body.get("stream"):
            return self._json(200, resp)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def ev(name: str, data: dict[str, Any]) -> None:
            self.wfile.write(f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n".encode())
            self.wfile.flush()

        ev("response.created", {"response": {**resp, "status": "in_progress", "output": []}})
        item = output[0]
        ev("response.output_item.added", {"output_index": 0, "item": {**item, "arguments": ""} if tool else item})
        if tool:
            ev("response.function_call_arguments.delta", {"output_index": 0, "item_id": "fc_1", "delta": item["arguments"]})
            ev("response.function_call_arguments.done", {"output_index": 0, "item_id": "fc_1", "arguments": item["arguments"]})
        else:
            ev("response.output_text.delta", {"output_index": 0, "item_id": "msg_1", "content_index": 0, "delta": text})
        ev("response.output_item.done", {"output_index": 0, "item": item})
        ev("response.completed", {"response": resp})


def _pick_tool(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer a harmless read tool so live agents only read a file when driven by the fake."""
    for t in tools:
        name = t.get("name") or (t.get("function") or {}).get("name")
        if name in ("Read", "read_file", "read"):
            return t
    return tools[0]


def _example_input(tool: dict[str, Any] | None) -> dict[str, Any]:
    schema = (tool or {}).get("input_schema") or {}
    props = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for name, spec in props.items():
        t = spec.get("type") if isinstance(spec, dict) else None
        out[name] = 7 if t in ("integer", "number") else (True if t == "boolean" else "README.md")
    return out


def _main(argv: list[str]) -> None:
    """``python tests/fakeserver.py PORT [field=value ...]`` serves a fake engine on 127.0.0.1:PORT."""
    kwargs: dict[str, Any] = {}
    for arg in argv[1:]:
        key, _, raw = arg.partition("=")
        default = getattr(Behaviour(), key)
        kwargs[key] = (raw.lower() in ("1", "true", "yes")) if isinstance(default, bool) else type(default)(raw)
    srv = FakeServer(Behaviour(**kwargs))
    srv.httpd.server_close()
    srv.httpd = ThreadingHTTPServer(("127.0.0.1", int(argv[0])), type("H", (_Handler,), {"fake": srv}))
    print(f"fake server on 127.0.0.1:{argv[0]} {kwargs}", flush=True)
    srv.httpd.serve_forever()


if __name__ == "__main__":
    import sys

    _main(sys.argv[1:] or ["18081", "n_ctx=200000"])
