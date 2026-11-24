"""Wire-conformance probes: does the server speak the exact dialect the harness sends?

Each probe separates two failure classes, because the fixes differ completely:
  * server/wire faults (FAIL): rejected fields, broken SSE grammar, tool arguments that do not
    assemble into JSON. Fixed on the server side or with a harness setting.
  * model faults (WARN): the wire is fine but the model ignored the tool or the tool result.
    Fixed by choosing a different model, not by configuration.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .client import Client, as_dict, StreamReply
from .engines import tools_fix
from .load import parse_window
from .model import EngineInfo, Finding, Status

JSON = dict[str, Any]

CODE = "7Q4-XK"
TOOL_NAME = "lookup_ticket"
TOOL_DESC = "Look up a support ticket by its numeric id and return its status code."
TOOL_SCHEMA: JSON = {"type": "object", "properties": {"ticket_id": {"type": "integer", "description": "ticket id"}},
                     "required": ["ticket_id"]}
ASK_TOOL = "Use the lookup_ticket tool to look up ticket 42. Do not answer from memory."
ASK_RESULT = "What exact status code did the tool return? Reply with the code only."

# The request features Claude Code 2.1.330 actually sends (captured with `localready proxy`).
# Only the field shapes are reproduced; none of Claude Code's prompt text is.
CLAUDE_CODE_FEATURES: dict[str, Callable[[JSON], None]] = {}


def _feature(name: str) -> Callable[[Callable[[JSON], None]], Callable[[JSON], None]]:
    def deco(fn: Callable[[JSON], None]) -> Callable[[JSON], None]:
        CLAUDE_CODE_FEATURES[name] = fn
        return fn
    return deco


@_feature("system as text blocks with cache_control")
def _f_system(body: JSON) -> None:
    body["system"] = [{"type": "text", "text": "x-client-header: probe;"},
                      {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}},
                      {"type": "text", "text": "Answer briefly.", "cache_control": {"type": "ephemeral"}}]


@_feature("a role:'system' message inside messages")
def _f_system_message(body: JSON) -> None:
    body["messages"].append({"role": "system", "content": [
        {"type": "text", "text": "# Environment\nWorking directory: /tmp/project", "cache_control": {"type": "ephemeral"}}]})
    body["messages"].append({"role": "user", "content": "Reply with the word READY."})


@_feature("thinking: adaptive")
def _f_thinking(body: JSON) -> None:
    body["thinking"] = {"type": "adaptive", "display": "omitted"}


@_feature("context_management")
def _f_ctx(body: JSON) -> None:
    body["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}


@_feature("output_config.effort")
def _f_effort(body: JSON) -> None:
    body["output_config"] = {"effort": "high"}


@_feature("metadata.user_id")
def _f_meta(body: JSON) -> None:
    body["metadata"] = {"user_id": "localready-probe"}


@_feature("max_tokens 32000")
def _f_max(body: JSON) -> None:
    body["max_tokens"] = 32000


@_feature("tools with cache_control")
def _f_tools(body: JSON) -> None:
    body["tools"] = [{"name": TOOL_NAME, "description": TOOL_DESC, "input_schema": TOOL_SCHEMA,
                      "cache_control": {"type": "ephemeral"}}]


FEATURE_FIX = {
    "max_tokens 32000": "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192   (Claude Code asks for 32000 output tokens per request)",
    "thinking: adaptive": "put a translating proxy (e.g. LiteLLM) in front, or use an engine whose /v1/messages accepts `thinking`",
}


def _cc_body(model: str, features: list[str]) -> JSON:
    body: JSON = {"model": model, "max_tokens": 512, "stream": False,
                  "messages": [{"role": "user", "content": "Hello."}]}
    if "a role:'system' message inside messages" not in features:
        body["messages"] = [{"role": "user", "content": "Reply with the word READY."}]
    for name in features:
        CLAUDE_CODE_FEATURES[name](body)
    return body


# ----------------------------------------------------------------------------------------------
# Anthropic Messages (Claude Code)
# ----------------------------------------------------------------------------------------------

def anthropic_probes(c: Client, model: str, engine: EngineInfo) -> list[Finding]:
    out: list[Finding] = []
    basic = c.accepts("/v1/messages", {"model": model, "max_tokens": 512,
                                       "messages": [{"role": "user", "content": "Reply with the word READY."}]})
    if basic.status in (404, 405) or (basic.status == 0):
        out.append(Finding("anthropic.endpoint", "Anthropic Messages endpoint (/v1/messages)", Status.FAIL,
                           f"server answered {basic.status or 'nothing'}: {basic.error or ''}".strip(),
                           fix=_no_messages_fix(engine)))
        return out
    if not basic.ok or not _starts_message(basic):
        out.append(Finding("anthropic.endpoint", "Anthropic Messages endpoint (/v1/messages)", Status.FAIL,
                           f"HTTP {basic.status}: {basic.error or 'stream does not start with message_start'}"))
        return out
    out.append(Finding("anthropic.endpoint", "Anthropic Messages endpoint (/v1/messages)", Status.PASS,
                       "streams an Anthropic message"))
    out.append(_claude_code_shape(c, model))
    out.append(_output_budget(c, model, engine))
    out.extend(_anthropic_stream_tool(c, model, engine))
    out.append(_anthropic_roundtrip(c, model))
    ct = c.post("/v1/messages/count_tokens", {"model": model, "messages": [{"role": "user", "content": "hello"}]},
                timeout=30)
    if ct.ok and isinstance(ct.body, dict) and isinstance(ct.body.get("input_tokens"), int):
        out.append(Finding("anthropic.count_tokens", "Token counting (/v1/messages/count_tokens)", Status.PASS,
                           f"input_tokens={ct.body['input_tokens']}"))
    else:
        out.append(Finding("anthropic.count_tokens", "Token counting (/v1/messages/count_tokens)", Status.INFO,
                           f"not served (HTTP {ct.status}); Claude Code falls back to estimates"))
    return out


def _no_messages_fix(engine: EngineInfo) -> str:
    return {
        "llama.cpp": "upgrade llama.cpp: recent llama-server builds serve /v1/messages",
        "ollama": "upgrade Ollama: current releases serve /v1/messages (docs.ollama.com/api/anthropic-compatibility)",
        "lmstudio": "upgrade LM Studio: current releases serve /v1/messages (lmstudio.ai/docs/developer/anthropic-compat)",
    }.get(engine.name, "put an Anthropic-compatible proxy (e.g. LiteLLM) in front of this server")


def _anthropic_message_ok(body: Any) -> bool:
    return isinstance(body, dict) and body.get("type") == "message" and isinstance(body.get("content"), list)


def _starts_message(s: StreamReply) -> bool:
    first = s.events[0][1] if s.events else None
    return isinstance(first, dict) and first.get("type") == "message_start"


def _claude_code_shape(c: Client, model: str) -> Finding:
    title = "Claude Code request shape (2.1.330 fields)"
    names = list(CLAUDE_CODE_FEATURES)
    full = c.accepts("/v1/messages", _cc_body(model, names))
    if full.ok and _starts_message(full):
        return Finding("anthropic.claude_code_shape", title, Status.PASS,
                       f"accepted all {len(names)} fields Claude Code sends")
    rejected: list[str] = []
    messages: dict[str, str] = {}
    for name in names:
        r = c.accepts("/v1/messages", _cc_body(model, [name]))
        if not (r.ok and _starts_message(r)):
            rejected.append(name)
            messages[name] = r.error or f"HTTP {r.status}"
    if not rejected:
        return Finding("anthropic.claude_code_shape", title, Status.FAIL,
                       f"each field is accepted alone but the combination fails: {full.error or full.status}")
    fixes = [FEATURE_FIX[n] for n in rejected if n in FEATURE_FIX]
    detail = "; ".join(f"{n}: {messages[n]}" for n in rejected)
    return Finding("anthropic.claude_code_shape", title, Status.FAIL, f"rejected {detail}",
                   fix=" ; ".join(fixes) or None, data={"rejected": rejected})


CC_FIRST_REQUEST = 11876
CC_MAX_TOKENS = 32000


def _output_budget(c: Client, model: str, engine: EngineInfo) -> Finding:
    """Does the server require prompt + max_tokens to fit the window? Claude Code asks for 32000."""
    title = "Claude Code's max_tokens 32000 fits the window"
    probe = (engine.window or 262144) * 2
    r = c.accepts("/v1/messages", {"model": model, "max_tokens": probe,
                                   "messages": [{"role": "user", "content": "Reply with the word READY."}]})
    if r.ok:
        return Finding("anthropic.output_budget", title, Status.PASS,
                       f"max_tokens {probe:,} accepted: the server stops at its window by itself",
                       data={"enforced": False})
    w = engine.window
    if w is None:
        w = parse_window(r.error or "")
    need = CC_FIRST_REQUEST + CC_MAX_TOKENS
    cap = max(1024, min(8192, (w or 32768) // 4))
    fix = f"export CLAUDE_CODE_MAX_OUTPUT_TOKENS={cap}"
    if w and need > w:
        return Finding("anthropic.output_budget", title, Status.FAIL,
                       f"server refuses prompt + max_tokens over its window ({r.error}); Claude Code's first request "
                       f"({CC_FIRST_REQUEST:,}) + max_tokens {CC_MAX_TOKENS:,} = {need:,} > {w:,}, so every request fails",
                       fix=fix, data={"enforced": True, "cap": cap})
    return Finding("anthropic.output_budget", title, Status.WARN,
                   f"server refuses prompt + max_tokens over its window ({r.error}); requests start failing once the "
                   f"conversation passes {(w or 0) - CC_MAX_TOKENS:,} tokens, before Claude Code compacts",
                   fix=fix, data={"enforced": True, "cap": cap})


def validate_anthropic_stream(s: StreamReply) -> tuple[list[str], list[JSON]]:
    """Check the Anthropic SSE grammar and return (problems, assembled tool_use blocks)."""
    problems: list[str] = []
    types = [d.get("type") if isinstance(d, dict) else None for _, d in s.events]
    if not types or types[0] != "message_start":
        problems.append(f"first event is {types[0] if types else 'missing'}, expected message_start")
    if "message_stop" not in types:
        problems.append("no message_stop event")
    open_blocks: dict[int, JSON] = {}
    done: list[JSON] = []
    stop_reason = None
    for _, d in s.events:
        if not isinstance(d, dict):
            continue
        t = d.get("type")
        if t == "content_block_start":
            idx = d.get("index")
            block = as_dict(d.get("content_block"))
            if not isinstance(idx, int):
                problems.append("content_block_start without an integer index")
                continue
            open_blocks[idx] = {"block": dict(block), "json": ""}
        elif t == "content_block_delta":
            idx = d.get("index")
            delta = as_dict(d.get("delta"))
            if not isinstance(idx, int) or idx not in open_blocks:
                problems.append(f"content_block_delta for index {idx} that was never started")
                continue
            if delta.get("type") == "input_json_delta":
                open_blocks[idx]["json"] += str(delta.get("partial_json", ""))
        elif t == "content_block_stop":
            idx = d.get("index")
            if not isinstance(idx, int) or idx not in open_blocks:
                problems.append(f"content_block_stop for index {idx} that was never started")
                continue
            entry = open_blocks.pop(idx)
            block = entry["block"]
            if block.get("type") == "tool_use":
                raw = entry["json"] or "{}"
                try:
                    block["input"] = json.loads(raw)
                except ValueError:
                    problems.append(f"tool_use input does not assemble into JSON: {raw[:80]!r}")
                    continue
            done.append(block)
        elif t == "message_delta":
            delta = as_dict(d.get("delta"))
            stop_reason = delta.get("stop_reason", stop_reason)
    if open_blocks:
        problems.append(f"content blocks never stopped: {sorted(open_blocks)}")
    if stop_reason is None and "message_stop" in types:
        problems.append("message_delta never carried a stop_reason")
    tools = [b for b in done if b.get("type") == "tool_use"]
    if tools and stop_reason not in (None, "tool_use"):
        problems.append(f"tool_use emitted but stop_reason is {stop_reason!r}")
    return problems, tools


def _anthropic_stream_tool(c: Client, model: str, engine: EngineInfo) -> list[Finding]:
    title = "Streaming tool call (SSE grammar + input_json_delta)"
    tool = {"name": TOOL_NAME, "description": TOOL_DESC, "input_schema": TOOL_SCHEMA}
    base: JSON = {"model": model, "max_tokens": 1024, "tools": [tool],
                  "messages": [{"role": "user", "content": ASK_TOOL}]}
    forced = c.stream("/v1/messages", {**base, "tool_choice": {"type": "tool", "name": TOOL_NAME}})
    used_forcing = forced.ok
    s = forced if forced.ok else c.stream("/v1/messages", base)
    if not s.ok:
        return [Finding("anthropic.stream_tool", title, Status.FAIL, f"HTTP {s.status}: {s.error}")]
    problems, tools = validate_anthropic_stream(s)
    if problems:
        return [Finding("anthropic.stream_tool", title, Status.FAIL, "; ".join(problems[:4]),
                        data={"events": len(s.events)})]
    if not tools:
        status = Status.FAIL if used_forcing else Status.WARN
        why = ("tool_choice forced the tool but the server streamed no tool_use block"
               if used_forcing else "server rejected tool_choice, and unforced the model answered without the tool")
        return [Finding("anthropic.stream_tool", title, status, why,
                        fix=tools_fix(engine.name, model) if used_forcing else None)]
    args = tools[0].get("input")
    ok_args = isinstance(args, dict) and "ticket_id" in args
    note = "" if used_forcing else " (tool_choice not supported; the model chose the tool on its own)"
    return [Finding("anthropic.stream_tool", title, Status.PASS if ok_args else Status.WARN,
                    f"{len(s.events)} events, tool_use {tools[0].get('name')}({json.dumps(args)}){note}")]


def _anthropic_roundtrip(c: Client, model: str) -> Finding:
    title = "Tool result reaches the model (tool_use -> tool_result)"
    tool = {"name": TOOL_NAME, "description": TOOL_DESC, "input_schema": TOOL_SCHEMA}
    body: JSON = {"model": model, "max_tokens": 1024, "tools": [tool], "messages": [
        {"role": "user", "content": ASK_TOOL},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_probe1", "name": TOOL_NAME,
                                           "input": {"ticket_id": 42}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_probe1",
                                      "content": f"Ticket 42 status code: {CODE}"},
                                     {"type": "text", "text": ASK_RESULT}]},
    ]}
    r = c.post("/v1/messages", body)
    if not r.ok:
        return Finding("anthropic.roundtrip", title, Status.FAIL, f"HTTP {r.status}: {r.error}")
    text = _anthropic_text(r.body)
    if CODE in text:
        return Finding("anthropic.roundtrip", title, Status.PASS, f"model quoted {CODE}")
    return Finding("anthropic.roundtrip", title, Status.WARN,
                   f"model did not quote the tool result {CODE} (got {text[:80]!r}); wire accepted it, "
                   "so this is the model, not the server")


def _anthropic_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    return "".join(b.get("text", "") for b in body.get("content") or [] if isinstance(b, dict) and b.get("type") == "text")


# ----------------------------------------------------------------------------------------------
# OpenAI Chat Completions (OpenCode, Aider, Cline, Continue, ...)
# ----------------------------------------------------------------------------------------------

def _chat_tool() -> JSON:
    return {"type": "function", "function": {"name": TOOL_NAME, "description": TOOL_DESC, "parameters": TOOL_SCHEMA}}


def chat_probes(c: Client, model: str, engine: EngineInfo) -> list[Finding]:
    out: list[Finding] = []
    basic = c.accepts("/v1/chat/completions", {"model": model, "max_tokens": 512,
                                               "messages": [{"role": "user", "content": "Reply with the word READY."}]})
    first = basic.events[0][1] if basic.events else None
    if not basic.ok or not isinstance(first, dict) or "choices" not in first:
        out.append(Finding("chat.endpoint", "OpenAI Chat endpoint (/v1/chat/completions)", Status.FAIL,
                           f"HTTP {basic.status}: {basic.error or 'stream has no choices'}"))
        return out
    out.append(Finding("chat.endpoint", "OpenAI Chat endpoint (/v1/chat/completions)", Status.PASS,
                       "streams chat.completion.chunk"))
    base: JSON = {"model": model, "max_tokens": 1024, "tools": [_chat_tool()],
                  "messages": [{"role": "user", "content": ASK_TOOL}]}
    forced = c.stream("/v1/chat/completions", {**base, "tool_choice": {"type": "function",
                                                                         "function": {"name": TOOL_NAME}}})
    used_forcing = forced.ok
    s = forced if forced.ok else c.stream("/v1/chat/completions", base)
    title = "Streaming tool call (tool_calls deltas assemble)"
    if not s.ok:
        out.append(Finding("chat.stream_tool", title, Status.FAIL, f"HTTP {s.status}: {s.error}"))
    else:
        problems, calls = validate_chat_stream(s)
        if problems:
            out.append(Finding("chat.stream_tool", title, Status.FAIL, "; ".join(problems[:4])))
        elif not calls:
            out.append(Finding("chat.stream_tool", title, Status.FAIL if used_forcing else Status.WARN,
                               "no tool call streamed" + ("" if used_forcing else " (tool_choice rejected; model chose not to)"),
                               fix=tools_fix(engine.name, model) if used_forcing else None))
        else:
            out.append(Finding("chat.stream_tool", title, Status.PASS,
                               f"{calls[0]['name']}({json.dumps(calls[0]['arguments'])})"))
    body: JSON = {"model": model, "max_tokens": 1024, "tools": [_chat_tool()], "messages": [
        {"role": "user", "content": ASK_TOOL},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_probe1", "type": "function", "function": {
            "name": TOOL_NAME, "arguments": json.dumps({"ticket_id": 42})}}]},
        {"role": "tool", "tool_call_id": "call_probe1", "content": f"Ticket 42 status code: {CODE}"},
        {"role": "user", "content": ASK_RESULT},
    ]}
    r = c.post("/v1/chat/completions", body)
    title = "Tool result reaches the model (tool_calls -> role:tool)"
    if not r.ok:
        out.append(Finding("chat.roundtrip", title, Status.FAIL, f"HTTP {r.status}: {r.error}"))
    else:
        text = _chat_text(r.body)
        out.append(Finding("chat.roundtrip", title, Status.PASS if CODE in text else Status.WARN,
                           f"model quoted {CODE}" if CODE in text else
                           f"model did not quote {CODE} (got {text[:80]!r}); the model, not the server"))
    return out


def validate_chat_stream(s: StreamReply) -> tuple[list[str], list[JSON]]:
    problems: list[str] = []
    calls: dict[int, JSON] = {}
    finish = None
    for _, d in s.events:
        if not isinstance(d, dict):
            continue
        for choice in d.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            finish = choice.get("finish_reason") or finish
            delta = as_dict(choice.get("delta"))
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0) if isinstance(tc, dict) else 0
                fn = as_dict(as_dict(tc).get("function"))
                if idx not in calls:
                    if not fn.get("name"):
                        problems.append(f"tool call {idx} started without a function name")
                    calls[idx] = {"name": fn.get("name"), "args": "", "id": as_dict(tc).get("id")}
                elif fn.get("name") and not calls[idx]["name"]:
                    calls[idx]["name"] = fn.get("name")
                calls[idx]["args"] += str(fn.get("arguments") or "")
    out: list[JSON] = []
    for idx, call in sorted(calls.items()):
        try:
            out.append({"name": call["name"], "arguments": json.loads(call["args"] or "{}")})
        except ValueError:
            problems.append(f"tool call {idx} arguments do not assemble into JSON: {call['args'][:80]!r}")
    if calls and finish not in (None, "tool_calls", "function_call"):
        problems.append(f"tool call streamed but finish_reason is {finish!r}")
    return problems, out


def _chat_text(body: Any) -> str:
    try:
        return str(body["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


# ----------------------------------------------------------------------------------------------
# OpenAI Responses (Codex)
# ----------------------------------------------------------------------------------------------

def _resp_tool() -> JSON:
    return {"type": "function", "name": TOOL_NAME, "description": TOOL_DESC, "parameters": TOOL_SCHEMA}


def responses_probes(c: Client, model: str, engine: EngineInfo) -> list[Finding]:
    out: list[Finding] = []
    basic = c.accepts("/v1/responses", {"model": model, "max_output_tokens": 512, "input": "Reply with the word READY."})
    first = basic.events[0][1] if basic.events else None
    if not basic.ok or not isinstance(first, dict) or not str(first.get("type", "")).startswith("response."):
        out.append(Finding("responses.endpoint", "OpenAI Responses endpoint (/v1/responses)", Status.FAIL,
                           f"HTTP {basic.status}: {basic.error or 'stream does not start with a response.* event'}",
                           fix="Codex needs /v1/responses: upgrade the server or put a Responses-compatible proxy in front"))
        return out
    out.append(Finding("responses.endpoint", "OpenAI Responses endpoint (/v1/responses)", Status.PASS,
                       f"streams {first.get('type')}"))
    base: JSON = {"model": model, "max_output_tokens": 1024, "tools": [_resp_tool()], "input": ASK_TOOL}
    forced = c.stream("/v1/responses", {**base, "tool_choice": {"type": "function", "name": TOOL_NAME}})
    used_forcing = forced.ok
    s = forced if forced.ok else c.stream("/v1/responses", base)
    title = "Streaming function call (response.* events)"
    if not s.ok:
        out.append(Finding("responses.stream_tool", title, Status.FAIL, f"HTTP {s.status}: {s.error}"))
    else:
        problems, calls = validate_responses_stream(s)
        if problems:
            out.append(Finding("responses.stream_tool", title, Status.FAIL, "; ".join(problems[:4])))
        elif not calls:
            out.append(Finding("responses.stream_tool", title, Status.FAIL if used_forcing else Status.WARN,
                               "no function_call streamed", fix=tools_fix(engine.name, model) if used_forcing else None))
        else:
            out.append(Finding("responses.stream_tool", title, Status.PASS,
                               f"{calls[0]['name']}({json.dumps(calls[0]['arguments'])})"))
    body: JSON = {"model": model, "max_output_tokens": 1024, "tools": [_resp_tool()], "input": [
        {"role": "user", "content": ASK_TOOL},
        {"type": "function_call", "call_id": "call_probe1", "name": TOOL_NAME, "arguments": json.dumps({"ticket_id": 42})},
        {"type": "function_call_output", "call_id": "call_probe1", "output": f"Ticket 42 status code: {CODE}"},
        {"role": "user", "content": ASK_RESULT},
    ]}
    r = c.post("/v1/responses", body)
    title = "Tool result reaches the model (function_call_output)"
    if not r.ok:
        out.append(Finding("responses.roundtrip", title, Status.FAIL, f"HTTP {r.status}: {r.error}"))
    else:
        text = _responses_text(r.body)
        out.append(Finding("responses.roundtrip", title, Status.PASS if CODE in text else Status.WARN,
                           f"model quoted {CODE}" if CODE in text else
                           f"model did not quote {CODE} (got {text[:80]!r}); the model, not the server"))
    return out


def validate_responses_stream(s: StreamReply) -> tuple[list[str], list[JSON]]:
    problems: list[str] = []
    types = [(d.get("type") if isinstance(d, dict) else None) or e for e, d in s.events]
    if "response.completed" not in types:
        problems.append("no response.completed event")
    calls: list[JSON] = []
    final: Any = None
    for _, d in s.events:
        if isinstance(d, dict) and d.get("type") == "response.completed":
            final = d.get("response")
    items = final.get("output") if isinstance(final, dict) else None
    if not isinstance(items, list):
        items = [d.get("item") for _, d in s.events
                 if isinstance(d, dict) and d.get("type") == "response.output_item.done"]
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            try:
                calls.append({"name": item.get("name"), "arguments": json.loads(item.get("arguments") or "{}")})
            except ValueError:
                problems.append(f"function_call arguments are not JSON: {str(item.get('arguments'))[:80]!r}")
    return problems, calls


def _responses_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    if isinstance(body.get("output_text"), str):
        return str(body["output_text"])
    parts: list[str] = []
    for item in body.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for blk in item.get("content") or []:
                if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
    return "".join(parts)
