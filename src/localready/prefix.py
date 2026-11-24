"""Prompt-prefix stability between consecutive agent requests.

Local engines (llama.cpp, Ollama, LM Studio, vLLM, MLX) reuse their KV cache only for the
longest common *token prefix* of two requests. If something near the top of the prompt changes
on every request (a timestamp, a per-request hash, a reordered tool list), every turn
re-processes the whole prompt. On a laptop that is the difference between 2 s and 90 s per turn.

This module flattens a request into the order a chat template renders it (system, tools,
messages) and reports where two requests first differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

JSON = dict[str, Any]


@dataclass(frozen=True)
class Segment:
    """One labelled piece of the rendered prompt, e.g. ``system[1]`` or ``messages[3]``."""

    label: str
    text: str


def _text_of(content: Any) -> str:
    """Render content the way a chat template sees it.

    A string and a list of text blocks with the same text render identically (Claude Code
    switches between the two forms for the same message between turns), and API-only
    fields such as ``cache_control`` or block ids never reach the model.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_text(b) for b in content)
    if isinstance(content, dict):
        return _block_text(content)
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return json.dumps(block, ensure_ascii=False, separators=(",", ":"))
    kind = block.get("type")
    if kind in ("text", "input_text", "output_text") and isinstance(block.get("text"), str):
        return str(block["text"])
    if kind == "thinking" and isinstance(block.get("thinking"), str):
        return str(block["thinking"])
    if kind == "tool_use":
        return f"<tool_use {block.get('name')} {_dumps(block.get('input'))}>"
    if kind == "tool_result":
        return f"<tool_result {_text_of(block.get('content', ''))}>"
    if kind == "function_call":
        return f"<function_call {block.get('name')} {block.get('arguments')}>"
    if kind == "function_call_output":
        return f"<function_call_output {_text_of(block.get('output', ''))}>"
    if kind in ("image", "input_image", "document"):
        return f"<{kind}>"
    if "content" in block and ("role" in block or kind == "message"):
        return _text_of(block["content"])
    return _dumps(_strip_cache(block))


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def segments(body: JSON) -> list[Segment]:
    """Flatten an Anthropic Messages, OpenAI Chat or OpenAI Responses request into segments.

    The order follows how chat templates render a prompt: system first, then tool
    definitions, then the conversation. Only fields that reach the model are included.
    """
    out: list[Segment] = []
    system = body.get("system")
    if isinstance(system, str):
        out.append(Segment("system", system))
    elif isinstance(system, list):
        for i, block in enumerate(system):
            text = block.get("text") if isinstance(block, dict) else None
            out.append(Segment(f"system[{i}]", text if isinstance(text, str) else _text_of(block)))
    instructions = body.get("instructions")
    if isinstance(instructions, str):
        out.append(Segment("instructions", instructions))
    tools = body.get("tools")
    if isinstance(tools, list):
        for i, tool in enumerate(tools):
            name = _tool_name(tool)
            out.append(Segment(f"tools[{i}:{name}]", _dumps(_strip_cache(tool))))
    convo = body.get("messages")
    key = "messages"
    if convo is None:
        convo = body.get("input")
        key = "input"
    if isinstance(convo, str):
        out.append(Segment(key, convo))
    elif isinstance(convo, list):
        for i, msg in enumerate(convo):
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type") or "?"
                text = _text_of(msg["content"]) if "content" in msg else _block_text(msg)
                if msg.get("tool_calls"):
                    text += _dumps(msg["tool_calls"])
                out.append(Segment(f"{key}[{i}:{role}]", text))
            else:
                out.append(Segment(f"{key}[{i}]", _text_of(msg)))
    return out


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return "?"
    name = tool.get("name")
    if isinstance(name, str):
        return name
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return str(fn["name"])
    return "?"


def _strip_cache(value: Any) -> Any:
    """Drop ``cache_control`` markers: they are API hints, never rendered into the prompt."""
    if isinstance(value, dict):
        return {k: _strip_cache(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache(v) for v in value]
    return value


@dataclass(frozen=True)
class Divergence:
    """Where request B first differs from request A, measured on the rendered prompt."""

    total_a: int
    total_b: int
    common: int
    segment: str | None
    offset_in_segment: int
    before: str
    after_a: str
    after_b: str

    @property
    def reuse(self) -> float:
        """Share of B's prompt an ideal prefix cache could reuse from A (0.0 to 1.0)."""
        return self.common / self.total_b if self.total_b else 1.0

    @property
    def identical_prefix(self) -> bool:
        """True when B simply extends A (the healthy agent-loop case)."""
        return self.common >= self.total_a


def _render(segs: list[Segment]) -> tuple[str, list[tuple[int, Segment]]]:
    parts: list[str] = []
    index: list[tuple[int, Segment]] = []
    pos = 0
    for seg in segs:
        index.append((pos, seg))
        parts.append(seg.text)
        pos += len(seg.text)
    return "".join(parts), index


def divergence(a: JSON, b: JSON, context: int = 40) -> Divergence:
    """Compare two requests and locate the first character where their prompts differ."""
    text_a, _ = _render(segments(a))
    text_b, index_b = _render(segments(b))
    limit = min(len(text_a), len(text_b))
    common = 0
    while common < limit and text_a[common] == text_b[common]:
        common += 1
    seg_label: str | None = None
    offset = 0
    if common < len(text_b):
        for start, seg in index_b:
            if start <= common < start + len(seg.text):
                seg_label, offset = seg.label, common - start
                break
    return Divergence(
        total_a=len(text_a),
        total_b=len(text_b),
        common=common,
        segment=seg_label,
        offset_in_segment=offset,
        before=text_b[max(0, common - context):common],
        after_a=text_a[common:common + context],
        after_b=text_b[common:common + context],
    )
