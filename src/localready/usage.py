"""Read token usage out of every response shape a local server may return."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    prompt_ms: float | None = None
    decode_ms: float | None = None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def merge(u: Usage, data: Any) -> None:
    """Fold any usage-like object (Anthropic, OpenAI, Responses, Ollama, llama.cpp) into ``u``."""
    if not isinstance(data, dict):
        return
    usage = data.get("usage")
    if isinstance(usage, dict):
        # Anthropic counts only *uncached* tokens in input_tokens; cache reads are separate.
        cache_read = _int(usage.get("cache_read_input_tokens"))
        inp = _int(usage.get("input_tokens"))
        if inp is None:
            inp = _int(usage.get("prompt_tokens"))
        if inp is not None:
            u.input_tokens = inp + (cache_read or 0) if "cache_read_input_tokens" in usage else inp
        out = _int(usage.get("output_tokens"))
        if out is None:
            out = _int(usage.get("completion_tokens"))
        if out is not None:
            u.output_tokens = out
        if cache_read is not None:
            u.cached_tokens = cache_read
        for details_key in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, dict) and _int(details.get("cached_tokens")) is not None:
                u.cached_tokens = _int(details.get("cached_tokens"))
    # Ollama native
    if _int(data.get("prompt_eval_count")) is not None:
        u.input_tokens = _int(data.get("prompt_eval_count"))
    if _int(data.get("eval_count")) is not None:
        u.output_tokens = _int(data.get("eval_count"))
    if isinstance(data.get("prompt_eval_duration"), int):
        u.prompt_ms = data["prompt_eval_duration"] / 1e6
    if isinstance(data.get("eval_duration"), int):
        u.decode_ms = data["eval_duration"] / 1e6
    # llama.cpp timings: prompt_n is what it actually had to process, cache_n what it reused.
    timings = data.get("timings")
    if isinstance(timings, dict):
        cache_n = _int(timings.get("cache_n"))
        prompt_n = _int(timings.get("prompt_n"))
        if cache_n is not None:
            u.cached_tokens = cache_n
        if prompt_n is not None and u.input_tokens is None:
            u.input_tokens = prompt_n + (cache_n or 0)
        if isinstance(timings.get("prompt_ms"), (int, float)):
            u.prompt_ms = float(timings["prompt_ms"])
        if isinstance(timings.get("predicted_ms"), (int, float)):
            u.decode_ms = float(timings["predicted_ms"])
    # Anthropic stream wrappers and Responses API wrappers
    for key in ("message", "response"):
        inner = data.get(key)
        if isinstance(inner, dict):
            merge(u, inner)


def usage_from_body(text: str) -> Usage:
    u = Usage()
    try:
        merge(u, json.loads(text))
    except ValueError:
        # Ollama native streaming is newline-delimited JSON; the last object carries the counts.
        for line in text.splitlines():
            try:
                merge(u, json.loads(line))
            except ValueError:
                continue
    return u


def usage_from_sse(text: str) -> Usage:
    u = Usage()
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            merge(u, json.loads(payload))
        except ValueError:
            continue
    return u
