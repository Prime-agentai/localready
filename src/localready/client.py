"""Minimal HTTP client for model servers: JSON calls, SSE streams, honest timings."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

JSON = dict[str, Any]


@dataclass
class Reply:
    status: int
    body: Any
    text: str
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and self.error is None


@dataclass
class StreamReply:
    status: int
    events: list[tuple[str | None, Any]] = field(default_factory=list)
    raw: str = ""
    first_byte_s: float | None = None
    first_token_s: float | None = None
    total_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and self.error is None


class Client:
    def __init__(self, base_url: str, api_key: str = "localready", timeout: float = 600.0) -> None:
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "authorization": f"Bearer {self.api_key}",
        }

    def get(self, path: str, timeout: float = 10.0) -> Reply:
        return self._do(urllib.request.Request(self.base + path, headers=self._headers()), timeout)

    def post(self, path: str, body: JSON, timeout: float | None = None) -> Reply:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers=self._headers(), method="POST")
        return self._do(req, timeout or self.timeout)

    def _do(self, req: urllib.request.Request, timeout: float) -> Reply:
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("utf-8", errors="replace")
                return Reply(r.status, _parse(text), text, time.monotonic() - start)
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace")
            return Reply(e.code, _parse(text), text, time.monotonic() - start, error_message(_parse(text), text))
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            return Reply(0, None, "", time.monotonic() - start, f"{type(e).__name__}: {reason}")

    def accepts(self, path: str, body: JSON, events: int = 1, timeout: float | None = None) -> StreamReply:
        """Stream ``body`` but hang up after ``events`` SSE events.

        For probes that only need to know whether the server *accepts* a request: a CPU server
        asked for max_tokens 32000 would otherwise generate for many minutes. Servers cancel the
        generation when the client disconnects (llama.cpp logs "cancel task").
        """
        return self.stream(path, body, timeout=timeout, max_events=events)

    def stream(self, path: str, body: JSON, timeout: float | None = None,
               max_events: int | None = None) -> StreamReply:
        """POST with ``stream: true`` and collect SSE events, timing the first content delta."""
        req = urllib.request.Request(self.base + path, data=json.dumps({**body, "stream": True}).encode(),
                                     headers=self._headers(), method="POST")
        out = StreamReply(status=0)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                out.status = r.status
                for event, data, raw in _sse(r):
                    if out.first_byte_s is None:
                        out.first_byte_s = time.monotonic() - start
                    if out.first_token_s is None and _is_content_delta(event, data):
                        out.first_token_s = time.monotonic() - start
                    out.events.append((event, data))
                    out.raw += raw
                    if max_events is not None and len(out.events) >= max_events:
                        break
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace")
            out.status, out.raw = e.code, text
            out.error = error_message(_parse(text), text)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            out.error = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
        out.total_s = time.monotonic() - start
        return out


def as_dict(value: Any) -> dict[str, Any]:
    """``value`` if it is a dict, else an empty dict (for walking untrusted JSON)."""
    return value if isinstance(value, dict) else {}


def _parse(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def error_message(body: Any, text: str) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if isinstance(body.get("detail"), str):
            return str(body["detail"])
    return (text or "").strip()[:400] or "empty error body"


def _sse(resp: Any) -> Iterator[tuple[str | None, Any, str]]:
    event: str | None = None
    data_lines: list[str] = []
    raw_lines: list[str] = []
    for bline in resp:
        line = bline.decode("utf-8", errors="replace").rstrip("\r\n")
        raw_lines.append(line + "\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                yield event, (None if payload == "[DONE]" else _parse(payload) if payload else None), "".join(raw_lines)
            event, data_lines, raw_lines = None, [], []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = "\n".join(data_lines)
        yield event, (None if payload == "[DONE]" else _parse(payload)), "".join(raw_lines)


def _is_content_delta(event: str | None, data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    kind = data.get("type") or event
    if kind == "content_block_delta":
        return True
    if isinstance(kind, str) and kind.startswith("response.") and kind.endswith(".delta"):
        return True
    for choice in data.get("choices") or []:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict) and (delta.get("content") or delta.get("tool_calls")
                                         or delta.get("reasoning_content") or delta.get("reasoning")):
            return True
    return False
