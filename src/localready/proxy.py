"""A transparent recording proxy between a coding agent and a local model server.

Point the agent at the proxy (``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL``) and the proxy at
the server. Every request is forwarded unchanged and streamed back unchanged. For each model
call the proxy records what an agent-on-local-model user actually needs to know:

* prompt and output tokens as the server reported them, and how many it served from cache;
* time to first byte and total time;
* how much of the prompt an ideal prefix cache could have reused from the previous call,
  and, when that is low, exactly which segment changed and what the change looks like;
* context-overflow and other server errors, verbatim.

Bodies and credentials are never written to disk unless ``--save-bodies`` is given, and
authorization headers are never written at all.
"""

from __future__ import annotations

import http.client
import json
import socketserver
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

from .prefix import JSON, divergence
from .usage import Usage, usage_from_body, usage_from_sse

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
MODEL_PATHS = ("/v1/messages", "/v1/chat/completions", "/v1/responses", "/v1/completions",
               "/api/chat", "/api/generate")


@dataclass
class CallRecord:
    n: int
    path: str
    model: str | None
    status: int
    stream: bool
    ttfb_s: float
    total_s: float
    request_bytes: int
    usage: Usage
    reuse: float | None = None
    prefix_ok: bool | None = None
    diverged_at: str | None = None
    diverged_before: str | None = None
    diverged_was: str | None = None
    diverged_now: str | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> JSON:
        data = asdict(self)
        data["usage"] = asdict(self.usage)
        return data


class Recorder:
    """Thread-safe store of call records plus the previous body per endpoint for diffing."""

    def __init__(self, log_path: Path | None, out: TextIO, save_bodies: Path | None) -> None:
        self._lock = threading.Lock()
        self._prev: dict[str, JSON] = {}
        self._log_path = log_path
        self._out = out
        self._save_bodies = save_bodies
        self.records: list[CallRecord] = []

    def next_n(self) -> int:
        with self._lock:
            return len(self.records) + 1

    def compare(self, path: str, body: JSON) -> tuple[float | None, bool | None, Any]:
        with self._lock:
            prev = self._prev.get(path)
            self._prev[path] = body
        if prev is None:
            return None, None, None
        d = divergence(prev, body)
        return d.reuse, d.identical_prefix, d

    def add(self, rec: CallRecord, body: JSON | None) -> None:
        with self._lock:
            self.records.append(rec)
            if self._log_path is not None:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            if self._save_bodies is not None and body is not None:
                self._save_bodies.mkdir(parents=True, exist_ok=True)
                (self._save_bodies / f"{rec.n:04d}.json").write_text(
                    json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
            self._out.write(format_record(rec) + "\n")
            self._out.flush()


def format_record(rec: CallRecord) -> str:
    u = rec.usage
    parts = [f"#{rec.n:<3} {rec.path} {rec.status}"]
    if u.input_tokens is not None:
        cached = f" (cache {u.cached_tokens})" if u.cached_tokens is not None else ""
        parts.append(f"in {u.input_tokens}{cached}")
    if u.output_tokens is not None:
        parts.append(f"out {u.output_tokens}")
    parts.append(f"ttfb {rec.ttfb_s:.2f}s total {rec.total_s:.2f}s")
    if rec.reuse is not None:
        parts.append(f"prefix reuse {rec.reuse:.0%}")
    line = "  ".join(parts)
    if rec.prefix_ok is False and rec.diverged_at:
        line += (f"\n      prefix broke in {rec.diverged_at}: ...{_show(rec.diverged_before)}"
                 f"[{_show(rec.diverged_was)} -> {_show(rec.diverged_now)}]")
    if rec.error:
        line += f"\n      error: {rec.error}"
    return line


def _show(text: str | None) -> str:
    return (text or "").replace("\n", "\\n")


class _Handler(BaseHTTPRequestHandler):
    upstream: urllib.parse.SplitResult
    recorder: Recorder
    upstream_timeout: float
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._forward(None)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        self._forward(self.rfile.read(length) if length else b"")

    def _forward(self, raw: bytes | None) -> None:
        up = self.upstream
        conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(up.hostname or "127.0.0.1", up.port, timeout=self.upstream_timeout)
        path = up.path.rstrip("/") + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        if raw is not None:
            headers["Content-Length"] = str(len(raw))
        body: JSON | None = None
        if raw:
            try:
                parsed = json.loads(raw)
                body = parsed if isinstance(parsed, dict) else None
            except ValueError:
                body = None
        is_model_call = raw is not None and self.path.split("?")[0].endswith(MODEL_PATHS)
        start = time.monotonic()
        try:
            conn.request(self.command, path, body=raw, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:
            self.send_error(502, f"localready proxy: upstream unreachable: {exc}")
            return
        ttfb = 0.0
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        captured = bytearray()
        first = True
        client_gone = False
        try:
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                if first:
                    ttfb, first = time.monotonic() - start, False
                if is_model_call:
                    captured.extend(chunk)
                _write_chunk(self.wfile, chunk)
        except (BrokenPipeError, ConnectionResetError):
            client_gone = True
        finally:
            conn.close()
        total = time.monotonic() - start
        if is_model_call:
            # Record before the final chunk, so a client that has its answer can rely on the log.
            self._record(body, resp.status, resp.getheader("content-type") or "", bytes(captured),
                         ttfb or total, total, len(raw or b""))
        if not client_gone:
            try:
                _write_chunk(self.wfile, b"")
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _record(self, body: JSON | None, status: int, ctype: str, data: bytes,
                ttfb: float, total: float, nbytes: int) -> None:
        rec_path = self.path.split("?")[0]
        stream = "event-stream" in ctype or (body or {}).get("stream") is True
        text = data.decode("utf-8", errors="replace")
        usage = usage_from_sse(text) if "event-stream" in ctype else usage_from_body(text)
        rec = CallRecord(
            n=self.recorder.next_n(), path=rec_path,
            model=(body or {}).get("model") if isinstance((body or {}).get("model"), str) else None,
            status=status, stream=stream, ttfb_s=round(ttfb, 3), total_s=round(total, 3),
            request_bytes=nbytes, usage=usage,
        )
        if status >= 400:
            rec.error = _error_message(text)
        if body is not None and status < 400:
            reuse, ok, d = self.recorder.compare(rec_path, body)
            rec.reuse = None if reuse is None else round(reuse, 4)
            rec.prefix_ok = ok
            if ok is False and d is not None:
                rec.diverged_at = d.segment
                rec.diverged_before = d.before[-30:]
                rec.diverged_was = d.after_a[:30]
                rec.diverged_now = d.after_b[:30]
        self.recorder.add(rec, body)


def _write_chunk(w: Any, chunk: bytes) -> None:
    w.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
    w.flush()


def _error_message(text: str) -> str:
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip()[:300]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return str(err["message"])[:300]
    if isinstance(err, str):
        return err[:300]
    return text.strip()[:300]


class FastHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse-DNS ``getfqdn`` call that stalls on some macOS hosts."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), int(port)


def serve(listen: str, upstream: str, recorder: Recorder, timeout: float = 900.0,
          ready: threading.Event | None = None) -> ThreadingHTTPServer:
    """Build the proxy server (call ``serve_forever`` on it). ``listen`` is ``host:port``."""
    host, _, port = listen.rpartition(":")
    handler = type("Handler", (_Handler,), {
        "upstream": urllib.parse.urlsplit(upstream.rstrip("/")),
        "recorder": recorder,
        "upstream_timeout": timeout,
    })
    server = FastHTTPServer((host or "127.0.0.1", int(port)), handler)
    if ready is not None:
        ready.set()
    return server


def summarize(records: list[CallRecord], out: TextIO | None = None) -> None:
    out = out or sys.stdout
    """Print the end-of-session verdict on prefix caching and errors."""
    calls = [r for r in records if r.status < 400]
    compared = [r for r in calls if r.prefix_ok is not None]
    broken = [r for r in compared if r.prefix_ok is False]
    errors = [r for r in records if r.status >= 400]
    out.write(f"\n{len(records)} model calls, {len(errors)} errors, "
              f"{len(broken)}/{len(compared)} follow-up calls broke the prompt prefix\n")
    sites: dict[str, int] = {}
    for r in broken:
        sites[r.diverged_at or "?"] = sites.get(r.diverged_at or "?", 0) + 1
    for site, n in sorted(sites.items(), key=lambda kv: -kv[1]):
        out.write(f"  {n}x prefix changed in {site}\n")
    for r in errors[:5]:
        out.write(f"  error #{r.n}: {r.error}\n")
