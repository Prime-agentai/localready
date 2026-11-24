"""localready command line."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from types import FrameType

from . import __version__
from .check import Report, env_lines, run
from .client import Client
from .harness import resolve
from .proxy import Recorder, serve, summarize
from .report import render

DEFAULT_PORTS = (("ollama", 11434), ("llama.cpp", 8080), ("LM Studio", 1234), ("vLLM", 8000), ("MLX / other", 8081))


def scan() -> list[tuple[str, str, list[str]]]:
    """Find model servers on the usual local ports (1 s per port, GET /v1/models only)."""
    found: list[tuple[str, str, list[str]]] = []
    for label, port in DEFAULT_PORTS:
        url = f"http://127.0.0.1:{port}"
        r = Client(url).get("/v1/models", timeout=1.0)
        if r.ok and isinstance(r.body, dict):
            data = r.body.get("data")
            ids = [str(m.get("id")) for m in data if isinstance(m, dict)] if isinstance(data, list) else []
            found.append((label, url, ids))
    return found


def _target(url: str | None) -> str:
    if url:
        return url if "://" in url else f"http://{url}"
    found = scan()
    if not found:
        ports = ", ".join(str(p) for _, p in DEFAULT_PORTS)
        raise SystemExit(f"localready: no model server answered GET /v1/models on 127.0.0.1 ports {ports}; "
                         "pass the URL, e.g. `localready check http://127.0.0.1:8080`")
    label, base, ids = found[0]
    print(f"localready: found {label} at {base} ({len(ids)} model(s))", file=sys.stderr)
    return base


class ProgressPrinter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, msg: str) -> None:
        if self.enabled:
            print(f"  .. {msg}", file=sys.stderr, flush=True)


def cmd_check(a: argparse.Namespace) -> int:
    harnesses = resolve(a.harness)
    rep = run(_target(a.url), a.model, harnesses, quick=a.quick, need=a.need, api_key=a.api_key,
              overflow_max=a.overflow_max,
              progress=ProgressPrinter(not a.json))
    if a.json:
        print(json.dumps(rep.to_json(), indent=1))
    else:
        render(rep)
        _print_env(rep)
    return rep.exit_code


def _print_env(rep: Report) -> None:
    for h in rep.harnesses:
        print(f"\n# {h.name} setup for this server")
        for line in env_lines(rep, h):
            print(line)


def cmd_env(a: argparse.Namespace) -> int:
    harnesses = resolve(a.harness)
    rep = run(_target(a.url), a.model, harnesses, quick=True, api_key=a.api_key)
    for h in harnesses:
        for line in env_lines(rep, h):
            print(line)
    return 0


def cmd_scan(_a: argparse.Namespace) -> int:
    found = scan()
    if not found:
        print("no model servers found on " + ", ".join(f"{n}:{p}" for n, p in DEFAULT_PORTS))
        return 1
    for label, url, ids in found:
        print(f"{url}  ({label} port)  models: {', '.join(ids) or '-'}")
    return 0


def cmd_proxy(a: argparse.Namespace) -> int:
    log = Path(a.log) if a.log else None
    rec = Recorder(log, sys.stdout, Path(a.save_bodies) if a.save_bodies else None)
    server = serve(a.listen, _target(a.upstream), rec)
    host, port = str(server.server_address[0]), int(server.server_address[1])
    print(f"localready proxy: http://{host}:{port} -> {a.upstream or 'auto'}\n"
          f"  Claude Code:  ANTHROPIC_BASE_URL=http://{host}:{port} claude\n"
          f"  OpenAI-style: OPENAI_BASE_URL=http://{host}:{port}/v1\n"
          "  Ctrl-C prints the summary.", flush=True)

    def stop(_sig: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        summarize(rec.records)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localready",
                                description="Is your local model server ready to drive a coding agent?")
    p.add_argument("--version", action="version", version=f"localready {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("url", nargs="?", help="server base URL (default: scan the usual local ports)")
        sp.add_argument("--model", "-m", help="model id (default: the first one the server lists)")
        sp.add_argument("--harness", "-H", action="append", default=None,
                        help="claude-code, codex, openai-chat (opencode/aider/cline), or all; repeatable "
                             "(default: claude-code)")
        sp.add_argument("--api-key", default="localready", help="key to send, if your server requires one")

    c = sub.add_parser("check", help="probe wire conformance, context window and speed; print verdict and setup")
    common(c)
    c.add_argument("--quick", action="store_true", help="wire probes only: skip the context and speed probes")
    c.add_argument("--need", type=int, help="prompt size in tokens for the fit probe (default: the harness's "
                                            "measured first request, 11,876 for Claude Code)")
    c.add_argument("--overflow-max", type=int, default=65536,
                   help="only run the overflow probe when the declared window is at most this many tokens "
                        "(default 65536; a silently truncating server would otherwise cost a full prefill)")
    c.add_argument("--json", action="store_true", help="machine-readable report")
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("env", help="print the client setup (export lines) for this server; quick probes only")
    common(e)
    e.set_defaults(func=cmd_env)

    s = sub.add_parser("scan", help="list model servers on the usual local ports")
    s.set_defaults(func=cmd_scan)

    x = sub.add_parser("proxy", help="recording proxy: per-call tokens, cache reuse, prefix breaks, errors")
    x.add_argument("upstream", nargs="?", help="server base URL (default: scan the usual local ports)")
    x.add_argument("--listen", default="127.0.0.1:8787", help="host:port to listen on (default 127.0.0.1:8787)")
    x.add_argument("--log", help="append one JSON line per model call to this file")
    x.add_argument("--save-bodies", metavar="DIR", help="also save each request body as JSON (contains your prompts)")
    x.set_defaults(func=cmd_proxy)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "harness", "unset") is None:
        args.harness = ["claude-code"]
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"localready: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"localready: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
