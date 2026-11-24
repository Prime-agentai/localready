"""Human-readable rendering of a check report."""

from __future__ import annotations

import os
import sys
from typing import TextIO

from .check import Report
from .model import Finding, Status

MARK = {Status.PASS: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL", Status.SKIP: "skip", Status.INFO: "info"}
COLOR = {Status.PASS: "32", Status.WARN: "33", Status.FAIL: "31", Status.SKIP: "90", Status.INFO: "36"}
VERDICT = {Status.PASS: "READY", Status.WARN: "READY, WITH CAVEATS", Status.FAIL: "NOT READY"}
API_TITLE = {"anthropic": "Anthropic Messages wire", "chat": "OpenAI Chat wire", "responses": "OpenAI Responses wire"}


def _use_color(out: TextIO) -> bool:
    return hasattr(out, "isatty") and out.isatty() and os.environ.get("NO_COLOR") is None


def _tag(status: Status, color: bool) -> str:
    text = f"{MARK[status]:<4}"
    return f"\033[{COLOR[status]}m{text}\033[0m" if color else text


def _finding(f: Finding, color: bool, out: TextIO) -> None:
    out.write(f"  {_tag(f.status, color)}  {f.title}\n")
    if f.detail:
        out.write(f"        {f.detail}\n")
    if f.fix and f.status in (Status.FAIL, Status.WARN):
        out.write(f"        fix: {f.fix}\n")


def render(rep: Report, out: TextIO | None = None) -> None:
    out = out or sys.stdout
    color = _use_color(out)
    e = rep.engine
    ver = f" {e.version}" if e.version else ""
    win = f", window {e.window:,} ({e.window_source})" if e.window else ""
    out.write(f"\nlocalready  {rep.base_url}  model {rep.model}\n")
    out.write(f"engine: {e.name}{ver}{win}\n")
    for note in e.notes:
        out.write(f"  note: {note}\n")
    for api, findings in rep.wire.items():
        out.write(f"\n{API_TITLE.get(api, api)}\n")
        for f in findings:
            _finding(f, color, out)
    if rep.load:
        via = f" (measured through {API_TITLE.get(rep.probe_api or '', rep.probe_api)})" if rep.probe_api else ""
        out.write(f"\nContext and speed{via}\n")
        for f in rep.load:
            _finding(f, color, out)
    out.write("\nVerdict\n")
    for h in rep.harnesses:
        v = rep.verdict(h)
        live = "" if h.measured_live else "  (wire-level: harness not run live by localready)"
        out.write(f"  {_tag(v, color)}  {h.name}: {VERDICT[v]}{live}\n")
    out.write("\nRun `localready env` with the same arguments for the exact client setup, "
              "or `localready proxy` to watch a real session.\n")
