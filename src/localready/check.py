"""Run every probe for the chosen harnesses and turn findings into a verdict and a setup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import load, wire
from .client import Client
from .engines import discover
from .harness import CLAUDE_CODE, CODEX, OPENAI_CHAT, Harness
from .model import EngineInfo, Finding, Speed, Status

ORDER = {Status.FAIL: 0, Status.WARN: 1, Status.PASS: 2, Status.INFO: 3, Status.SKIP: 4}


@dataclass
class Report:
    base_url: str
    model: str
    engine: EngineInfo
    harnesses: list[Harness]
    wire: dict[str, list[Finding]] = field(default_factory=dict)
    load: list[Finding] = field(default_factory=list)
    speed: Speed | None = None
    window: int | None = None
    window_source: str | None = None
    probe_api: str | None = None

    def findings_for(self, h: Harness) -> list[Finding]:
        return self.wire.get(h.api, []) + self.load

    def verdict(self, h: Harness) -> Status:
        statuses = [f.status for f in self.findings_for(h)]
        if Status.FAIL in statuses:
            return Status.FAIL
        if Status.WARN in statuses:
            return Status.WARN
        return Status.PASS

    @property
    def exit_code(self) -> int:
        return 2 if any(self.verdict(h) == Status.FAIL for h in self.harnesses) else 0

    def to_json(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url, "model": self.model,
            "engine": {"name": self.engine.name, "version": self.engine.version, "window": self.engine.window,
                       "window_source": self.engine.window_source, "train_window": self.engine.train_window,
                       "notes": self.engine.notes},
            "effective_window": self.window, "effective_window_source": self.window_source,
            "probe_api": self.probe_api,
            "wire": {api: [f.to_json() for f in fs] for api, fs in self.wire.items()},
            "load": [f.to_json() for f in self.load],
            "verdicts": {h.key: self.verdict(h).value for h in self.harnesses},
            "env": {h.key: env_lines(self, h) for h in self.harnesses},
        }


PROBES: dict[str, Callable[[Client, str, EngineInfo], list[Finding]]] = {
    "anthropic": wire.anthropic_probes,
    "chat": wire.chat_probes,
    "responses": wire.responses_probes,
}


def run(base_url: str, model: str | None, harnesses: list[Harness], *, quick: bool = False,
        need: int | None = None, api_key: str = "localready", overflow_max: int = 65536,
        progress: Callable[[str], None] = lambda _m: None) -> Report:
    client = Client(base_url, api_key=api_key)
    progress("discovering engine")
    engine = discover(client, model)
    if model is None:
        if not engine.models:
            raise RuntimeError(f"{base_url} lists no models (GET /v1/models); pass --model")
    chosen: str = model or engine.models[0]
    rep = Report(base_url=client.base, model=chosen, engine=engine, harnesses=harnesses)
    rep.window, rep.window_source = engine.window, engine.window_source
    for api in dict.fromkeys(h.api for h in harnesses):
        progress(f"wire probes: {api}")
        rep.wire[api] = PROBES[api](client, chosen, engine)
    if quick:
        return rep
    probe_api = _probe_api(rep)
    if probe_api is None:
        rep.load.append(Finding("context.skipped", "Context and speed", Status.SKIP,
                                "no working endpoint to measure with"))
        return rep
    rep.probe_api = probe_api
    p = load.Prober(client, chosen, probe_api)
    progress("calibrating tokenizer")
    cal = load.calibrate(p)
    if not cal.measured:
        rep.load.append(Finding("context.calibration", "Token accounting", Status.WARN,
                                "server reports no prompt token counts, so truncation is judged from recall only"))
    progress("overflow probe")
    over = load.overflow_probe(p, cal, engine, overflow_max)
    rep.load.append(over)
    if over.status == Status.FAIL and isinstance(over.data.get("processed_tokens"), int):
        rep.window, rep.window_source = over.data["processed_tokens"], "measured: tokens processed before truncation"
    elif over.status == Status.PASS and rep.window is None:
        win = load.parse_window(str(over.data.get("error", "")))
        if win:
            rep.window, rep.window_source = win, "measured: server's overflow error"
    sized = max(harnesses, key=lambda h: h.first_request_tokens or 0)
    target = need or sized.first_request_tokens or CLAUDE_CODE.first_request_tokens or 12000
    label = "" if need else (sized.name if sized.first_request_tokens else "Claude Code")
    progress(f"fit probe at {target:,} tokens (slow on CPU: this is a real agent-sized prefill)")
    fit, speed, prompt = load.fit_probe(p, cal, engine, target, label)
    rep.load.append(fit)
    if fit.status == Status.FAIL:
        win = fit.data.get("effective_window") or fit.data.get("window")
        if isinstance(win, int):
            rep.window, rep.window_source = win, "measured: fit probe"
    if speed is not None:
        progress("prefix-cache probe")
        cache, speed = load.cache_probe(p, prompt, speed)
        rep.load.append(cache)
        rep.speed = speed
        rep.load.extend(load.speed_findings(speed, target, label))
    rep.load.append(_window_finding(rep, harnesses))
    return rep


def _probe_api(rep: Report) -> str | None:
    """Measure context and speed through the first API (in harness order) whose endpoint works."""
    for api in rep.wire:
        fs = rep.wire[api]
        if fs and fs[0].status == Status.PASS:
            return api
    return None


def _window_finding(rep: Report, harnesses: list[Harness]) -> Finding:
    rec = max(h.recommended_window for h in harnesses)
    w = rep.window
    title = "Context window for agent work"
    if w is None:
        return Finding("context.window", title, Status.WARN,
                       "the server never said how large its window is; set CLAUDE_CODE_MAX_CONTEXT_TOKENS (or your "
                       "harness's equivalent) to the window you configured, or Claude Code assumes 1M and never "
                       "compacts in time")
    first = max(h.first_request_tokens or 0 for h in harnesses)
    src = f" ({rep.window_source})" if rep.window_source else ""
    if first and w < first:
        return Finding("context.window", title, Status.FAIL, f"{w:,} tokens{src} < the first request ({first:,})",
                       data={"window": w})
    if w < rec:
        turns = f"; roughly {max(0, w - first) // 2000} tool turns of headroom after the first request" if first else ""
        return Finding("context.window", title, Status.WARN,
                       f"{w:,} tokens{src}, below the {rec:,} recommended for coding agents{turns}. "
                       "Tell the harness the real window so it compacts before the server refuses.",
                       data={"window": w})
    return Finding("context.window", title, Status.PASS, f"{w:,} tokens{src}", data={"window": w})


def env_lines(rep: Report, h: Harness) -> list[str]:
    """The exact client-side setup for this harness against this server."""
    w = rep.window
    if h.key == CLAUDE_CODE.key:
        lines = [
            f"export ANTHROPIC_BASE_URL={rep.base_url}",
            "export ANTHROPIC_AUTH_TOKEN=local",
            f"export ANTHROPIC_MODEL={rep.model}",
            f"export ANTHROPIC_DEFAULT_HAIKU_MODEL={rep.model}   # background tasks must hit the local model too",
        ]
        if w:
            lines.append(f"export CLAUDE_CODE_MAX_CONTEXT_TOKENS={w}   # Claude Code assumes 1M for unknown ids "
                         "and would not compact before the server refuses")
        else:
            lines.append("export CLAUDE_CODE_MAX_CONTEXT_TOKENS=<your server's window>   # unknown: see report")
        out_cap = _output_cap(rep, w)
        if out_cap:
            lines.append(f"export CLAUDE_CODE_MAX_OUTPUT_TOKENS={out_cap}   # Claude Code asks for 32000 per request")
        lines.append("export CLAUDE_CODE_ATTRIBUTION_HEADER=0   # LM Studio's Claude Code guide; drops a version "
                     "header block from the system prompt")
        return lines
    if h.key == CODEX.key:
        return [
            "# ~/.codex/config.toml  (wire-level probes only: localready has not run Codex live)",
            "[model_providers.local]",
            'name = "local"',
            f'base_url = "{rep.base_url}/v1"',
            'wire_api = "responses"',
            "# then: codex -c model_provider=local -m " + rep.model,
        ]
    if h.key == OPENAI_CHAT.key:
        return [f"export OPENAI_BASE_URL={rep.base_url}/v1", "export OPENAI_API_KEY=local",
                f"# model: {rep.model}" + (f"  context window: {w}" if w else "")]
    return []


def _output_cap(rep: Report, window: int | None) -> int | None:
    found = {f.id: f for f in rep.wire.get("anthropic", [])}
    shape, budget = found.get("anthropic.claude_code_shape"), found.get("anthropic.output_budget")
    rejected = shape.data.get("rejected", []) if shape else []
    if budget is not None and budget.data.get("enforced"):
        return int(budget.data["cap"])
    if "max_tokens 32000" in rejected:
        return 8192 if not window else max(1024, min(8192, window // 4))
    if window and window < 65536:
        return max(1024, min(8192, window // 4))
    return None
