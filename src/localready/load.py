"""Context-window and speed probes, measured with the server's own token accounting.

Four requests, cheapest first:

1. calibrate: two small prompts give tokens-per-line for *this* model's tokenizer, so later
   prompts can be sized in real tokens rather than guessed from characters.
2. overflow: if the server declared a window W, send ~1.25 W. A healthy server refuses before
   doing any work (llama.cpp answers a 400 in milliseconds). A server that answers 200 is
   silently dropping part of the prompt, which in an agent means dropping the system prompt,
   the tool definitions or the task.
3. fit: send the harness's real first-request size with a secret code at the very top and the
   question at the bottom. Checks that the server processed every token and the model can still
   see the top of the prompt. Its time to first token gives the prefill speed.
4. cache: resend the same prompt with a different final question. An engine that reuses the
   prompt prefix (every agent turn resends the whole conversation) answers far faster.
"""

from __future__ import annotations

import random
import re
import string
import time
from dataclasses import dataclass
from typing import Any

from .client import Client, StreamReply
from .engines import window_fix
from .model import EngineInfo, Finding, Speed, Status
from .usage import Usage, usage_from_sse

JSON = dict[str, Any]
WORDS = ("alpha", "river", "copper", "lantern", "orbit", "maple", "signal", "harbor", "pixel", "quartz",
         "meadow", "cobalt", "ember", "falcon", "glacier", "ivory", "juniper", "kernel", "lumen", "nectar")
CONTEXT_ERROR = re.compile(r"context|too long|too many tokens|exceed|maximum|n_ctx|num_ctx|max_model_len", re.I)
N_CTX = re.compile(r"(?:context(?: size| length| window)?|n_ctx|max_model_len)\D{0,40}?\(?(\d{3,7})\s*(?:tokens)?\)?", re.I)


@dataclass
class Calibration:
    tokens_per_line: float
    overhead: float
    measured: bool


def _filler(lines: int, seed: int) -> str:
    rnd = random.Random(seed)
    return "\n".join(f"line {i}: " + " ".join(rnd.choice(WORDS) for _ in range(9)) + f" {rnd.randint(100, 999)}."
                     for i in range(lines))


def _salt() -> str:
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def _secret() -> str:
    return f"{random.choice(string.ascii_uppercase)}{random.randint(10, 99)}-{_salt()[:4]}"


def build_prompt(lines: int, salt: str, secret: str, question: str) -> str:
    return (f"Run {salt}. The secret code is {secret}. Remember it.\n\n" + _filler(lines, hash(salt) & 0xFFFF)
            + f"\n\n{question}")


QUESTION = "What is the secret code given at the very top? Reply with the code only."
QUESTION_2 = "Ignore the code. How many lines of filler were there, roughly? One number."


class Prober:
    def __init__(self, client: Client, model: str, api: str) -> None:
        self.c, self.model, self.api = client, model, api

    @property
    def path(self) -> str:
        return {"anthropic": "/v1/messages", "chat": "/v1/chat/completions", "responses": "/v1/responses"}[self.api]

    def body(self, prompt: str, max_tokens: int) -> JSON:
        if self.api == "responses":
            return {"model": self.model, "max_output_tokens": max_tokens, "input": prompt}
        return {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}

    def send(self, prompt: str, max_tokens: int, timeout: float | None = None) -> StreamReply:
        return self.c.stream(self.path, self.body(prompt, max_tokens), timeout=timeout)


def usage_of(s: StreamReply) -> Usage:
    return usage_from_sse(s.raw)


def text_of(s: StreamReply) -> str:
    """Everything the model produced, reasoning included.

    The recall check asks whether the model *saw* the top of the prompt; a reasoning model that
    quotes the code while thinking has seen it, even if its answer is cut off by max_tokens.
    """
    parts: list[str] = []
    for _, d in s.events:
        if not isinstance(d, dict):
            continue
        delta = d.get("delta")
        if d.get("type") == "content_block_delta" and isinstance(delta, dict):
            parts.append(str(delta.get("text") or delta.get("thinking") or ""))
        elif str(d.get("type", "")).startswith("response.") and str(d.get("type")).endswith(".delta") \
                and isinstance(delta, str):
            parts.append(delta)
        for ch in d.get("choices") or []:
            dl = ch.get("delta") if isinstance(ch, dict) else None
            if isinstance(dl, dict):
                for key in ("content", "reasoning_content", "reasoning"):
                    if isinstance(dl.get(key), str):
                        parts.append(dl[key])
    return "".join(parts)


def calibrate(p: Prober) -> Calibration:
    small, big = 10, 60
    a = p.send(build_prompt(small, _salt(), _secret(), "Reply OK."), 1, timeout=300)
    b = p.send(build_prompt(big, _salt(), _secret(), "Reply OK."), 1, timeout=300)
    ua, ub = usage_of(a), usage_of(b)
    if a.ok and b.ok and ua.input_tokens and ub.input_tokens and ub.input_tokens > ua.input_tokens:
        per = (ub.input_tokens - ua.input_tokens) / (big - small)
        return Calibration(per, ua.input_tokens - per * small, True)
    return Calibration(26.0, 40.0, False)  # ~4 chars/token on these lines; flagged as estimated


def lines_for(cal: Calibration, tokens: int) -> int:
    return max(1, int((tokens - cal.overhead) / cal.tokens_per_line))


def parse_window(message: str) -> int | None:
    nums = [int(n) for n in N_CTX.findall(message or "")]
    return max(nums) if nums else None


def overflow_probe(p: Prober, cal: Calibration, engine: EngineInfo, max_window: int = 65536) -> Finding:
    title = "Over-long prompt is refused, not silently cut"
    w = engine.window
    if not w:
        return Finding("context.overflow", title, Status.SKIP,
                       "server did not declare a window, so there is no cheap overflow test; the fit test below "
                       "still catches truncation up to the harness's size")
    if w > max_window:
        return Finding("context.overflow", title, Status.SKIP,
                       f"declared window {w:,} > {max_window:,}: an over-long prompt would cost a full prefill if the "
                       "server truncates silently (raise with --overflow-max)")
    target = int(w * 1.25) + 256
    s = p.send(build_prompt(lines_for(cal, target), _salt(), _secret(), QUESTION), 16, timeout=900)
    if not s.ok and s.status and s.status >= 400:
        loud = bool(CONTEXT_ERROR.search(s.error or ""))
        return Finding("context.overflow", title, Status.PASS if loud else Status.WARN,
                       f"~{target} tokens refused in {s.total_s:.2f}s: {s.error}"
                       + ("" if loud else " (refused, but the error does not mention the context limit)"),
                       data={"sent_tokens": target, "error": s.error})
    if not s.ok:
        return Finding("context.overflow", title, Status.WARN, f"request failed without an HTTP error: {s.error}")
    got = usage_of(s).input_tokens
    return Finding("context.overflow", title, Status.FAIL,
                   f"sent ~{target} tokens to a {w}-token window and got HTTP 200"
                   + (f" after processing {got} tokens" if got else "")
                   + ": the server silently drops part of every over-long prompt. An agent then loses its system "
                     "prompt, tools or task with no error.",
                   fix=window_fix(engine.name, max(w, 65536), p.model),
                   data={"sent_tokens": target, "processed_tokens": got})


def fit_probe(p: Prober, cal: Calibration, engine: EngineInfo, need: int, label: str) -> tuple[Finding, Speed | None, str]:
    title = f"{label}'s first request fits ({need:,} tokens)" if label else f"A {need:,}-token prompt fits"
    salt, secret = _salt(), _secret()
    prompt = build_prompt(lines_for(cal, need), salt, secret, QUESTION)
    if engine.window and engine.window < need:
        return (Finding("context.fit", title, Status.FAIL,
                        f"declared window {engine.window:,} < {need:,} tokens; the harness cannot even start "
                        "(llama.cpp: 'exceeds the available context size'; Claude Code: 'Prompt is too long')",
                        fix=window_fix(engine.name, 65536, p.model)), None, prompt)
    s = p.send(prompt, 128, timeout=3600)
    if not s.ok:
        win = parse_window(s.error or "")
        if s.status and s.status >= 400 and CONTEXT_ERROR.search(s.error or ""):
            return (Finding("context.fit", title, Status.FAIL, f"refused: {s.error}",
                            fix=window_fix(engine.name, 65536, p.model), data={"window": win}), None, prompt)
        return Finding("context.fit", title, Status.FAIL, f"HTTP {s.status}: {s.error}"), None, prompt
    u = usage_of(s)
    answer = text_of(s)
    speed = _speed(s, u, need)
    expected = need
    if u.input_tokens is not None and cal.measured and u.input_tokens < 0.85 * expected:
        return (Finding("context.fit", title, Status.FAIL,
                        f"sent ~{expected:,} tokens, server processed {u.input_tokens:,} and answered 200: "
                        "SILENT TRUNCATION. The top of the prompt (where agents put their system prompt and "
                        "tools) never reached the model.",
                        fix=window_fix(engine.name, 65536, p.model),
                        data={"processed_tokens": u.input_tokens, "effective_window": u.input_tokens}), speed, prompt)
    seen = secret.lower() in answer.lower()
    data: JSON = {"processed_tokens": u.input_tokens, "secret_recalled": seen}
    if seen:
        return Finding("context.fit", title, Status.PASS,
                       f"processed {u.input_tokens or '?'} tokens; model recalled the code from the top", data=data), speed, prompt
    if u.input_tokens is None:
        return (Finding("context.fit", title, Status.WARN,
                        f"server reported no token usage and the model did not recall the code from the top "
                        f"(answered {answer[:60]!r}): either the prompt was cut or the model missed it", data=data),
                speed, prompt)
    return (Finding("context.fit", title, Status.WARN,
                    f"all {u.input_tokens:,} tokens processed, but the model did not recall the code from the top "
                    f"(answered {answer[:60]!r}). The server is fine; this model loses track at this length.",
                    data=data), speed, prompt)


def _speed(s: StreamReply, u: Usage, prompt_tokens: int) -> Speed:
    if u.prompt_ms and u.input_tokens:
        prefill = (u.input_tokens - (u.cached_tokens or 0)) / (u.prompt_ms / 1000) if u.prompt_ms > 0 else None
        decode = (u.output_tokens or 0) / (u.decode_ms / 1000) if u.decode_ms else None
        return Speed(u.input_tokens, prefill, decode, s.first_token_s, source="server timings")
    ttft = s.first_token_s or s.total_s
    prefill = prompt_tokens / ttft if ttft else None
    decode = None
    if u.output_tokens and s.first_token_s and s.total_s > s.first_token_s:
        decode = max(0, u.output_tokens - 1) / (s.total_s - s.first_token_s) or None
    return Speed(u.input_tokens or prompt_tokens, prefill, decode, s.first_token_s, source="client timing")


def cache_probe(p: Prober, prompt: str, first: Speed) -> tuple[Finding, Speed]:
    title = "Prompt prefix is reused between turns"
    again = prompt.rsplit("\n\n", 1)[0] + "\n\n" + QUESTION_2
    t0 = time.monotonic()
    s = p.send(again, 16, timeout=3600)
    if not s.ok:
        return Finding("speed.cache", title, Status.WARN, f"second request failed: {s.error}"), first
    u = usage_of(s)
    ttft2 = s.first_token_s or (time.monotonic() - t0)
    first.cached_ttft_s, first.cached_tokens = ttft2, u.cached_tokens
    ratio = (first.ttft_s or 0) / ttft2 if ttft2 else 0
    cached_note = f", server reports {u.cached_tokens:,} cached tokens" if u.cached_tokens else ""
    detail = f"first-token time {first.ttft_s or 0:.2f}s cold -> {ttft2:.2f}s with the same prefix{cached_note}"
    if (u.cached_tokens and first.prompt_tokens and u.cached_tokens >= 0.8 * first.prompt_tokens) or ratio >= 3:
        return Finding("speed.cache", title, Status.PASS, detail), first
    if (first.ttft_s or 0) < 1.5:
        return Finding("speed.cache", title, Status.INFO, detail + " (too fast to tell)"), first
    return (Finding("speed.cache", title, Status.WARN,
                    detail + ": every agent turn will re-process the whole conversation",
                    fix="llama.cpp: keep cache_prompt on (default) and use one slot per agent (-np 1); "
                        "Ollama/LM Studio: keep the model loaded (OLLAMA_KEEP_ALIVE); then run `localready proxy` "
                        "to see whether your harness changes the prompt prefix between turns"), first)


def speed_findings(sp: Speed, need: int, label: str) -> list[Finding]:
    out: list[Finding] = []
    if sp.prefill_tps:
        cold = need / sp.prefill_tps
        status = Status.PASS if cold <= 30 else Status.WARN if cold <= 120 else Status.FAIL
        out.append(Finding("speed.prefill", "Prefill speed", status,
                           f"{sp.prefill_tps:,.0f} tokens/s ({sp.source}): {label}'s first request takes ~{cold:,.0f}s "
                           f"before the first token",
                           fix=None if status == Status.PASS else
                           "use a smaller model or a GPU/Metal build; prefill, not generation, dominates agent latency",
                           data={"prefill_tps": round(sp.prefill_tps, 1), "first_request_seconds": round(cold, 1)}))
    if sp.decode_tps:
        status = Status.PASS if sp.decode_tps >= 15 else Status.WARN
        out.append(Finding("speed.decode", "Generation speed", status,
                           f"{sp.decode_tps:,.1f} tokens/s ({sp.source})",
                           data={"decode_tps": round(sp.decode_tps, 1)}))
    return out
