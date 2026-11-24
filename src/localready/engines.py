"""Identify the engine behind a URL, what window it says it will use, and how to fix it.

Discovery only reads cheap GET endpoints; it never loads or unloads a model. What the engine
*says* is reported as declared; the context probe then measures what it actually *does*.
"""

from __future__ import annotations

from typing import Any

from .client import Client, as_dict
from .model import EngineInfo


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def discover(client: Client, model: str | None) -> EngineInfo:
    models_reply = client.get("/v1/models")
    entries: list[dict[str, Any]] = []
    if models_reply.ok and isinstance(models_reply.body, dict):
        data = models_reply.body.get("data")
        if isinstance(data, list):
            entries = [e for e in data if isinstance(e, dict)]
    ids = [str(e.get("id")) for e in entries if e.get("id")]
    entry = next((e for e in entries if e.get("id") == model), entries[0] if entries else {})

    ollama = client.get("/api/version")
    if ollama.ok and isinstance(ollama.body, dict) and "version" in ollama.body:
        return _ollama(client, str(ollama.body["version"]), ids, model)

    if entry.get("owned_by") == "llamacpp" or _is_llamacpp(client):
        info = EngineInfo("llama.cpp", models=ids)
        meta = as_dict(entry.get("meta"))
        props = client.get("/props")
        if props.ok and isinstance(props.body, dict):
            info.version = str(props.body.get("build_info") or "") or None
            caps = props.body.get("chat_template_caps")
            if isinstance(caps, dict) and caps.get("supports_tool_calls") is False:
                info.notes.append("chat template does not support tool calls (chat_template_caps)")
            gen = props.body.get("default_generation_settings")
            if isinstance(gen, dict) and _int(gen.get("n_ctx")):
                info.window, info.window_source = _int(gen.get("n_ctx")), "/props n_ctx"
        if _int(meta.get("n_ctx")):
            info.window, info.window_source = _int(meta.get("n_ctx")), "/v1/models meta.n_ctx"
        info.train_window = _int(meta.get("n_ctx_train"))
        return info

    if entry.get("owned_by") == "vllm" or _int(entry.get("max_model_len")):
        version = client.get("/version")
        info = EngineInfo("vllm", models=ids)
        if version.ok and isinstance(version.body, dict):
            info.version = str(version.body.get("version") or "") or None
        if _int(entry.get("max_model_len")):
            info.window, info.window_source = _int(entry.get("max_model_len")), "/v1/models max_model_len"
        return info

    lms = client.get("/api/v0/models")
    if lms.ok and isinstance(lms.body, dict) and isinstance(lms.body.get("data"), list):
        info = EngineInfo("lmstudio", models=ids)
        for m in lms.body["data"]:
            if not isinstance(m, dict) or (model and m.get("id") != model):
                continue
            if _int(m.get("loaded_context_length")):
                info.window, info.window_source = _int(m.get("loaded_context_length")), "/api/v0/models loaded_context_length"
            info.train_window = _int(m.get("max_context_length"))
            break
        return info

    return EngineInfo("unknown", models=ids)


def _is_llamacpp(client: Client) -> bool:
    props = client.get("/props")
    return props.ok and isinstance(props.body, dict) and "build_info" in props.body


def _ollama(client: Client, version: str, ids: list[str], model: str | None) -> EngineInfo:
    info = EngineInfo("ollama", version=version, models=ids)
    ps = client.get("/api/ps")
    if ps.ok and isinstance(ps.body, dict):
        for m in ps.body.get("models") or []:
            if isinstance(m, dict) and (model is None or m.get("name") == model or m.get("model") == model):
                if _int(m.get("context_length")):
                    info.window, info.window_source = _int(m.get("context_length")), "/api/ps context_length (loaded)"
                break
    if model:
        show = client.post("/api/show", {"model": model}, timeout=15)
        if show.ok and isinstance(show.body, dict):
            mi = show.body.get("model_info")
            if isinstance(mi, dict):
                for k, v in mi.items():
                    if k.endswith(".context_length") and _int(v):
                        info.train_window = _int(v)
            caps = show.body.get("capabilities")
            if isinstance(caps, list) and "tools" not in caps:
                info.notes.append(f"model capabilities {caps} do not include 'tools'")
    if info.window is None:
        info.notes.append("model not loaded yet, so Ollama has not said which window it will use "
                          "(its docs: 4k by default below 24 GiB VRAM unless OLLAMA_CONTEXT_LENGTH is set)")
    return info


def window_fix(engine: str, tokens: int, model: str | None) -> str:
    m = model or "<model>"
    return {
        "llama.cpp": f"restart llama-server with  -c {tokens}  (with --parallel N each slot gets its share; "
                     f"set --kv-unified-per-slot {tokens} or use -np 1)",
        "ollama": f"OLLAMA_CONTEXT_LENGTH={tokens} ollama serve   (then check the CONTEXT column of `ollama ps`)",
        "lmstudio": f"lms load {m} --context-length {tokens}",
        "vllm": f"vllm serve {m} --max-model-len {tokens}",
    }.get(engine, f"raise the server's context window to at least {tokens} tokens")


def tools_fix(engine: str, model: str | None) -> str:
    return {
        "llama.cpp": "run llama-server with --jinja (default in recent builds) and a model whose chat template "
                     "supports tools; check /props chat_template_caps.supports_tool_calls",
        "ollama": f"use a model tagged with the 'tools' capability (`ollama show {model or '<model>'}`)",
        "lmstudio": "load a model LM Studio marks as tool-use capable",
        "vllm": "vllm serve ... --enable-auto-tool-choice --tool-call-parser <parser for your model> "
                "(e.g. hermes for Qwen2.5, qwen3_xml for Qwen3-Coder, openai for gpt-oss)",
    }.get(engine, "enable the server's tool-call parser for this model")
