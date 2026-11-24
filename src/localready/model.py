"""Shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    INFO = "info"


@dataclass
class Finding:
    """One probe outcome. ``fix`` is a concrete command or setting, never prose advice."""

    id: str
    title: str
    status: Status
    detail: str = ""
    fix: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "status": self.status.value,
                "detail": self.detail, "fix": self.fix, "data": self.data}


@dataclass
class EngineInfo:
    name: str  # llama.cpp | ollama | vllm | lmstudio | unknown
    version: str | None = None
    models: list[str] = field(default_factory=list)
    window: int | None = None  # the window the server will actually use for this model, if it says
    window_source: str | None = None
    train_window: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Speed:
    prompt_tokens: int
    prefill_tps: float | None
    decode_tps: float | None
    ttft_s: float | None
    cached_ttft_s: float | None = None
    cached_tokens: int | None = None
    source: str = "client timing"
