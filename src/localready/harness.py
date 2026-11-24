"""What each coding-agent harness needs from a model server.

Every number here is either measured (with how) or a documented recommendation (with where).
Nothing is a guess. Harness requirements change between releases; ``localready proxy``
measures your own harness, tools and project instead of trusting these defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Harness:
    key: str
    name: str
    api: str  # anthropic | chat | responses
    path: str
    first_request_tokens: int | None
    first_request_source: str
    recommended_window: int
    recommended_source: str
    max_output_tokens: int | None  # what the harness asks for per request, if known
    max_output_source: str
    measured_live: bool


CLAUDE_CODE = Harness(
    key="claude-code",
    name="Claude Code",
    api="anthropic",
    path="/v1/messages",
    first_request_tokens=11876,
    first_request_source=("measured: Claude Code 2.1.330, `claude -p` in an empty project, 12 built-in "
                          "tools, no MCP servers, Qwen3 tokenizer (llama.cpp b12680 usage)"),
    recommended_window=65536,
    recommended_source="Ollama's Claude Code guide: 'set a 64k+ context window' (docs.ollama.com/integrations/claude-code)",
    max_output_tokens=32000,
    max_output_source="measured: Claude Code 2.1.330 request body (max_tokens), override with CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    measured_live=True,
)

CODEX = Harness(
    key="codex",
    name="Codex CLI",
    api="responses",
    path="/v1/responses",
    first_request_tokens=None,
    first_request_source="not measured by localready yet: run `localready proxy` in front of your server to measure it",
    recommended_window=65536,
    recommended_source="Ollama's context-length guide: coding tools 'should be set to at least 64000 tokens'",
    max_output_tokens=None,
    max_output_source="not measured",
    measured_live=False,
)

OPENAI_CHAT = Harness(
    key="openai-chat",
    name="OpenAI-compatible agents (OpenCode, Aider, Cline, Continue, ...)",
    api="chat",
    path="/v1/chat/completions",
    first_request_tokens=None,
    first_request_source="not measured by localready yet: run `localready proxy` in front of your server to measure it",
    recommended_window=65536,
    recommended_source="Ollama's context-length guide: coding tools 'should be set to at least 64000 tokens'",
    max_output_tokens=None,
    max_output_source="not measured",
    measured_live=False,
)

HARNESSES = {h.key: h for h in (CLAUDE_CODE, CODEX, OPENAI_CHAT)}
ALIASES = {"claude": "claude-code", "cc": "claude-code", "opencode": "openai-chat", "chat": "openai-chat",
           "openai": "openai-chat", "aider": "openai-chat", "cline": "openai-chat", "responses": "codex"}


def resolve(names: list[str]) -> list[Harness]:
    out: list[Harness] = []
    for raw in names:
        for part in raw.split(","):
            key = part.strip().lower()
            if not key:
                continue
            if key == "all":
                return list(HARNESSES.values())
            key = ALIASES.get(key, key)
            if key not in HARNESSES:
                raise ValueError(f"unknown harness {part!r}; choose from {', '.join(HARNESSES)} or all")
            if HARNESSES[key] not in out:
                out.append(HARNESSES[key])
    return out
