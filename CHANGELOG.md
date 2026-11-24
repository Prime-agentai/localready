# Changelog

## 0.1.0 (2026-11-24)

First release.

- `localready check`: wire conformance for the Anthropic Messages API (Claude Code 2.1.330's real
  request shape, the output budget, SSE grammar, tool-argument assembly, the tool-result round trip,
  token counting), OpenAI Chat Completions and OpenAI Responses. Also measures silent truncation
  and the effective window from the server's own token counts, plus fit and recall at the harness's
  measured first-request size (11,876 tokens for Claude Code), prefill and decode speed, and prefix
  caching. Prints a verdict per harness and the exact client setup.
- `localready env`: the export lines and config for Claude Code, Codex and OpenAI-compatible agents.
- `localready proxy`: a transparent recording proxy that logs per-call tokens and server cache hits,
  prefix reuse, the segment where a prefix broke, and errors. It never logs credentials.
- `localready scan`: finds servers on the usual local ports.
- Engines: llama.cpp (live-tested, b12680), Ollama, LM Studio and vLLM (discovery and fixes from their docs,
  tested against fake engines).
