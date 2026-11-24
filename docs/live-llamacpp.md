# Live run: llama.cpp + Claude Code (2026-11-24)

Everything here was run on the build machine on 2026-11-24. Every number is copied from tool output.

## Setup

| | |
|---|---|
| Machine | 6 vCPU, 11 GiB RAM, no GPU, shared with other services (load average ~6-8 during the runs) |
| Server | llama.cpp release `b12680` (`llama-b12680-bin-ubuntu-x64.tar.gz`, sha256 `91968c0f...e96085`, verified) |
| Model | `Qwen/Qwen3-0.6B-GGUF` `Qwen3-0.6B-Q8_0.gguf` (sha256 `9465e63a...6d043bb031`, verified) |
| Harness | Claude Code 2.1.330 |

## 1. `localready check` against `llama-server -c 16384 -np 1 --reasoning off`

Exit code 2. Raw output:

```text
localready  http://127.0.0.1:18080  model models/qwen3-0.6b-q8.gguf
engine: llama.cpp b12680-12d47ba2f, window 16,384 (/v1/models meta.n_ctx)

Anthropic Messages wire
  PASS  Anthropic Messages endpoint (/v1/messages)
        streams an Anthropic message
  PASS  Claude Code request shape (2.1.330 fields)
        accepted all 8 fields Claude Code sends
  PASS  Claude Code's max_tokens 32000 fits the window
        max_tokens 32,768 accepted: the server stops at its window by itself
  PASS  Streaming tool call (SSE grammar + input_json_delta)
        13 events, tool_use lookup_ticket({"ticket_id": 42})
  PASS  Tool result reaches the model (tool_use -> tool_result)
        model quoted 7Q4-XK
  PASS  Token counting (/v1/messages/count_tokens)
        input_tokens=13

Context and speed (measured through Anthropic Messages wire)
  PASS  Over-long prompt is refused, not silently cut
        ~20736 tokens refused in 0.55s: request (21272 tokens) exceeds the available context size (16384 tokens), try increasing it
  PASS  Claude Code's first request fits (11,876 tokens)
        processed 12131 tokens; model recalled the code from the top
  PASS  Prompt prefix is reused between turns
        first-token time 329.56s cold -> 12.52s with the same prefix, server reports 12,105 cached tokens
  FAIL  Prefill speed
        36 tokens/s (client timing): Claude Code's first request takes ~330s before the first token
        fix: use a smaller model or a GPU/Metal build; prefill, not generation, dominates agent latency
  WARN  Generation speed
        1.4 tokens/s (client timing)
  WARN  Context window for agent work
        16,384 tokens (/v1/models meta.n_ctx), below the 65,536 recommended for coding agents; roughly 2 tool turns of headroom after the first request. Tell the harness the real window so it compacts before the server refuses.

Verdict
  FAIL  Claude Code: NOT READY

Run `localready env` with the same arguments for the exact client setup, or `localready proxy` to watch a real session.

# Claude Code setup for this server
export ANTHROPIC_BASE_URL=http://127.0.0.1:18080
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=models/qwen3-0.6b-q8.gguf
export ANTHROPIC_DEFAULT_HAIKU_MODEL=models/qwen3-0.6b-q8.gguf   # background tasks must hit the local model too
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=16384   # Claude Code assumes 1M for unknown ids and would not compact before the server refuses
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096   # Claude Code asks for 32000 per request
export CLAUDE_CODE_ATTRIBUTION_HEADER=0   # LM Studio's Claude Code guide; drops a version header block from the system prompt
exit=2
```

The prefill here is 36 tokens/s. A first run at `-c 40960` with reasoning on got slower and slower.
llama-server held 54% of RAM on this shared box, and decode fell to 0.45 tokens/s. That run was
stopped and replaced by the 16k run above. It also exposed a real bug: acceptance probes that sent
`max_tokens: 32000` made the server generate for many minutes. Those probes now stream and
disconnect after the first event, and llama.cpp logs `cancel task`.

## 2. What Claude Code sends, captured with `localready proxy`

`claude -p "..."` in an empty directory, `ANTHROPIC_BASE_URL` pointed at the proxy:

- With an 8k server (`-c 8192`), Claude Code printed `Prompt is too long · the provider said: request
  (11620 tokens) exceeds the available context size (8192 tokens)`.
- With the server at 40k, the first request was **11,876 prompt tokens**. Prefill took 311.8 s
  (38 tokens/s, llama.cpp `print_timing`).
- Request body: `model, messages, system (3 text blocks, 2 with cache_control), tools (12, 31,544 chars),
  metadata, max_tokens 32000, thinking {type: adaptive, display: omitted}, context_management
  {edits: [clear_thinking_20251015]}, output_config {effort: high}, stream: true`. The environment
  block arrives as a `role: "system"` entry inside `messages`.
- Without `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, Claude Code warned: `Claude Code doesn't recognize the
  model "qwen3", so it assumes a context window of 1m tokens.` With it set, the warning is gone.
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096` changed `max_tokens` to 4096. `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
  removed `system[0]` (`x-anthropic-billing-header: cc_version=2.1.330.a50; cc_entrypoint=sdk-cli;`).

### Prefix stability between turns

To capture several turns without waiting minutes per turn, Claude Code was driven by the repo's
fake engine (`tests/fakeserver.py`, which answers instantly and calls the `Read` tool once):

```text
#1   /v1/messages 200  in 12413 (cache 0)  out 12  ttfb 0.00s total 0.01s
#2   /v1/messages 200  in 12509 (cache 12413)  out 11  ttfb 0.01s total 0.01s  prefix reuse 99%
```

The first version of the prefix analyser reported 78% reuse and a "break" in `messages[1:system]`.
Claude Code sends the same environment text as a list of blocks on turn 1 and as a plain string on
turn 2. A chat template renders both identically, so this was an artefact of the analyser, not a
cache break. The analyser now renders content the way a template does. The system header was
byte-identical across both requests. **We found no prefix break in Claude Code 2.1.330.**

## 3. A real Claude Code session through `localready proxy` (16k server)

Setup printed by `localready env` (`CLAUDE_CODE_MAX_CONTEXT_TOKENS=16384`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096`,
`CLAUDE_CODE_ATTRIBUTION_HEADER=0`). Task: `Use the Read tool to read calc.py, then tell me what add(2, 3) returns.`

```text
#1   /v1/messages 200  in 11867 (cache 1)  out 21  ttfb 33.40s total 330.68s
#2   /v1/messages 200  in 13214 (cache 11863)  out 291  ttfb 30.13s total 264.40s  prefix reuse 88%
#3   /v1/messages 200  in 12388 (cache 9356)  out 70  ttfb 30.20s total 197.54s  prefix reuse 74%
      prefix broke in messages[0:user]: ...me":"Write"}<system-reminder>\n[Attribution for git commits an -> Called the Read tool with the ]

3 model calls, 0 errors, 1/2 follow-up calls broke the prompt prefix
  1x prefix changed in messages[0:user]
```

- Call 2 extends call 1 (no break). The server reused 11,863 of 13,214 tokens, matching the
  proxy's 88%. Most of call 2's 264 s went into generating 291 tokens at about 1.4 tokens/s.
- In call 3 the conversation itself was rewritten: `messages[0]` now starts with `Called the Read tool
  with the ...`. The system prompt and tools stayed cached (9,356 tokens). This is consistent with
  Claude Code compacting the history at the 16k window it was told about. It is an inference: the
  request body was not saved for this run.
- The 0.6B model then answered with a hallucinated `<agent>` block. That is the model, which is
  why `localready` reports model problems as WARN and keeps them apart from wire FAILs.

## 4. `localready check` against `llama-server -c 8192` (exit 2)

```text
Context and speed (measured through Anthropic Messages wire)
  PASS  Over-long prompt is refused, not silently cut
        ~10496 tokens refused in 0.43s: request (10859 tokens) exceeds the available context size (8192 tokens), try increasing it
  FAIL  Claude Code's first request fits (11,876 tokens)
        declared window 8,192 < 11,876 tokens; the harness cannot even start (llama.cpp: 'exceeds the available context size'; Claude Code: 'Prompt is too long')
        fix: restart llama-server with  -c 65536  (with --parallel N each slot gets its share; set --kv-unified-per-slot 65536 or use -np 1)
  FAIL  Context window for agent work
        8,192 tokens (/v1/models meta.n_ctx) < the first request (11,876)

Verdict
  FAIL  Claude Code: NOT READY
```

`localready env` puts a `# NOT READY` line first in this case, so nobody exports a setup that cannot work.
