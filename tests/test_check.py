from typing import Any

from fakeserver import Behaviour, FakeServer

from localready.check import Report, env_lines, run
from localready.client import Client
from localready.engines import discover
from localready.harness import CLAUDE_CODE, CODEX, OPENAI_CHAT, resolve
from localready.model import Finding, Status

ALL = [CLAUDE_CODE, CODEX, OPENAI_CHAT]


def by_id(rep: Report) -> dict[str, Finding]:
    out = {f.id: f for fs in rep.wire.values() for f in fs}
    out.update({f.id: f for f in rep.load})
    return out


def check(**kw: Any) -> Report:
    quick = kw.pop("quick", False)
    harnesses = kw.pop("harnesses", [CLAUDE_CODE])
    with FakeServer(Behaviour(**kw)) as srv:
        return run(srv.url, None, harnesses, quick=quick)


def test_healthy_server_is_ready_for_every_harness() -> None:
    rep = check(n_ctx=131072, harnesses=ALL)
    f = by_id(rep)
    for key in ("anthropic.endpoint", "anthropic.claude_code_shape", "anthropic.stream_tool", "anthropic.roundtrip",
                "chat.stream_tool", "chat.roundtrip", "responses.stream_tool", "responses.roundtrip",
                "context.fit", "speed.cache", "context.window"):
        assert f[key].status == Status.PASS, (key, f[key].detail)
    assert f["context.overflow"].status == Status.SKIP  # 131072 > default overflow cap
    assert all(rep.verdict(h) == Status.PASS for h in ALL)
    assert rep.exit_code == 0


def test_loud_overflow_passes_and_window_is_read() -> None:
    rep = check(n_ctx=32768)
    f = by_id(rep)
    assert f["context.overflow"].status == Status.PASS
    assert "exceeds the available context size" in f["context.overflow"].detail
    assert f["context.window"].status == Status.WARN  # below the 64k recommendation
    assert "export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768" in "\n".join(env_lines(rep, CLAUDE_CODE))


def test_silent_truncation_is_caught_and_measured() -> None:
    # Like a model loaded with a small default num_ctx that keeps only the end of the prompt.
    rep = check(n_ctx=4096, overflow="truncate", engine="unknown")
    f = by_id(rep)
    assert f["context.overflow"].status == Status.SKIP  # unknown engine declares no window
    fit = f["context.fit"]
    assert fit.status == Status.FAIL and "SILENT TRUNCATION" in fit.detail
    assert rep.window == fit.data["processed_tokens"] and 4000 <= rep.window <= 4096
    assert f["context.window"].status == Status.FAIL
    assert rep.exit_code == 2
    assert f"CLAUDE_CODE_MAX_CONTEXT_TOKENS={rep.window}" in "\n".join(env_lines(rep, CLAUDE_CODE))


def test_silent_truncation_with_declared_window_fails_overflow_probe() -> None:
    rep = check(n_ctx=4096, overflow="truncate", engine="ollama")
    f = by_id(rep)
    assert rep.engine.name == "ollama"
    assert f["context.overflow"].status == Status.FAIL
    assert "OLLAMA_CONTEXT_LENGTH" in (f["context.overflow"].fix or "")
    assert f["context.fit"].status == Status.FAIL  # declared window < Claude Code's first request


def test_window_smaller_than_first_request_fails_without_sending_it() -> None:
    rep = check(n_ctx=8192)
    fit = by_id(rep)["context.fit"]
    assert fit.status == Status.FAIL and "cannot even start" in fit.detail
    assert "-c 65536" in (fit.fix or "")


def test_broken_tool_argument_streaming_is_a_wire_failure() -> None:
    rep = check(stream_tool_args="bad_json", harnesses=ALL, quick=True)
    f = by_id(rep)
    for key in ("anthropic.stream_tool", "chat.stream_tool", "responses.stream_tool"):
        assert f[key].status == Status.FAIL, key
        assert "JSON" in f[key].detail


def test_missing_block_start_breaks_sse_grammar() -> None:
    rep = check(stream_tool_args="missing_start", harnesses=[CLAUDE_CODE, OPENAI_CHAT], quick=True)
    f = by_id(rep)
    assert f["anthropic.stream_tool"].status == Status.FAIL
    assert "never started" in f["anthropic.stream_tool"].detail
    assert f["chat.stream_tool"].status == Status.FAIL
    assert "without a function name" in f["chat.stream_tool"].detail


def test_rejected_claude_code_field_is_named() -> None:
    rep = check(reject_cache_control=True, quick=True)
    shape = by_id(rep)["anthropic.claude_code_shape"]
    assert shape.status == Status.FAIL
    assert set(shape.data["rejected"]) == {"system as text blocks with cache_control",
                                           "a role:'system' message inside messages", "tools with cache_control"}


def test_vllm_style_output_budget_gets_the_env_fix() -> None:
    rep = check(n_ctx=32768, reject_over_budget=True, engine="vllm", quick=True)
    f = by_id(rep)
    assert f["anthropic.claude_code_shape"].status == Status.PASS  # a tiny prompt + 32000 still fits 32768
    budget = f["anthropic.output_budget"]
    assert budget.status == Status.FAIL and "43,876 > 32,768" in budget.detail
    assert "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192" in "\n".join(env_lines(rep, CLAUDE_CODE))
    rep = check(n_ctx=131072, reject_over_budget=True, engine="vllm", quick=True)
    assert by_id(rep)["anthropic.output_budget"].status == Status.WARN
    rep = check(n_ctx=32768, quick=True)
    assert by_id(rep)["anthropic.output_budget"].status == Status.PASS


def test_missing_anthropic_endpoint() -> None:
    rep = check(anthropic=False, quick=True)
    f = by_id(rep)["anthropic.endpoint"]
    assert f.status == Status.FAIL and "upgrade llama.cpp" in (f.fix or "")


def test_no_prefix_cache_and_slow_prefill() -> None:
    rep = check(n_ctx=131072, cache_prefix=False, prefill_tps=5000)
    f = by_id(rep)
    assert f["speed.cache"].status == Status.WARN
    assert "re-process the whole conversation" in f["speed.cache"].detail
    rep = check(n_ctx=131072, cache_prefix=True, prefill_tps=5000)
    assert by_id(rep)["speed.cache"].status == Status.PASS


def test_prefill_prediction_flags_cpu_class_speed() -> None:
    rep = check(n_ctx=131072, prefill_tps=6000)  # ~2 s for 11.9k tokens: fine
    assert by_id(rep)["speed.prefill"].status == Status.PASS
    sp = rep.speed
    assert sp is not None and sp.prefill_tps is not None and 3000 < sp.prefill_tps < 7000


def test_engine_discovery() -> None:
    for engine, window in (("llama.cpp", 40960), ("ollama", 8192), ("vllm", 32768), ("lmstudio", 16384)):
        with FakeServer(Behaviour(engine=engine, n_ctx=window)) as srv:
            info = discover(Client(srv.url), "fake-coder")
        assert info.name == engine
        assert info.window == window, engine


def test_resolve_harness_aliases() -> None:
    assert resolve(["opencode", "cc"]) == [OPENAI_CHAT, CLAUDE_CODE]
    assert resolve(["all"]) == ALL
