import io
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fakeserver import Behaviour, FakeServer

from localready import cli
from localready.client import Client
from localready.proxy import Recorder, serve, summarize
from localready.wire import validate_anthropic_stream


def turn(system_header: str, extra: list[dict[str, Any]] | None = None, **kw: Any) -> dict[str, Any]:
    return {"model": "fake-coder", "max_tokens": 64, **kw,
            "system": [{"type": "text", "text": system_header}, {"type": "text", "text": "You are a coding agent. " * 40}],
            "messages": [{"role": "user", "content": "Reply with the word READY."}] + (extra or [])}


@pytest.fixture()
def proxied(tmp_path: Path) -> Any:
    with FakeServer(Behaviour(n_ctx=2000)) as srv:
        out = io.StringIO()
        rec = Recorder(tmp_path / "calls.jsonl", out, None)
        server = serve("127.0.0.1:0", srv.url, rec)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            yield Client(f"http://127.0.0.1:{server.server_address[1]}"), rec, out, tmp_path / "calls.jsonl"
        finally:
            server.shutdown()
            server.server_close()


def test_proxy_passes_streams_through_and_measures_prefix(proxied: Any) -> None:
    client, rec, out, log = proxied
    first = client.stream("/v1/messages", turn("hdr: v1;"))
    problems, _ = validate_anthropic_stream(first)
    assert first.ok and not problems  # SSE survives the proxy byte-for-byte
    follow = [{"role": "assistant", "content": "READY"}, {"role": "user", "content": "Again."}]
    assert client.post("/v1/messages", turn("hdr: v1;", follow)).ok
    assert client.post("/v1/messages", turn("hdr: v2;", follow)).ok  # header changed: prefix broken
    r1, r2, r3 = rec.records
    assert r1.reuse is None and r1.usage.input_tokens and r1.stream
    assert r2.prefix_ok is True and (r2.reuse or 0) > 0.9
    assert r3.prefix_ok is False and r3.diverged_at == "system[0]"
    assert r3.diverged_was == "1;" + r3.diverged_was[2:] and r3.diverged_now.startswith("2;")
    text = out.getvalue()
    assert "prefix broke in system[0]" in text
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["n"] for x in lines] == [1, 2, 3]
    assert "authorization" not in log.read_text().lower() and "x-api-key" not in log.read_text().lower()


def test_proxy_records_overflow_errors(proxied: Any) -> None:
    client, rec, out, _ = proxied
    r = client.post("/v1/messages", turn("hdr;", [{"role": "user", "content": "x " * 20000}]))
    assert r.status == 400
    assert rec.records[0].error and "exceeds the available context size" in rec.records[0].error
    summary = io.StringIO()
    summarize(rec.records, summary)
    assert "1 errors" in summary.getvalue()


def test_cli_check_json_and_env(capsys: pytest.CaptureFixture[str]) -> None:
    with FakeServer(Behaviour(n_ctx=32768, engine="vllm", reject_over_budget=True)) as srv:
        code = cli.main(["check", srv.url, "--json", "--quick"])
        data = json.loads(capsys.readouterr().out)
        assert code == 2  # output budget makes Claude Code fail on this server
        assert data["engine"]["name"] == "vllm" and data["verdicts"] == {"claude-code": "fail"}
        assert cli.main(["env", srv.url]) == 0
        env = capsys.readouterr().out
    assert f"export ANTHROPIC_BASE_URL={srv.url}" in env
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768" in env and "CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192" in env


def test_cli_human_report(capsys: pytest.CaptureFixture[str]) -> None:
    with FakeServer(Behaviour(n_ctx=131072)) as srv:
        assert cli.main(["check", srv.url, "-H", "claude-code,codex"]) == 0
    text = capsys.readouterr().out
    assert "Claude Code: READY" in text and "Codex CLI: READY" in text
    assert "wire-level: harness not run live" in text
    assert "[model_providers.local]" in text


def test_cli_unknown_harness_and_no_server(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli.main(["check", "http://127.0.0.1:9", "-H", "nope"]) == 1
    assert "unknown harness" in capsys.readouterr().err
    monkeypatch.setattr(cli, "DEFAULT_PORTS", (("none", 9),))
    with pytest.raises(SystemExit, match="no model server answered"):
        cli.main(["check"])
