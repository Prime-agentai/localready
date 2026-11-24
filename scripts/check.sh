#!/usr/bin/env bash
# Types, tests, and the CLI end to end against fake engines. Usage: PYTHON=python3 bash scripts/check.sh
set -euo pipefail
trap 'echo "check.sh: failed at line $LINENO" >&2' ERR
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1

"$PY" -m mypy
"$PY" -m mypy --platform win32
"$PY" -m mypy --platform darwin
"$PY" -m pytest -q

port() { "$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
serve() {  # serve PORT [behaviour...]: start a fake engine, wait until it answers
  "$PY" tests/fakeserver.py "$@" > /dev/null 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 50); do
    "$PY" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$1/v1/models', timeout=1)" 2>/dev/null && return 0
    sleep 0.1
  done
  echo "fake server on $1 did not start" >&2; return 1
}
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
out="$(mktemp -d)"

echo "== healthy llama.cpp-like server: READY, exit 0"
P1=$(port); serve "$P1" n_ctx=131072
"$PY" -m localready check "http://127.0.0.1:$P1" -H all > "$out/ok.txt" 2>/dev/null
grep -q "Claude Code: READY" "$out/ok.txt"
grep -q "export CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072" "$out/ok.txt"

echo "== silently truncating server (4k window): NOT READY, exit 2, window measured"
P2=$(port); serve "$P2" n_ctx=4096 overflow=truncate engine=unknown
code=0; "$PY" -m localready check "http://127.0.0.1:$P2" > "$out/trunc.txt" 2>/dev/null || code=$?
[ "$code" = 2 ] || { echo "expected exit 2, got $code" >&2; cat "$out/trunc.txt"; exit 1; }
grep -q "SILENT TRUNCATION" "$out/trunc.txt"
grep -q "Claude Code: NOT READY" "$out/trunc.txt"

echo "== vLLM-like output budget: env prints the output cap"
P3=$(port); serve "$P3" n_ctx=32768 engine=vllm reject_over_budget=true
"$PY" -m localready env "http://127.0.0.1:$P3" > "$out/env.txt"
grep -q "CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192" "$out/env.txt"

echo "== proxy: records calls, locates a prefix break"
P4=$(port)
"$PY" -m localready proxy "http://127.0.0.1:$P1" --listen "127.0.0.1:$P4" --log "$out/calls.jsonl" > "$out/proxy.txt" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 50); do grep -q "localready proxy" "$out/proxy.txt" 2>/dev/null && break; sleep 0.1; done
"$PY" - "$P4" <<'PYEOF'
import json, sys, urllib.request
def call(header):
    body = {"model": "fake-coder", "max_tokens": 16, "system": [{"type": "text", "text": header}, {"type": "text", "text": "Agent rules. " * 50}],
            "messages": [{"role": "user", "content": "Reply with the word READY."}]}
    req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/messages", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()
call("build: 1;"); call("build: 1;"); call("build: 2;")
PYEOF
[ "$(wc -l < "$out/calls.jsonl" | tr -d ' ')" = 3 ]
grep -q "prefix broke in system\[0\]" "$out/proxy.txt"

echo "check.sh: all green"
