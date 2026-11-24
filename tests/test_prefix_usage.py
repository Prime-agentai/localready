from localready.prefix import divergence, segments
from localready.usage import usage_from_body, usage_from_sse

TOOLS = [{"name": "Read", "description": "read", "input_schema": {"type": "object"}}]


def cc_turn(system_message_as_blocks: bool, extra: list[dict[str, object]] | None = None) -> dict[str, object]:
    env = "# Environment\nWorking directory: /tmp/p"
    content: object = ([{"type": "text", "text": env, "cache_control": {"type": "ephemeral"}}]
                       if system_message_as_blocks else env)
    return {
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.330;"},
                   {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}],
        "tools": TOOLS,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Read calc.py"}]},
                     {"role": "system", "content": content}] + (extra or []),
    }


def test_blocks_and_string_render_the_same() -> None:
    # Claude Code 2.1.330 sends the same environment message as blocks on turn 1 and as a
    # string on turn 2. A chat template renders both identically, so the prefix is intact.
    turn2_extra = [{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                                      "input": {"file_path": "calc.py"},
                                                      "cache_control": {"type": "ephemeral"}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "def add"}]}]
    d = divergence(cc_turn(True), cc_turn(False, turn2_extra))
    assert d.identical_prefix
    assert d.reuse > 0.7


def test_changed_header_is_located() -> None:
    a = cc_turn(True)
    b = cc_turn(True)
    b["system"][0]["text"] = "x-anthropic-billing-header: cc_version=2.1.331;"  # type: ignore[index]
    d = divergence(a, b)
    assert not d.identical_prefix
    assert d.segment == "system[0]"
    assert d.after_a.startswith("0;") and d.after_b.startswith("1;")
    assert d.reuse < 0.3


def test_cache_control_moving_does_not_count() -> None:
    a = cc_turn(True)
    b = cc_turn(True)
    b["system"][1].pop("cache_control")  # type: ignore[index]
    b["tools"] = [dict(TOOLS[0], cache_control={"type": "ephemeral"})]
    assert divergence(a, b).identical_prefix


def test_segments_cover_openai_and_responses() -> None:
    chat = {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                         {"role": "assistant", "content": None,
                          "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]}]}
    labels = [s.label for s in segments(chat)]
    assert labels == ["tools[0:f]", "messages[0:system]", "messages[1:user]", "messages[2:assistant]"]
    resp = {"instructions": "be brief", "input": [{"role": "user", "content": "hi"},
                                                  {"type": "function_call_output", "call_id": "c", "output": "7"}]}
    assert [s.label for s in segments(resp)] == ["instructions", "input[0:user]", "input[1:function_call_output]"]


def test_usage_anthropic_stream_counts_cache_reads() -> None:
    sse = ('event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":120,'
           '"cache_read_input_tokens":9000,"output_tokens":0}}}\n\n'
           'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":42}}\n\n')
    u = usage_from_sse(sse)
    assert (u.input_tokens, u.cached_tokens, u.output_tokens) == (9120, 9000, 42)


def test_usage_openai_llamacpp_and_ollama() -> None:
    u = usage_from_body('{"usage":{"prompt_tokens":500,"completion_tokens":9,'
                        '"prompt_tokens_details":{"cached_tokens":400}}}')
    assert (u.input_tokens, u.cached_tokens, u.output_tokens) == (500, 400, 9)
    u = usage_from_body('{"timings":{"prompt_n":100,"cache_n":900,"prompt_ms":250.0,"predicted_ms":80.0}}')
    assert (u.input_tokens, u.cached_tokens, u.prompt_ms) == (1000, 900, 250.0)
    u = usage_from_body('{"message":{"content":"a"},"done":false}\n'
                        '{"done":true,"prompt_eval_count":4096,"eval_count":7,"prompt_eval_duration":2000000000}')
    assert (u.input_tokens, u.output_tokens, u.prompt_ms) == (4096, 7, 2000.0)
