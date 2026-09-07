"""retry 模块：错误识别 + 异步重试循环的行为测试。"""
from __future__ import annotations

from openai import APITimeoutError, BadRequestError
from langchain_core.messages import AIMessage, HumanMessage

from llm_graph_agent.context.messages import get_message_turn_id, mark_ai_message
from llm_graph_agent.context.retry import (
    invoke_with_retry,
    is_context_overflow_error,
)


def _overflow_error() -> BadRequestError:
    body = {"message": "maximum context length exceeded", "param": None}
    response = type("R", (), {"status_code": 400, "request": None, "headers": {}})()
    return BadRequestError(
        message="maximum context length exceeded",
        response=response,
        body=body,
    )


def test_is_context_overflow_error_matches_bad_request():
    assert is_context_overflow_error(_overflow_error())


def test_is_context_overflow_error_false_for_other_errors():
    response = type("R", (), {"status_code": 500, "request": None, "headers": {}})()
    not_overflow = BadRequestError(
        message="some other error",
        response=response,
        body={"message": "some other error", "param": None},
    )
    assert not is_context_overflow_error(not_overflow)
    assert not is_context_overflow_error(RuntimeError("boom"))


def test_is_context_overflow_error_matches_param_input_tokens():
    body = {"message": "please reduce input", "param": "input_tokens"}
    response = type("R", (), {"status_code": 400, "request": None, "headers": {}})()
    err = BadRequestError(
        message="please reduce input",
        response=response,
        body=body,
    )
    assert is_context_overflow_error(err)


async def test_invoke_with_retry_returns_on_first_success():
    async def invoke(messages):
        return AIMessage(content="ok", id="mo-1")

    messages = [HumanMessage(content="q", id="q-1")]
    result = await invoke_with_retry(
        invoke,
        messages,
        messages,
        compress_fn=_async_noop_compress,
        turn_id=5,
    )
    assert result.id == "mo-1"
    assert get_message_turn_id(result) == 5


async def test_invoke_with_retry_marks_turn_id_without_overriding():
    async def invoke(messages):
        return AIMessage(content="ok", id="mo-1")

    already_marked = AIMessage(content="ok", id="mo-1")
    mark_ai_message(already_marked, turn_id=7)

    async def invoke_marked(messages):
        return already_marked

    messages = [HumanMessage(content="q", id="q-1")]
    result = await invoke_with_retry(
        invoke_marked,
        messages,
        messages,
        compress_fn=_async_noop_compress,
        turn_id=7,
    )
    assert get_message_turn_id(result) == 7


async def test_timeout_retries_with_backoff_then_success():
    calls = {"n": 0}

    async def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise APITimeoutError("timeout")
        return AIMessage(content="ok", id="mo-1")

    messages = [HumanMessage(content="q", id="q-1")]
    result = await invoke_with_retry(
        flaky,
        messages,
        messages,
        compress_fn=_async_noop_compress,
        base_delay=0.001,
        max_delay=0.01,
    )
    assert result.content == "ok"
    assert calls["n"] == 2


async def test_timeout_relents_no_more_after_max():
    calls = {"n": 0}

    async def always_timeout(messages):
        calls["n"] += 1
        raise APITimeoutError("timeout")

    messages = [HumanMessage(content="q", id="q-1")]
    try:
        await invoke_with_retry(
            always_timeout,
            messages,
            messages,
            compress_fn=_async_noop_compress,
            max_timeout_retries=2,
            base_delay=0.001,
            max_delay=0.01,
        )
        raise AssertionError("应超时抛错")
    except RuntimeError as e:
        assert "超时" in str(e)
    assert calls["n"] == 3  # 首次 + 2 次重试


async def test_context_overflow_triggers_compress_and_retries():
    """上下文超限 → 调 compress_fn → 重试成功。"""
    calls = {"n": 0}
    compress_calls = {"n": 0}

    async def first_overflow_then_ok(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _overflow_error()
        return AIMessage(content="ok-after-compress", id="mo-1")

    async def compress_fn(current, original, level):
        compress_calls["n"] += 1
        assert level == 1
        return original

    messages = [HumanMessage(content="q", id="q-1")]
    result = await invoke_with_retry(
        first_overflow_then_ok,
        messages,
        messages,
        compress_fn=compress_fn,
    )
    assert result.content == "ok-after-compress"
    assert calls["n"] == 2
    assert compress_calls["n"] == 1


async def test_context_overflow_fails_after_max_compressions():
    calls = {"n": 0}

    async def always_overflow(messages):
        calls["n"] += 1
        raise _overflow_error()

    async def compress_fn(current, original, level):
        return original

    messages = [HumanMessage(content="q", id="q-1")]
    try:
        await invoke_with_retry(
            always_overflow,
            messages,
            messages,
            compress_fn=compress_fn,
            max_context_retries=2,
        )
        raise AssertionError("应压缩重试耗尽后抛错")
    except RuntimeError as e:
        assert "压缩" in str(e) or "超限" in str(e)
    assert calls["n"] == 3  # 首次 + 2 次压缩重试


async def test_cancelled_error_propagates():
    import asyncio

    async def cancel(messages):
        raise asyncio.CancelledError()

    messages = [HumanMessage(content="q", id="q-1")]
    try:
        await invoke_with_retry(
            cancel,
            messages,
            messages,
            compress_fn=_async_noop_compress,
        )
        raise AssertionError("应在取消失败后抛错")
    except asyncio.CancelledError:
        pass


async def _async_noop_compress(current, original, level):
    return original
