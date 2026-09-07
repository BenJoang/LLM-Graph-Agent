"""压缩引擎（MessageManage）与重试适配器的行为测试。

这些测试锁定"降级压缩层级语义"和"prepare / project / adapter"的
对外契约，属于改动最敏感、最需要行为锁定的部分。
"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from llm_graph_agent.context.compression import CompressionRetryAdapter, MessageManage
from llm_graph_agent.context.compression.session import make_empty_compression_session
from llm_graph_agent.context.messages import mark_ai_message


def _manager(max_tokens: int = 32768, **kwargs) -> MessageManage:
    async def summarize(text: str) -> str:
        return "S:" + text[:20]

    return MessageManage(
        max_tokens=max_tokens,
        asummarize_fn=kwargs.pop("asummarize_fn", summarize),
        **kwargs,
    )


def _plain_pair():
    return [
        HumanMessage(content="问题", id="q-1"),
        AIMessage(content="回答", id="a-1"),
    ]


async def test_prepare_no_overflow_is_identity():
    mgr = _manager(max_tokens=32768)
    messages = _plain_pair()
    out, compressed, session = await mgr.prepare_messages_for_query(messages, None)
    assert compressed is False
    assert [m.id for m in out] == ["q-1", "a-1"]
    assert isinstance(session, dict)


async def test_prepare_small_returns_empty_session():
    mgr = _manager()
    _, _, session = await mgr.prepare_messages_for_query(_plain_pair(), None)
    assert session["collapse_commits"] == []
    assert session["collapse_message_ids"] == []


async def test_project_committed_context_round_trips():
    mgr = _manager()
    messages = [HumanMessage(content="x", id="x-1")]
    projected, session = mgr.project_committed_context(messages, None)
    assert [m.id for m in projected] == ["x-1"]
    assert session["version"] == 1


async def test_compress_for_retry_level_2_snips_tool_outputs():
    """level 2 是纯本地裁剪，不调用摘要模型。"""
    mgr = _manager()
    big_tool = ToolMessage(
        content="x" * 6000,
        id="t-1",
        tool_call_id="c-1",
    )
    messages = [big_tool]
    out, session, changed = await mgr.compress_for_retry(
        messages=messages,
        original_messages=messages,
        level=2,
        compression_session=None,
        current_turn_id=1,
    )
    assert changed is True
    # 4000 head + 省略标记 + 尾部，必须比原始短
    assert len(out[0].content) < 6000
    assert "[tool output snipped" in out[0].content


async def test_compress_for_retry_invalid_level_raises():
    mgr = _manager()
    messages = _plain_pair()
    try:
        await mgr.compress_for_retry(
            messages=messages,
            original_messages=messages,
            level=99,
            compression_session=None,
            current_turn_id=1,
        )
        raise AssertionError("应抛出 ValueError")
    except ValueError as e:
        assert "99" in str(e)


async def test_compress_for_retry_level_1_requires_summarizer():
    """level 1 会折摘要，但没有可合并摘要时返回 changed=False（不崩）。"""
    mgr = _manager()
    messages = _plain_pair()
    out, session, changed = await mgr.compress_for_retry(
        messages=messages,
        original_messages=messages,
        level=1,
        compression_session=None,
        current_turn_id=1,
    )
    assert changed is False
    assert [m.id for m in out] == ["q-1", "a-1"]


async def test_retry_adapter_tracks_last_level_and_session():
    mgr = _manager()
    messages = _plain_pair()
    adapter = CompressionRetryAdapter(mgr, None, current_turn_id=1)
    out = await adapter.acall(messages, messages, level=2)
    assert adapter.last_level == 2
    assert isinstance(out, list)
    assert isinstance(adapter.compression_session, dict)


async def test_engine_requires_asummarize_fn_when_summarizing():
    """调用摘要但没配 asummarize_fn 时给出明确错误。"""
    mgr = _manager(asummarize_fn=None)
    messages = _plain_pair()
    try:
        await mgr._build_simple_summary(messages)
        raise AssertionError("应抛出 RuntimeError")
    except RuntimeError as e:
        assert "asummarize_fn".upper() in str(e).upper() or "摘要" in str(e)


async def test_make_summary_message_and_turn_markers_round_trip():
    """引擎生成的摘要消息能被 segment 识别为摘要。"""
    from llm_graph_agent.context.compression.collapse import (
        _make_compressed_turn_message,
        _make_summary_message,
    )
    from llm_graph_agent.context.segment import (
        is_compressed_turn_message,
        is_summary_message,
    )

    summary = _make_summary_message(
        summary_content="摘要内容",
        collapse_id="c-1",
        turn_id=3,
        summary_kind="normal",
    )
    session = make_empty_compression_session()
    session["collapse_commits"].append(
        {"summary_id": summary.id}
    )
    assert is_summary_message(summary, session)

    compressed = _make_compressed_turn_message(
        content="压缩轮次",
        collapse_id="c-2",
        turn_id=2,
    )
    assert is_compressed_turn_message(compressed)
