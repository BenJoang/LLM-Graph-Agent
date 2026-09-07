"""overflow 阈值纯函数与 serialize 序列化的测试。"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from llm_graph_agent.context.compression.overflow import (
    CompactThresholds,
    at_or_below_retry_target,
    compute_thresholds,
    should_snip,
    should_summarize,
    still_over_max,
)
from llm_graph_agent.context.compression.serialize import (
    format_messages_for_summary,
)


def test_compute_thresholds_default_ratios():
    t = compute_thresholds(max_tokens=1000)
    assert t.max_tokens == 1000
    assert t.snip_tokens == 500
    assert t.summarize_tokens == 700
    assert t.retry_target_tokens == 850


def test_compute_thresholds_custom_ratios():
    t = compute_thresholds(
        max_tokens=2000,
        snip_ratio=0.4,
        summarize_ratio=0.6,
        retry_target_ratio=0.9,
    )
    assert t.snip_tokens == 800
    assert t.summarize_tokens == 1200
    assert t.retry_target_tokens == 1800


def test_threshold_frozen_dataclass():
    t = compute_thresholds(max_tokens=1000)
    assert isinstance(t, CompactThresholds)
    assert t.max_tokens == 1000


def test_should_snip_summarize_and_over_max():
    t = compute_thresholds(max_tokens=1000)
    assert should_snip(501, t)
    assert not should_snip(500, t)
    assert should_summarize(701, t)
    assert not should_summarize(700, t)
    assert still_over_max(1001, t)
    assert not still_over_max(1000, t)
    assert not still_over_max(900, t)


def test_at_or_below_retry_target():
    t = compute_thresholds(max_tokens=1000)
    assert at_or_below_retry_target(850, t)
    assert at_or_below_retry_target(800, t)
    assert not at_or_below_retry_target(851, t)


def test_format_messages_for_summary_lists_role_content():
    messages = [
        HumanMessage(content="你好", id="h-1"),
        AIMessage(content="回复", id="a-1"),
        ToolMessage(content="工具结果", id="t-1", tool_call_id="c-1"),
    ]
    text = format_messages_for_summary(messages)
    assert text.startswith("[")
    assert "user" in text and "hi你好".replace("hi", "") in text
    assert '"role": "assistant"' in text
    assert '"role": "tool"' in text


def test_format_messages_for_summary_skips_empty():
    messages = [HumanMessage(content="", id="h-1"), AIMessage(content="x", id="a-1")]
    text = format_messages_for_summary(messages)
    assert '"role": "user"' not in text
    assert '"role": "assistant"' in text
