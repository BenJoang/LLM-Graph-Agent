"""graph 结构测试：工具名收敛、子代理收官注入。"""
from __future__ import annotations

from llm_graph_agent.graph.tool_agent import (
    DEFAULT_TOOL_NAMES,
    SUBAGENT_EXCLUDE,
    get_available_tool_names,
)
from llm_graph_agent.graph.sub_agent import (
    FINALIZE_THRESHOLD,
    build_subagent_injection_message,
)


def test_get_available_tool_names_uses_defaults_when_none():
    names = get_available_tool_names(None, DEFAULT_TOOL_NAMES)
    assert names == DEFAULT_TOOL_NAMES


def test_get_available_tool_names_dedupes_and_excludes():
    names = get_available_tool_names(
        ["read_file", "grep", "read_file", "agenttool"],
        DEFAULT_TOOL_NAMES,
        exclude=SUBAGENT_EXCLUDE,
    )
    assert names == ["read_file", "grep"]
    assert "agenttool" not in names


def test_get_available_tool_names_keeps_order():
    names = get_available_tool_names(["grep", "read_file"], DEFAULT_TOOL_NAMES)
    assert names == ["grep", "read_file"]


def test_subagent_exclude_contains_agenttool():
    assert "agenttool" in SUBAGENT_EXCLUDE


def test_finalize_threshold_is_four():
    assert FINALIZE_THRESHOLD == 4


def test_subagent_injection_message_shape():
    msg = build_subagent_injection_message()
    assert msg.name == "subagent_budget_controller"
    assert msg.additional_kwargs.get("synthetic") is True
    assert "禁止调用任何工具" in msg.content
