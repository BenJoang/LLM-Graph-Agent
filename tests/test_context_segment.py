"""segment 消息判定访问层的测试。"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from llm_graph_agent.context.segment import (
    compression_metadata,
    get_content,
    get_message_id,
    has_tool_calls,
    is_ai_message,
    is_compressed_turn_message,
    is_human_message,
    is_summary_message,
    is_tool_message,
    message_role,
    set_content,
)
from llm_graph_agent.context.compression.session import (
    make_empty_compression_session,
)


def test_message_role_maps_langchain_types():
    assert message_role(HumanMessage(content="h")) == "user"
    assert message_role(AIMessage(content="a")) == "assistant"
    assert message_role(ToolMessage(content="t", tool_call_id="c")) == "tool"
    assert message_role(SystemMessage(content="s")) == "system"


def test_message_role_falls_back_to_class_name():
    class Weird:
        pass

    assert message_role(Weird()) == "Weird"


def test_is_ai_is_human_is_tool():
    human = HumanMessage(content="h")
    ai = AIMessage(content="a")
    tool = ToolMessage(content="t", tool_call_id="c")

    assert is_human_message(human)
    assert not is_human_message(ai)
    assert is_ai_message(ai)
    assert not is_ai_message(human)
    assert is_tool_message(tool)
    assert not is_tool_message(ai)


def test_desugar_dict_shapes():
    """同一套判定函数也要能处理纯 dict 消息。"""
    assert is_human_message({"role": "user", "content": "h"})
    assert is_human_message({"role": "human", "content": "h"})
    assert is_ai_message({"role": "assistant"})
    assert is_ai_message({"role": "ai"})
    assert is_tool_message({"role": "tool"})
    assert message_role({"role": "unknown-ish"}) == "unknown-ish"


def test_get_and_set_content_both_shapes():
    obj = HumanMessage(content="before")
    assert get_content(obj) == "before"
    assert get_content({"content": "d"}) == "d"
    assert get_content({"id": "x"}) is None

    set_content(obj, "after")
    assert obj.content == "after"

    d = {"content": "a"}
    set_content(d, "b")
    assert d["content"] == "b"


def test_get_message_id_both_shapes():
    assert get_message_id(HumanMessage(content="h", id="obj-1")) == "obj-1"
    assert get_message_id({"id": "dict-1"}) == "dict-1"
    assert get_message_id({"role": "user"}) is None


def test_compression_metadata_normalizes_missing():
    assert compression_metadata(HumanMessage(content="h")) == {}
    assert compression_metadata({"role": "user"}) == {}


def test_is_summary_message_by_metadata():
    msg = AIMessage(
        content="sum",
        id="s-1",
        response_metadata={"context_compression": {"is_summary": True}},
    )
    assert is_summary_message(msg, make_empty_compression_session())


def test_is_summary_message_by_committed_ids():
    session = make_empty_compression_session()
    session["collapse_commits"].append({"summary_id": "committed-sum"})
    msg = HumanMessage(content="x", id="committed-sum")
    assert is_summary_message(msg, session)


def test_is_compressed_turn_message():
    plain = HumanMessage(content="h")
    compressed = HumanMessage(
        content="c",
        id="c-1",
        response_metadata={
            "context_compression": {"is_compressed_turn": True}
        },
    )
    assert is_compressed_turn_message(compressed)
    assert not is_compressed_turn_message(plain)


def test_has_tool_calls():
    with_tools = AIMessage(
        content="",
        id="a-1",
        tool_calls=[{"name": "weather", "args": {}, "id": "c-1", "type": "tool_call"}],
    )
    plain = AIMessage(content="ok", id="a-2")
    assert has_tool_calls(with_tools)
    assert not has_tool_calls(plain)
