"""dto 序列化测试。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llm_graph_agent.api.dto import (
    messages_to_dto,
    update_to_jsonable,
)


def test_message_dto_roles():
    assert messages_to_dto(HumanMessage(content="h"))[0]["role"] == "user"
    assert messages_to_dto(AIMessage(content="a"))[0]["role"] == "assistant"
    assert messages_to_dto(ToolMessage(content="t", tool_call_id="c"))[0]["role"] == "tool"
    assert messages_to_dto(SystemMessage(content="s"))[0]["role"] == "system"


def test_message_dto_handles_dicts_and_nonstring_content():
    dto = messages_to_dto({"role": "user", "content": {"a": 1}})[0]
    assert dto["role"] == "user"
    assert '"a": 1' in dto["content"]


def test_update_to_jsonable_converts_messages():
    update = {
        "assistant": {
            "messages": [AIMessage(content="回复", id="ai-1")],
            "turn_id": 1,
        }
    }
    result = update_to_jsonable(update)
    assert result["assistant"]["messages"][0]["role"] == "assistant"
    assert result["assistant"]["messages"][0]["content"] == "回复"

    import json
    json.dumps(result)  # 应可序列化


def test_update_to_jsonable_passthrough_non_message_fields():
    result = update_to_jsonable({"assistant": {"status": "completed"}})
    assert result["assistant"]["status"] == "completed"
