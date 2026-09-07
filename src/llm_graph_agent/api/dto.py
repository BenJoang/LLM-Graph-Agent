"""消息/更新 → JSON 可序列化 DTO。

供 API 层把 LangChain 消息和节点更新转成可下发给客户端的纯 dict。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def _message_to_dto(message: Any) -> dict:
    """把单条消息（对象或 dict）转成 JSON 安全的 dict。"""
    if isinstance(message, dict):
        role = str(message.get("role", message.get("type", "unknown")))
        return {
            "id": str(message.get("id") or ""),
            "role": role,
            "content": _text_content(message.get("content")),
            "tool_calls": message.get("tool_calls") or [],
            "tool_call_id": message.get("tool_call_id"),
            "name": message.get("name"),
            "status": message.get("status"),
            "turn_id": message.get("turn_id"),
        }

    class_name = message.__class__.__name__
    role = {
        "HumanMessage": "user",
        "AIMessage": "assistant",
        "ToolMessage": "tool",
        "SystemMessage": "system",
    }.get(class_name, class_name)

    return {
        "id": str(getattr(message, "id", None) or ""),
        "role": role,
        "content": _text_content(getattr(message, "content", "")),
        "tool_calls": getattr(message, "tool_calls", None) or [],
        "tool_call_id": getattr(message, "tool_call_id", None),
        "name": getattr(message, "name", None),
        "status": getattr(message, "status", None),
    }


def messages_to_dto(messages: Any) -> list[dict]:
    """把消息（单个/列表）转成 DTO 列表。"""
    if isinstance(messages, (list, tuple)):
        return [_message_to_dto(m) for m in messages]
    if isinstance(messages, BaseMessage) or isinstance(messages, dict):
        return [_message_to_dto(messages)]
    return []


def update_to_jsonable(update: dict) -> dict:
    """把节点更新（node → {messages: [...]}) 转成 JSON 可序列化 dict。

    非 messages 字段保持原样（通常是小值），dangerous 内容在输出端兜底。
    """
    result: dict[str, Any] = {}
    for node_name, output in update.items():
        if isinstance(output, dict) and isinstance(output.get("messages"), (list, BaseMessage)):
            node_payload = dict(output)
            node_payload["messages"] = messages_to_dto(output.get("messages"))
            result[node_name] = node_payload
        else:
            result[node_name] = output
    return result
