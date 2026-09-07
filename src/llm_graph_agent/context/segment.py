"""消息结构判定的统一访问层。

把散落在 collapse.py / engine.py 里的消息类型判定与字段访问收拢到这一处，
避免各模块各自维护一份 dict + 对象 duck-typing 判断。

所有函数同时兼容两种消息表示：
- LangChain BaseMessage 对象
- 纯 dict（{"role": ...} / {"id": ...}）

只要这里的判定保持正确，上层模块（collapse / engine / snip）就不必关心
消息到底是对象还是 dict。
"""
from __future__ import annotations

from typing import Any


def get_message_id(msg: Any) -> str | None:
    """消息 ID；dict 用 "id" 键，对象用 .id。"""
    if isinstance(msg, dict):
        return msg.get("id")
    return getattr(msg, "id", None)


def get_content(msg: Any) -> Any:
    """消息内容；dict 用 "content" 键，对象用 .content。"""
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def set_content(msg: Any, content: str) -> None:
    """就地改写消息内容（返回 None，与旧 _set_content 行为一致）。"""
    if isinstance(msg, dict):
        msg["content"] = content
        return
    msg.content = content


def compression_metadata(msg: Any) -> dict:
    """取 response_metadata 里的 context_compression 段（没有则为空 dict）。"""
    if isinstance(msg, dict):
        metadata = msg.get("response_metadata", {}) or {}
    else:
        metadata = getattr(msg, "response_metadata", {}) or {}
    return metadata.get("context_compression", {}) or {}


def message_role(msg: Any) -> str:
    """把消息归一成 user / assistant / tool / system；未知类型返回类名。"""
    if isinstance(msg, dict):
        return msg.get("role", "unknown")

    name = msg.__class__.__name__
    if name == "HumanMessage":
        return "user"
    if name == "AIMessage":
        return "assistant"
    if name == "ToolMessage":
        return "tool"
    if name == "SystemMessage":
        return "system"
    return name


def is_human_message(msg: Any) -> bool:
    """是否用户消息。"""
    if isinstance(msg, dict):
        return msg.get("role") in {"user", "human"}
    return msg.__class__.__name__ == "HumanMessage"


def is_ai_message(msg: Any) -> bool:
    """是否助手消息。"""
    if isinstance(msg, dict):
        return msg.get("role") in {"assistant", "ai"}
    return msg.__class__.__name__ == "AIMessage"


def is_tool_message(msg: Any) -> bool:
    """是否工具结果消息。"""
    if isinstance(msg, dict):
        return msg.get("role") == "tool"
    return msg.__class__.__name__ == "ToolMessage"


def has_tool_calls(msg: Any) -> bool:
    """是否带 tool_calls。"""
    if isinstance(msg, dict):
        return bool(msg.get("tool_calls"))
    return bool(getattr(msg, "tool_calls", None))


def is_summary_message(msg: Any, session: dict) -> bool:
    """是否压缩摘要消息。

    依据：collapse_commits 里记录过的 summary_id，或
    response_metadata.context_compression.is_summary 标记。
    """
    msg_id = get_message_id(msg)

    summary_ids = {
        commit.get("summary_id")
        for commit in session["collapse_commits"]
    }

    if msg_id in summary_ids:
        return True

    return compression_metadata(msg).get("is_summary") is True


def is_compressed_turn_message(msg: Any) -> bool:
    """是否已被折叠的历史轮次消息。"""
    return (
        compression_metadata(msg).get("is_compressed_turn")
        is True
    )
