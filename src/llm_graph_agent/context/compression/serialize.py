"""消息 → 摘要输入文本 的序列化（纯函数）。

负责把一组消息格式化成 role/content 的 JSON 文本，作为摘要模型的输入。
不含任何模型调用或状态改写。
"""
from __future__ import annotations

import json

from llm_graph_agent.context.segment import (
    get_content,
    message_role,
)


def format_messages_for_summary(messages: list) -> str:
    """把消息列表序列化成 role/content 的 JSON 字符串。"""
    parts = []

    for msg in messages:
        content = get_content(msg)
        if not content:
            continue

        parts.append(
            {
                "role": message_role(msg),
                "content": str(content),
            }
        )

    return json.dumps(parts, ensure_ascii=False, indent=2)
