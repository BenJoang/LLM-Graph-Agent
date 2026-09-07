"""token 估算。

这是一个很粗略的估算：按字符数 / 2 估算 token 数。
仅用于决定是否需要压缩，不追求精确。
"""
from __future__ import annotations


def estimate_tokens(messages: list) -> int:

    chars = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", "")

        if content:
            chars += len(str(content))
    return chars // 2
