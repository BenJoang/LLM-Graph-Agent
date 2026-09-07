"""压缩会话的序列化与存取。

CompressionSession 记录已经执行的压缩计划（collapse commits），
以及被折叠的消息 ID 集合。这里只做纯数据操作，不涉及模型调用。
"""
from __future__ import annotations

from copy import deepcopy
from typing_extensions import TypedDict


class CompressionSession(TypedDict):
    version: int
    collapse_commits: list[dict]
    collapse_message_ids: list[str]

def make_empty_compression_session() -> CompressionSession:
    return {
        "version": 1,
        "collapse_commits": [],
        "collapse_message_ids": [],
    }

def load_compression_session(session: CompressionSession | None) -> dict:
    session = session or make_empty_compression_session()

    return {
        "collapse_commits": deepcopy(
            session.get("collapse_commits", [])
        ),
        "collapse_message_ids": set(
            session.get("collapse_message_ids", [])
        ),
    }


def dump_compression_session(session: dict) -> CompressionSession:
    return {
        "version": 1,
        "collapse_commits": deepcopy(
            session["collapse_commits"]
        ),
        "collapse_message_ids": sorted(
            session["collapse_message_ids"]
        ),
    }
