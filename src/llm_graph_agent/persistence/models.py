"""对话日志的数据模型：类型、异常、数据记录。

原 conversation_store.py 的数据定义部分，独立成模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

from langchain_core.messages import BaseMessage

from llm_graph_agent.paths import PROJECT_ROOT


DEFAULT_CONVERSATION_SQLITE_PATH = (
    PROJECT_ROOT / "outputs" / "conversations" / "conversation.sqlite3"
)


EventType = Literal[
    "user",
    "assistant",
    "tool",
    "system",
]

FinalRunStatus = Literal[
    "completed",
    "cancelled",
    "failed",
    "interrupted",
]

VALID_EVENT_TYPES = frozenset({
    "user",
    "assistant",
    "tool",
    "system",
})

VALID_FINAL_RUN_STATUSES = frozenset({
    "completed",
    "cancelled",
    "failed",
    "interrupted",
})

VALID_RUN_STATUSES = frozenset({
    "running",
    *VALID_FINAL_RUN_STATUSES,
})

DEFAULT_CONVERSATION_SQLITE_PATH = (
    PROJECT_ROOT / "outputs" / "conversations" / "conversation.sqlite3"
)

class ActiveRunError(RuntimeError):
    """同一个会话已经存在正在运行的任务。"""


class RunNotFoundError(RuntimeError):
    """指定的 run 不存在。"""


class RunNotWritableError(RuntimeError):
    """run 当前状态不允许追加新事件。"""


class EventConflictError(RuntimeError):
    """同一个 message_id 被用于不同的消息内容。"""


class RunStateConflictError(RuntimeError):
    """run 已经结束，不能转换到另一个终态。"""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    session_id: str
    turn_no: int
    status: str
    base_event_id: int
    checkpoint_thread_id: str | None
    heartbeat_at: datetime | None
    started_at: datetime
    finished_at: datetime | None
    error: str | None


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    session_id: str
    run_id: str
    turn_no: int
    ordinal: int
    message_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ContextRecord:
    session_id: str
    through_event_id: int
    version: int
    projected_messages: list[BaseMessage]
    compression_session: dict[str, Any]
    updated_at: datetime
def _clean_identifier(
    value: str,
    *,
    field_name: str,
) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{field_name} 不能为空")

    if len(cleaned) > 255:
        raise ValueError(f"{field_name} 长度不能超过 255")

    return cleaned


def _clean_event_type(value: str) -> EventType:
    cleaned = value.strip().lower()

    if cleaned not in VALID_EVENT_TYPES:
        raise ValueError(
            "event_type 必须是："
            + ", ".join(sorted(VALID_EVENT_TYPES))
        )

    return cast(EventType, cleaned)


def _clean_final_run_status(value: str) -> FinalRunStatus:
    cleaned = value.strip().lower()

    if cleaned not in VALID_FINAL_RUN_STATUSES:
        raise ValueError(
            "最终 run 状态必须是："
            + ", ".join(sorted(VALID_FINAL_RUN_STATUSES))
        )

    return cast(FinalRunStatus, cleaned)


def _clean_run_statuses(statuses: Iterable[str]) -> tuple[str, ...]:
    cleaned = tuple(dict.fromkeys(
        status.strip().lower()
        for status in statuses
    ))

    if not cleaned:
        raise ValueError("statuses 不能为空")

    invalid = set(cleaned) - VALID_RUN_STATUSES
    if invalid:
        raise ValueError(
            "未知 run 状态："
            + ", ".join(sorted(invalid))
        )

    return cleaned


def _message_identifier(
    message: BaseMessage,
    explicit_message_id: str | None,
) -> str:
    if (
        explicit_message_id is not None
        and message.id is not None
        and str(message.id).strip() != explicit_message_id.strip()
    ):
        raise ValueError(
            "显式 message_id 与 message.id 不一致"
        )

    message_id = explicit_message_id or message.id
    if message_id is None:
        raise ValueError(
            "消息没有 id；请传入稳定的 message_id，"
            "以保证事件写入可以幂等重试"
        )

    return _clean_identifier(
        str(message_id),
        field_name="message_id",
    )


