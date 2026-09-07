"""对话存储抽象接口。

定义 ConversationStore Protocol。SQLite / Postgres 两个实现都满足该接口。
"""
from __future__ import annotations

from typing import Any, Iterable, Literal, Protocol, cast

from langchain_core.messages import BaseMessage

from llm_graph_agent.persistence.models import (
    ContextRecord,
    EventRecord,
    RunRecord,
)


class ConversationStore(Protocol):
    """PostgreSQL 和 SQLite 共同遵守的最小存储契约。"""

    database_url: str

    def setup(self) -> None: ...

    def close(self) -> None: ...

    def ensure_session(self, session_id: str) -> None: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...

    def delete_session(self, session_id: str) -> bool: ...

    def interrupt_running_runs(
        self,
        *,
        reason: str = "process restarted",
    ) -> int: ...

    def begin_run(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
    ) -> RunRecord: ...

    def append_event(
        self,
        run_id: str,
        *,
        message_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> EventRecord: ...

    def append_message(
        self,
        run_id: str,
        message: BaseMessage,
        *,
        message_id: str | None = None,
    ) -> EventRecord: ...

    def list_events(
        self,
        session_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[EventRecord]: ...

    def load_messages(
        self,
        session_id: str,
        *,
        statuses: Iterable[str] = ("completed",),
    ) -> list[BaseMessage]: ...

    def load_context(
        self,
        session_id: str,
    ) -> ContextRecord | None: ...

    def commit_run(
        self,
        run_id: str,
        *,
        projected_messages: Iterable[BaseMessage],
        compression_session: dict[str, Any],
    ) -> RunRecord: ...

    def replace_context(
        self,
        session_id: str,
        *,
        projected_messages: Iterable[BaseMessage],
        compression_session: dict[str, Any],
    ) -> ContextRecord: ...

    def finish_run(
        self,
        run_id: str,
        *,
        status: FinalRunStatus,
        error: str | None = None,
    ) -> RunRecord: ...

    def complete_run(self, run_id: str) -> RunRecord: ...

    def cancel_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
    ) -> RunRecord: ...

    def cancel_run_with_context(
        self,
        run_id: str,
        *,
        projected_messages: Iterable[BaseMessage],
        compression_session: dict[str, Any],
        reason: str | None = None,
    ) -> RunRecord: ...

    def fail_run(
        self,
        run_id: str,
        *,
        error: str,
    ) -> RunRecord: ...

    def interrupt_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
    ) -> RunRecord: ...
