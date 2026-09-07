"""PostgreSQL 对话日志的对象适配器。

对 persistence 的其它模块暴露统一的 ConversationStore 接口。
"""
from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import BaseMessage

from llm_graph_agent.persistence.models import (
    ContextRecord,
    EventRecord,
    RunRecord,
)
from llm_graph_agent.persistence.postgres.operations import (
    _postgres_connection_url,
    append_event,
    append_message,
    begin_run,
    cancel_run,
    cancel_run_with_context,
    commit_run,
    complete_run,
    delete_session,
    ensure_session,
    fail_run,
    finish_run,
    get_run,
    interrupt_run,
    interrupt_running_runs,
    list_events,
    load_context,
    load_messages,
    replace_context,
    setup_conversation_store,
)


class PostgresConversationStore:
    """保留现有函数 API 的 PostgreSQL 对象适配器。"""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = _postgres_connection_url(database_url)

    def setup(self) -> None:
        setup_conversation_store(self.database_url)

    def close(self) -> None:
        """PostgreSQL 实现按调用打开连接，无常驻资源。"""

    def ensure_session(self, session_id: str) -> None:
        ensure_session(session_id, database_url=self.database_url)

    def get_run(self, run_id: str) -> RunRecord | None:
        return get_run(run_id, database_url=self.database_url)

    def delete_session(self, session_id: str) -> bool:
        return delete_session(session_id, database_url=self.database_url)

    def interrupt_running_runs(
        self,
        *,
        reason: str = "process restarted",
    ) -> int:
        return interrupt_running_runs(
            reason=reason,
            database_url=self.database_url,
        )

    def begin_run(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
    ) -> RunRecord:
        return begin_run(
            session_id,
            run_id=run_id,
            database_url=self.database_url,
        )

    def append_event(
        self,
        run_id: str,
        *,
        message_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> EventRecord:
        return append_event(
            run_id,
            message_id=message_id,
            event_type=event_type,
            payload=payload,
            database_url=self.database_url,
        )

    def append_message(
        self,
        run_id: str,
        message: BaseMessage,
        *,
        message_id: str | None = None,
    ) -> EventRecord:
        return append_message(
            run_id,
            message,
            message_id=message_id,
            database_url=self.database_url,
        )

    def list_events(
        self,
        session_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[EventRecord]:
        return list_events(
            session_id,
            statuses=statuses,
            database_url=self.database_url,
        )

    def load_messages(
        self,
        session_id: str,
        *,
        statuses: Iterable[str] = ("completed",),
    ) -> list[BaseMessage]:
        return load_messages(
            session_id,
            statuses=statuses,
            database_url=self.database_url,
        )

    def load_context(
        self,
        session_id: str,
    ) -> ContextRecord | None:
        return load_context(
            session_id,
            database_url=self.database_url,
        )

    def commit_run(
        self,
        run_id: str,
        *,
        projected_messages: Iterable[BaseMessage],
        compression_session: dict[str, Any],
    ) -> RunRecord:
        return commit_run(
            run_id,
            projected_messages=projected_messages,
            compression_session=compression_session,
            database_url=self.database_url,
        )

    def replace_context(
        self,
        session_id: str,
        *,
        projected_messages: Iterable[BaseMessage],
        compression_session: dict[str, Any],
    ) -> ContextRecord:
        return replace_context(
            session_id,
            projected_messages=projected_messages,
            compression_session=compression_session,
            database_url=self.database_url,
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: FinalRunStatus,
        error: str | None = None,
    ) -> RunRecord:
        return finish_run(
            run_id,
            status=status,
            error=error,
            database_url=self.database_url,
        )

    def complete_run(self, run_id: str) -> RunRecord:
        return complete_run(
            run_id,
            database_url=self.database_url,
        )

    def cancel_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
    ) -> RunRecord:
        return cancel_run(
            run_id,
            reason=reason,
            database_url=self.database_url,
        )

    def cancel_run_with_context(
        self,
        run_id: str,
        *,
        projected_messages: Iterable[BaseMessage],
        compression_session: dict[str, Any],
        reason: str | None = None,
    ) -> RunRecord:
        return cancel_run_with_context(
            run_id,
            projected_messages=projected_messages,
            compression_session=compression_session,
            reason=reason,
            database_url=self.database_url,
        )

    def fail_run(
        self,
        run_id: str,
        *,
        error: str,
    ) -> RunRecord:
        return fail_run(
            run_id,
            error=error,
            database_url=self.database_url,
        )

    def interrupt_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
    ) -> RunRecord:
        return interrupt_run(
            run_id,
            reason=reason,
            database_url=self.database_url,
        )

