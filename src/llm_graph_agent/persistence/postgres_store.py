"""PostgreSQL 对话日志实现（兼容转发层）。

真正的实现在 postgres/ 子目录：operations.py（业务函数）+ store.py（适配器类）。
此文件仅 re-export，保持旧 import 路径不变。
"""
from __future__ import annotations

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
from llm_graph_agent.persistence.postgres.store import PostgresConversationStore

__all__ = [
    "PostgresConversationStore",
    "_postgres_connection_url",
    "append_event",
    "append_message",
    "begin_run",
    "cancel_run",
    "cancel_run_with_context",
    "commit_run",
    "complete_run",
    "delete_session",
    "ensure_session",
    "fail_run",
    "finish_run",
    "get_run",
    "interrupt_run",
    "interrupt_running_runs",
    "list_events",
    "load_context",
    "load_messages",
    "replace_context",
    "setup_conversation_store",
]
