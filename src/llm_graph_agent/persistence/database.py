"""对话数据库 URL 选择与 store 工厂。

原 conversation_store.py 的配置选择 + 工厂部分。
"""
from __future__ import annotations

import os
from pathlib import Path

from llm_graph_agent.persistence.checkpoints import checkpoint_backend, postgres_url
from llm_graph_agent.persistence.models import (
    DEFAULT_CONVERSATION_SQLITE_PATH,
    EventType,  # noqa: F401
    RunRecord,
)
from llm_graph_agent.persistence.protocol import ConversationStore

def conversation_database_url() -> str:
    """返回对话业务数据库 URL。

    显式配置优先；未配置时跟随 checkpoint 后端，但 SQLite 使用
    独立文件，避免业务表和 LangGraph 内部表混在一起。
    """

    configured = os.getenv(
        "LLM_GRAPH_DATABASE_URL",
        "",
    ).strip()

    if configured:
        return configured

    if checkpoint_backend() == "sqlite":
        return (
            "sqlite:///"
            + DEFAULT_CONVERSATION_SQLITE_PATH.as_posix()
        )

    return postgres_url()

def _postgres_connection_url(
    database_url: str | None,
) -> str:
    connection_url = database_url or conversation_database_url()
    scheme = connection_url.partition(":")[0].lower()
    if scheme not in {"postgres", "postgresql"}:
        raise ValueError(
            "PostgreSQL 专用函数需要 postgres/postgresql URL；"
            "需要自动选择后端时请使用 create_conversation_store()"
        )
    return connection_url

def create_conversation_store(
    database_url: str | None = None,
) -> ConversationStore:
    """根据数据库 URL 创建对应的对话日志实现。"""

    connection_url = database_url or conversation_database_url()
    scheme = connection_url.partition(":")[0].lower()

    if scheme == "sqlite":
        from llm_graph_agent.persistence.sqlite_store import (
            SQLiteConversationStore,
        )

        return SQLiteConversationStore(connection_url)
    if scheme in {"postgres", "postgresql"}:
        from llm_graph_agent.persistence.postgres_store import (
            PostgresConversationStore,
        )
        return PostgresConversationStore(connection_url)

    raise ValueError(
        "不支持的 conversation database URL："
        f"{connection_url!r}"
    )
