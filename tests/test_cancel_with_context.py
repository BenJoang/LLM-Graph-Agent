"""取消并固化上下文投影的测试。

对应 cancel_run_with_context 的改动：
- 取消后 run 状态为 cancelled
- 已生成消息的上下文投影被固化（load_context 可读）
- 原始消息仍可读（显式 statuses 包含 cancelled）
- 取消是幂等的（重复取消不报错）
- 只有 running 才能取消（completed 后取消报错）
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from llm_graph_agent.persistence import (
    RunStateConflictError,
    conversation_database_url,
    create_conversation_store,
)


@pytest.fixture(params=("sqlite", "postgres"))
def cancel_store(request, tmp_path: Path):
    """与 test_persistence 相同的双后端 fixture。"""
    backend = request.param
    if backend == "sqlite":
        database_path = tmp_path / "cancel.sqlite3"
        database_url = f"sqlite:///{database_path.as_posix()}"
    else:
        if os.getenv("LLM_GRAPH_TEST_POSTGRES") != "1":
            pytest.skip(
                "设置 LLM_GRAPH_TEST_POSTGRES=1 后运行 PostgreSQL 契约测试"
            )
        database_url = conversation_database_url()

    store = create_conversation_store(database_url)
    store.setup()
    session_id = f"cancel-with-context-test-{uuid4().hex}"

    yield backend, store, session_id

    if backend == "postgres":
        with psycopg.connect(database_url, autocommit=True) as connection:
            connection.execute(
                """
                DELETE FROM conversation_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )


def _projected_messages():
    return [
        HumanMessage(content="你好", id="cancel-user-1"),
        AIMessage(content="我已经生成了一部分", id="cancel-ai-1"),
    ]


def test_cancel_with_context_marks_cancelled_and_commits_context(cancel_store):
    """取消后 run 是 cancelled，且上下文投影被固化。"""
    _, store, session_id = cancel_store
    run = store.begin_run(session_id, run_id=f"run-{uuid4().hex}")

    # 先写入原始消息
    store.append_message(
        run.run_id,
        HumanMessage(content="你好", id="cancel-user-1"),
    )
    store.append_message(
        run.run_id,
        AIMessage(content="我已经生成了一部分", id="cancel-ai-1"),
    )

    # 取消并固化
    result = store.cancel_run_with_context(
        run.run_id,
        projected_messages=_projected_messages(),
        compression_session={},
        reason="user cancelled",
    )

    assert result.status == "cancelled"
    assert result.finished_at is not None
    assert result.error == "user cancelled"

    # 投影被固化：load_context 能读到
    context = store.load_context(session_id)
    assert context is not None
    assert [m.id for m in context.projected_messages] == [
        "cancel-user-1",
        "cancel-ai-1",
    ]


def test_cancel_with_context_messages_readable(cancel_store):
    """取消轮的消息通过显式 statuses 能读回原始消息。"""
    _, store, session_id = cancel_store
    run = store.begin_run(session_id, run_id=f"run-{uuid4().hex}")
    store.append_message(
        run.run_id,
        HumanMessage(content="问题", id="cancel-user-1"),
    )
    store.append_message(
        run.run_id,
        AIMessage(content="部分回答", id="cancel-ai-1"),
    )
    store.cancel_run_with_context(
        run.run_id,
        projected_messages=_projected_messages(),
        compression_session={},
        reason="user cancelled",
    )

    # 显式包含 cancelled 才能读到
    messages = store.load_messages(
        session_id,
        statuses=("completed", "cancelled"),
    )
    assert len(messages) == 2
    assert [m.id for m in messages] == ["cancel-user-1", "cancel-ai-1"]

    # 默认 statuses 仍只看 completed（取消轮不进模型上下文）
    assert store.load_messages(session_id) == []


def test_cancel_with_context_is_idempotent(cancel_store):
    """重复取消同一 run 不报错，返回相同状态。"""
    _, store, session_id = cancel_store
    run = store.begin_run(session_id, run_id=f"run-{uuid4().hex}")

    first = store.cancel_run_with_context(
        run.run_id,
        projected_messages=_projected_messages(),
        compression_session={},
        reason="first",
    )
    second = store.cancel_run_with_context(
        run.run_id,
        projected_messages=_projected_messages(),
        compression_session={},
        reason="second",
    )

    assert first.status == "cancelled"
    assert second.status == "cancelled"


def test_cancel_with_context_rejects_completed_run(cancel_store):
    """已完成（completed）的 run 不能再用取消固化，应报冲突。"""
    _, store, session_id = cancel_store
    run = store.begin_run(session_id, run_id=f"run-{uuid4().hex}")
    store.complete_run(run.run_id)

    with pytest.raises(RunStateConflictError):
        store.cancel_run_with_context(
            run.run_id,
            projected_messages=_projected_messages(),
            compression_session={},
            reason="too late",
        )
