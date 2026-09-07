"""AgentRunner 运行协调器测试。

用 sqlite 临时库 + 假 graph builder 完整走通：
- execute 的持久化链路（user 消息写入、节点消息去重、压缩提交）
- 取消/失败的状态标记（用 spy 验证）
- astream 队列适配

注意：store 方法是同步的，runner 内部用 asyncio.to_thread 包裹；
测试里读取也应走 to_thread（或直接同步调用）。
"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from llm_graph_agent.runner import AgentRunner
from llm_graph_agent.persistence import create_conversation_store


def _make_sqlite_store(tmp_path):
    database_url = f"sqlite:///{(tmp_path / 'runner.sqlite3').as_posix()}"
    return create_conversation_store(database_url)


async def _load_messages(runner, session_id):
    return await asyncio.to_thread(runner.store.load_messages, session_id)


async def _load_context(runner, session_id):
    return await asyncio.to_thread(runner.store.load_context, session_id)


class FakeGraph:
    """产出若干节点更新的假图。"""

    def __init__(self, updates):
        self._updates = updates
        self.stream_calls = 0

    async def astream(self, initial_state, config=None, stream_mode=None):
        self.stream_calls += 1
        for update in self._updates:
            yield update


def _fake_builder(*, messages_to_produce=None):
    updates = messages_to_produce or [
        {"assistant": {"messages": [AIMessage(content="第一步", id="step-1")]}}
    ]
    graph = FakeGraph(updates)

    def _build(**kwargs):
        return graph

    _build.graph = graph
    return _build


async def test_execute_persists_user_and_node_messages(tmp_path):
    builder = _fake_builder()
    store = _make_sqlite_store(tmp_path)
    runner = AgentRunner(graph_builder=builder, store=store)

    result = await runner.run(
        question="你好",
        session_id="sess-1",
    )

    assert any(m.content == "你好" for m in result["messages"])
    assert any(m.content == "第一步" for m in result["messages"])

    persisted = await _load_messages(runner, "sess-1")
    assert len(persisted) >= 2
    assert any(m.content == "你好" for m in persisted)
    assert any(m.content == "第一步" for m in persisted)

    context = await _load_context(runner, "sess-1")
    assert context is not None
    assert context.compression_session["version"] == 1

    await runner.close()


async def test_execute_runs_second_turn_with_history(tmp_path):
    builder = _fake_builder(
        messages_to_produce=[
            {"assistant": {"messages": [AIMessage(content="回答1", id="a-1")]}}
        ]
    )
    store = _make_sqlite_store(tmp_path)
    runner = AgentRunner(graph_builder=builder, store=store)

    first = await runner.run(question="问1", session_id="sess-2")
    assert any(m.content == "回答1" for m in first["messages"])

    second = await runner.run(question="问2", session_id="sess-2")
    content = [m.content for m in second["messages"] if m.content]
    assert "问1" in content
    assert "问2" in content

    await runner.close()


async def test_builder_failure_marks_run_failed(tmp_path):
    def broken_builder(**kwargs):
        raise RuntimeError("builder exploded")

    store = _make_sqlite_store(tmp_path)
    runner = AgentRunner(graph_builder=broken_builder, store=store)

    # store 方法是同步的（runner 内部用 to_thread 包裹），spy 也应是同步函数
    captured = {}

    def fake_fail_run(run_id, error=None):
        captured["run_id"] = run_id
        captured["error"] = error
        return None

    runner.store.fail_run = fake_fail_run

    with pytest.raises(RuntimeError):
        await runner.run(question="会失败", session_id="sess-3")

    assert captured.get("run_id")
    assert "builder exploded" in captured.get("error", "")
    await runner.close()


async def test_astream_yields_node_updates(tmp_path):
    builder = _fake_builder(
        messages_to_produce=[
            {"assistant": {"messages": [AIMessage(content="s1", id="s-1")]}},
            {"tools": {"messages": [AIMessage(content="t1", id="t-1")]}},
        ]
    )
    store = _make_sqlite_store(tmp_path)
    runner = AgentRunner(graph_builder=builder, store=store)

    updates = []
    async for update in runner.astream(
        question="流式",
        session_id="sess-4",
    ):
        updates.append(update)

    assert len(updates) == 2
    assert "assistant" in updates[0] and "tools" in updates[1]

    await runner.close()


async def test_cancelled_execute_marks_run_cancelled(tmp_path):
    builder = _fake_builder()
    store = _make_sqlite_store(tmp_path)
    runner = AgentRunner(graph_builder=builder, store=store)

    captured = {}

    def fake_cancel_run(run_id, reason=None):
        captured["run_id"] = run_id
        captured["reason"] = reason
        return None

    runner.store.cancel_run = fake_cancel_run

    # 用一个永不结束的 graph，确保 execute 会真正挂起等待 astream
    gate = asyncio.Event()

    class HangingGraph:
        async def astream(self, initial_state, config=None, stream_mode=None):
            await gate.wait()  # 永不结束，除非被取消
            yield {"assistant": {"messages": []}}

    def hanging_builder(**kwargs):
        return HangingGraph()

    runner._graph_builder = hanging_builder

    task = asyncio.create_task(
        runner.run(question="会被取消", session_id="sess-5")
    )
    # 等 execute 进入 graph.astream 挂起点
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert captured.get("run_id")
    assert captured.get("reason") == "user cancelled"
    await runner.close()
