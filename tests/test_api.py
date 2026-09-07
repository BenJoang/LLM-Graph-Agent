"""API 路由测试：用假 runner factory 验证 JSON 与流式接口。"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from llm_graph_agent.api.app import create_app


def _make_fake_runner():
    class FakeRunner:
        def __init__(self):
            self.close_called = False

        async def run(self, **kwargs):
            return {
                "messages": [AIMessage(content="最终回答", id="final-1")],
                "turn_id": 1,
            }

        async def execute(self, **kwargs):
            on_update = kwargs.get("on_update")
            await on_update(
                {"assistant": {"messages": [AIMessage(content="中间步", id="m-1")]}}
            )
            await on_update(
                {"tools": {"messages": [AIMessage(content="工具结果", id="t-1")]}}
            )
            return {
                "messages": [AIMessage(content="流式回答", id="final-2")],
                "turn_id": 1,
            }

        async def close(self):
            self.close_called = True

    runner = FakeRunner()
    return runner, lambda: runner


def test_health_endpoint():
    _, factory = _make_fake_runner()
    app = create_app(runner_factory=factory)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_tool_agent_returns_answer():
    runner, factory = _make_fake_runner()
    app = create_app(runner_factory=factory)
    client = TestClient(app)

    response = client.post(
        "/agent/tool",
        json={"question": "你好", "session_id": "sess-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["answer"] == "最终回答"
    assert body["session_id"] == "sess-1"


def test_tool_agent_stream_emits_updates_and_done():
    runner, factory = _make_fake_runner()
    app = create_app(runner_factory=factory)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/agent/turn",
        json={"question": "流式测试", "session_id": "sess-2"},
    ) as response:
        assert response.status_code == 200
        events = list(response.iter_lines())

    # 事件行（TestClient 返回 str）
    data_lines = [e for e in events if isinstance(e, str) and e.startswith("data: ")]
    assert len(data_lines) == 3  # 两个节点更新 + done

    last = data_lines[-1]
    assert '"done": true' in last
    assert "流式回答" in last


def test_tool_agent_validates_input():
    _, factory = _make_fake_runner()
    app = create_app(runner_factory=factory)
    client = TestClient(app)

    response = client.post(
        "/agent/tool",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_app_rejects_missing_factory():
    # runner_factory 为 None 时用默认（不测试真实，只验证可创建 app）
    app = create_app(runner_factory=None)
    assert app.state.runner_factory is not None
