"""运行时工厂：创建 AgentRunner。

业务方可用自定义 runner_builder 覆盖默认实现（测试时注入假图）。
"""
from __future__ import annotations

from typing import Callable

from llm_graph_agent.api import build_tool_agent_graph
from llm_graph_agent.runner import AgentRunner

AgentRunnerFactory = Callable[[], AgentRunner]

_default_runner: AgentRunner | None = None


def make_default_runner() -> AgentRunner:
    """默认运行时：tool_agent 图 + SQLite 存储。"""
    return AgentRunner(
        graph_builder=build_tool_agent_graph,
    )


def make_runner_factory() -> AgentRunnerFactory:
    """返回惰性单例工厂，业务方可用 create_app 覆盖。"""
    global _default_runner

    def _factory() -> AgentRunner:
        global _default_runner
        if _default_runner is None:
            _default_runner = make_default_runner()
        return _default_runner

    return _factory
