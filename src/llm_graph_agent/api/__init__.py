"""API 层：干净的 JSON + 流式接口，暴露 Agent 运行时。

去掉了原项目的 GUI/Electron 与会话管理 UI 栈，只保留业务可用的：
- POST /agent/tool   跑一轮工具 Agent，返回最终回答
- POST /agent/turn   流式执行（SSE update 事件）
- GET  /health       存活检查

通过 create_app(runner_builder) 注入运行时，便于测试与业务自定义。
"""
from __future__ import annotations

from llm_graph_agent.graph.tool_agent import build_graph as build_tool_agent_graph
from llm_graph_agent.runner import AgentRunner

__all__ = ["AgentRunner", "build_tool_agent_graph"]
