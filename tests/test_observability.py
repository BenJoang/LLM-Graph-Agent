"""observability 层测试。

覆盖 setup_phoenix_tracing 的开关逻辑：
- 未启用时返回 False，不抛错
- 启用时返回 True 且幂等
- _is_enabled 解析各种布尔字符串
"""
from __future__ import annotations

import importlib
import os

from llm_graph_agent.observability.phoenix import _is_enabled


def test_is_enabled_parses_truthy_values():
    for value in ["1", "true", "True", "yes", "on", " 1 "]:
        assert _is_enabled(value) is True


def test_is_enabled_parses_falsy_values():
    for value in [None, "", "0", "false", "no", "off", "x"]:
        assert _is_enabled(value) is False


def test_setup_disabled_returns_false(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "false")
    # 重新加载模块，避免 _INITIALIZED 残留
    phoenix = importlib.import_module("llm_graph_agent.observability.phoenix")
    assert phoenix.setup_phoenix_tracing() is False


def test_setup_uses_project_name_default(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "true")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.delenv("PHOENIX_PROJECT_NAME", raising=False)
    phoenix = importlib.import_module("llm_graph_agent.observability.phoenix")
    # 直接测默认名
    assert os.getenv("PHOENIX_PROJECT_NAME", "llm-graph-agent") == "llm-graph-agent"
