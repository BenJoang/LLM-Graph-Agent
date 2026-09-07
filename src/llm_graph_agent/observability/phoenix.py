"""Phoenix (Arize) 链路追踪集成。

通过环境变量控制是否启用：
- PHOENIX_TRACING_ENABLED=1 时启用
- PHOENIX_COLLECTOR_ENDPOINT 指定 collector 地址（默认 http://127.0.0.1:6006）
- PHOENIX_PROJECT_NAME 指定项目名（默认 llm-graph-agent）

这是全项目唯一的 tracing 入口，幂等可重复调用。
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from llm_graph_agent.paths import PROJECT_ROOT


_INITIALIZED = False


def _is_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def setup_phoenix_tracing() -> bool:
    """Enable Phoenix tracing when explicitly requested in the environment."""
    global _INITIALIZED

    if _INITIALIZED:
        return True

    load_dotenv(PROJECT_ROOT / ".env")

    if not _is_enabled(os.getenv("PHOENIX_TRACING_ENABLED")):
        return False

    try:
        from phoenix.otel import register
    except ImportError as error:
        raise RuntimeError(
            "Phoenix tracing 已启用，但缺少依赖；请安装 observability extra。"
        ) from error

    project_name = os.getenv("PHOENIX_PROJECT_NAME", "llm-graph-agent").strip()
    register(
        project_name=project_name or "llm-graph-agent",
        # Keep batch uploads off gRPC: large graph states can exceed its
        # collector request limit and cause an entire batch of spans to be lost.
        # Phoenix derives /v1/traces from PHOENIX_COLLECTOR_ENDPOINT for HTTP.
        protocol="http/protobuf",
        auto_instrument=True,
        batch=True,
        verbose=False,
    )

    _INITIALIZED = True
    logging.getLogger(__name__).info(
        "Phoenix tracing 已启用：project=%s collector=%s protocol=http/protobuf",
        project_name or "llm-graph-agent",
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006"),
    )
    return True
