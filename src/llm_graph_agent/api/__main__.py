"""uvicorn 启动入口：python -m llm_graph_agent.api

环境变量：
- LLM_GRAPH_AGENT_PORT  监听端口（默认 8200）
- LLM_GRAPH_AGENT_HOST  监听地址（默认 127.0.0.1）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    port = int(os.environ.get("LLM_GRAPH_AGENT_PORT", "8200"))
    host = os.environ.get("LLM_GRAPH_AGENT_HOST", "127.0.0.1")

    uvicorn.run(
        "llm_graph_agent.api.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
