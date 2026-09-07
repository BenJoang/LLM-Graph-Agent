"""上下文压缩模块。

对外暴露 MessageManage（压缩引擎）和会话/裁剪/折叠子模块。
"""
from llm_graph_agent.context.compression.engine import MessageManage
from llm_graph_agent.context.compression.session import (
    CompressionSession,
    dump_compression_session,
    load_compression_session,
    make_empty_compression_session,
)
from llm_graph_agent.context.compression.retry_adapter import CompressionRetryAdapter

__all__ = [
    "CompressionRetryAdapter",
    "CompressionSession",
    "MessageManage",
    "dump_compression_session",
    "load_compression_session",
    "make_empty_compression_session",
]
