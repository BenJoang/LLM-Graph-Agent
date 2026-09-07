"""上下文压缩模块。

对外暴露 MessageManage（压缩引擎，全 async 通道）和纯逻辑子模块：
- overflow:  阈值判断                 serialize: 消息 → 摘要文本
- collapse:  折叠规划/投影/构造         snip:      工具输出裁剪
- tokens:    token 估算               session:   压缩会话存取
- engine:    编排                      retry_adapter: 重试压缩适配
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
