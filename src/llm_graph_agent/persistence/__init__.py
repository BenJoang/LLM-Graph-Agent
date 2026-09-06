"""持久化层：对话日志存储。

分层结构（对标 opencode）：
- models.py      数据模型（类型/异常/记录）
- codec.py       消息编解码（零依赖）
- checkpoints.py LangGraph checkpoint 兼容
- protocol.py    存储抽象接口
- postgres_store.py  Postgres 实现
- sqlite_store.py   SQLite 实现
- database.py    后端选择 + 工厂
"""

from llm_graph_agent.persistence.codec import (
    MessageCodecError,
    UnsupportedMessageTypeError,
    decode_json_value,
    deserialize_message,
    encode_json_value,
    event_type_for_message,
    serialize_message,
)
from llm_graph_agent.persistence.models import (
    ActiveRunError,
    ContextRecord,
    EventConflictError,
    EventRecord,
    EventType,
    FinalRunStatus,
    RunNotFoundError,
    RunNotWritableError,
    RunRecord,
    RunStateConflictError,
)
from llm_graph_agent.persistence.protocol import ConversationStore
from llm_graph_agent.persistence.database import (
    create_conversation_store,
    conversation_database_url,
)
from llm_graph_agent.persistence.sqlite_store import SQLiteConversationStore
from llm_graph_agent.persistence.postgres_store import (
    PostgresConversationStore,
    cancel_run_with_context,
)
from llm_graph_agent.persistence.checkpoints import (
    checkpoint_backend,
    delete_checkpoint_thread,
    open_async_checkpointer,
    open_checkpointer,
    postgres_url,
    setup_checkpoint_backend,
    sqlite_path,
)

__all__ = [
    # codec
    "MessageCodecError",
    "UnsupportedMessageTypeError",
    "encode_json_value",
    "decode_json_value",
    "event_type_for_message",
    "serialize_message",
    "deserialize_message",
    # models
    "ActiveRunError",
    "RunNotFoundError",
    "RunNotWritableError",
    "EventConflictError",
    "RunStateConflictError",
    "RunRecord",
    "EventRecord",
    "ContextRecord",
    "EventType",
    "FinalRunStatus",
    # protocol + 实现
    "ConversationStore",
    "SQLiteConversationStore",
    "PostgresConversationStore",
    "cancel_run_with_context",
    # database
    "create_conversation_store",
    "conversation_database_url",
    # checkpoints
    "checkpoint_backend",
    "delete_checkpoint_thread",
    "open_async_checkpointer",
    "open_checkpointer",
    "postgres_url",
    "setup_checkpoint_backend",
    "sqlite_path",
]
