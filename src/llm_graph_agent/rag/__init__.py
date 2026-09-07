"""RAG：知识库切片 → embedding → 召回（混合检索）。

分层：
- chunking         纯文本切片
- embedding_client OpenAI 兼容 embedding 客户端
- postgres_store   pgvector 存储
- service          入库/召回编排
- hybrid_retriever 向量 + BM25 混合检索
- metrics          RAG 评估指标（hit@k / MRR）
"""
from llm_graph_agent.rag.chunking import TextChunk, split_document
from llm_graph_agent.rag.embedding_client import (
    AsyncEmbeddingClient,
    EmbeddingClient,
    EmbeddingSettings,
)
from llm_graph_agent.rag.metrics import hit_at_k, load_cases, reciprocal_rank
from llm_graph_agent.rag.postgres_store import (
    AsyncPostgresRagStore,
    PostgresRagStore,
    RagSearchResult,
    RagStoredChunk,
)
from llm_graph_agent.rag.service import (
    ingest_file,
    retrieve_chunks,
)
from llm_graph_agent.rag.hybrid_retriever import (
    HybridSearchResult,
    hybrid_retrieve,
    tokenize,
)

__all__ = [
    # chunking
    "TextChunk",
    "split_document",
    # embedding
    "EmbeddingSettings",
    "EmbeddingClient",
    "AsyncEmbeddingClient",
    # postgres
    "PostgresRagStore",
    "AsyncPostgresRagStore",
    "RagSearchResult",
    "RagStoredChunk",
    # service
    "ingest_file",
    "retrieve_chunks",
    # hybrid
    "HybridSearchResult",
    "hybrid_retrieve",
    "tokenize",
    # metrics
    "hit_at_k",
    "load_cases",
    "reciprocal_rank",
]
