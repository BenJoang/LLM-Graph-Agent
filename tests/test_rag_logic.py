"""RAG 纯逻辑测试：chunking + 并发 embedding 编排。"""
from __future__ import annotations

import asyncio

import pytest

from llm_graph_agent.rag.chunking import TextChunk, split_document
from llm_graph_agent.rag.service import _embed_chunks_concurrently


def test_split_document_empty_and_invalid_params():
    assert split_document("", document_id="d") == []
    assert split_document("   \n  ", document_id="d") == []

    with pytest.raises(ValueError):
        split_document("x", document_id="d", chunk_size=0)
    with pytest.raises(ValueError):
        split_document("x", document_id="d", overlap=10, chunk_size=10)


def test_split_document_creates_hashed_ids_and_boundaries():
    text = "第一段。" + "第二段。" * 50
    chunks = split_document(
        text,
        document_id="doc-1",
        chunk_size=30,
        overlap=5,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.chunk_id.startswith("doc-1:")
        assert chunk.content
        assert len(chunk.content_hash) == 64
        assert chunk.end_char >= chunk.start_char


def test_split_document_concatenates_back_to_original():
    text = ("企业知识库内容是用于检索的素材。" * 30)
    chunks = split_document(text, document_id="d", chunk_size=40, overlap=8)
    # 拼接应基本还原正文（因 overlap 会有重复）
    concatenated = "".join(chunk.content for chunk in chunks)
    assert "企业知识库内容" in concatenated
    assert chunks[0].index == 0
    assert chunks[-1].index == len(chunks) - 1


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.active_requests = 0
        self.max_active_requests = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.active_requests += 1
        self.max_active_requests = max(
            self.max_active_requests,
            self.active_requests,
        )
        try:
            await asyncio.sleep(0.02)
            return [
                [float(text.removeprefix("chunk-"))]
                for text in texts
            ]
        finally:
            self.active_requests -= 1


def make_chunks(count: int) -> list[TextChunk]:
    return [
        TextChunk(
            index=index,
            chunk_id=f"document:{index}",
            content=f"chunk-{index}",
            content_hash=f"hash-{index}",
            start_char=index,
            end_char=index + 1,
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_embedding_batches_are_concurrent_and_ordered() -> None:
    client = FakeEmbeddingClient()

    embeddings = await _embed_chunks_concurrently(
        make_chunks(10),
        embedding_client=client,  # type: ignore[arg-type]
        batch_size=2,
        max_concurrency=3,
    )

    assert embeddings == [
        [float(index)]
        for index in range(10)
    ]
    assert client.max_active_requests == 3


@pytest.mark.asyncio
async def test_empty_chunks_do_not_call_embedding() -> None:
    client = FakeEmbeddingClient()

    embeddings = await _embed_chunks_concurrently(
        [],
        embedding_client=client,  # type: ignore[arg-type]
        batch_size=8,
        max_concurrency=3,
    )

    assert embeddings == []
    assert client.max_active_requests == 0


@pytest.mark.asyncio
async def test_embedding_batch_params_validated() -> None:
    client = FakeEmbeddingClient()
    with pytest.raises(ValueError):
        await _embed_chunks_concurrently(
            make_chunks(1), embedding_client=client, batch_size=0, max_concurrency=1
        )
    with pytest.raises(ValueError):
        await _embed_chunks_concurrently(
            make_chunks(1), embedding_client=client, batch_size=1, max_concurrency=0
        )
