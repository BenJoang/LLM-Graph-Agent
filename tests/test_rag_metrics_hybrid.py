"""RAG 评估指标 + hybrid 检索纯逻辑测试。"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from llm_graph_agent.rag.metrics import hit_at_k, load_cases, reciprocal_rank
from llm_graph_agent.rag.hybrid_retriever import bm25_search, tokenize


@dataclass(frozen=True)
class FakeResult:
    document_id: str


def test_retrieval_metrics() -> None:
    results = [
        FakeResult("wrong"),
        FakeResult("expected"),
        FakeResult("other"),
    ]
    expected = {"expected"}

    assert hit_at_k(results, expected, 1) is False
    assert hit_at_k(results, expected, 2) is True
    assert reciprocal_rank(results, expected) == 0.5


def test_metrics_return_miss_for_absent_document() -> None:
    results = [FakeResult("wrong")]
    expected = {"expected"}

    assert hit_at_k(results, expected, 3) is False
    assert reciprocal_rank(results, expected) == 0.0


def test_hit_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k 必须大于 0"):
        hit_at_k([], {"expected"}, 0)


def test_load_cases(tmp_path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "tenant_id": "study",
                "cases": [
                    {
                        "question": "测试问题",
                        "expected_document_ids": ["document-1"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tenant_id, cases = load_cases(cases_path)

    assert tenant_id == "study"
    assert cases[0]["question"] == "测试问题"


def test_load_cases_rejects_empty_cases(tmp_path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps({"tenant_id": "study", "cases": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="非空 cases"):
        load_cases(cases_path)


def test_tokenize_filters_punctuation_and_whitespace():
    tokens = tokenize("  你好，World! 测试  ")
    # 分词后都是有效内容 token
    assert tokens
    assert all(token.strip() for token in tokens)
    assert all(any(c.isalnum() for c in token) for token in tokens)


def test_bm25_search_ranks_relevant_docs_first():
    from llm_graph_agent.rag.postgres_store import RagStoredChunk

    # rank_bm25 的 idf 在小语料下会变 0（df ≈ N 时），
    # 用足够大的语料让 idf > 0，聚焦 BM25 排序本身。
    chunks = [
        RagStoredChunk(
            chunk_id=f"c{i}", document_id=f"d{i}", source=f"{i}.txt",
            section=None,
            content=content,
            metadata={},
        )
        for i, content in enumerate([
            "machine learning model architecture",
            "sunny weather today perfect for travel",
            "quantum physics and particle experiments",
            "basketball game statistics and scores",
        ])
    ]

    results = bm25_search("machine learning model", chunks, top_k=2)

    assert results
    assert results[0][0].chunk_id == "c0"
    assert results[0][1] > 0


def test_bm25_empty_corpus_returns_empty():
    assert bm25_search("query", [], top_k=5) == []
