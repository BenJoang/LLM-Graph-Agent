"""RAG 评估指标（纯函数）。

公共接口的 objects 只需暴露 document_id 字段即可参与打分；
从 run_retrieval 的结果（RagSearchResult / HybridSearchResult）都能直接使用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class RetrievalResult(Protocol):
    document_id: str


def reciprocal_rank(
    results: list[RetrievalResult],
    expected_document_ids: set[str],
) -> float:
    """第一个命中期望文档的位置的倒数的排名（MRR 单条）。"""
    for rank, result in enumerate(results, start=1):
        if result.document_id in expected_document_ids:
            return 1.0 / rank
    return 0.0


def hit_at_k(
    results: list[RetrievalResult],
    expected_document_ids: set[str],
    k: int,
) -> bool:
    """前 k 条里是否有期望文档命中（Hit@k）。"""
    if k <= 0:
        raise ValueError("k 必须大于 0")

    return any(
        result.document_id in expected_document_ids
        for result in results[:k]
    )


def load_cases(path: Path) -> tuple[str, list[dict]]:
    """读取评估用例文件：{tenant_id, cases: [{question, expected_document_ids}]}。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    tenant_id = str(data.get("tenant_id", "")).strip()
    cases = data.get("cases")

    if not tenant_id:
        raise ValueError("评估文件缺少 tenant_id")

    if not isinstance(cases, list) or not cases:
        raise ValueError("评估文件必须包含非空 cases 列表")

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"第 {index} 个评估用例不是对象")

        question = str(case.get("question", "")).strip()
        expected = case.get("expected_document_ids")

        if not question:
            raise ValueError(f"第 {index} 个评估用例缺少 question")

        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"第 {index} 个评估用例缺少 expected_document_ids"
            )

        case["question"] = question
        case["expected_document_ids"] = list(expected)

    return tenant_id, cases
