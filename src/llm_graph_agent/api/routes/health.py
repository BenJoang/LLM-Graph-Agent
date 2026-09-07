"""健康检查路由。"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "llm-graph-agent"}
