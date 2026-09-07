"""Agent 运行时 JSON + 流式接口。

依赖注入：路由通过 app.state.runner_factory 取 runner，测试可替换 factory。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from llm_graph_agent.api.dto import update_to_jsonable

router = APIRouter(prefix="/agent", tags=["Agent"])


class ToolAgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(
        default_factory=lambda: f"session-{uuid4().hex}",
        description="长期会话 ID；同一 session 保留历史与压缩上下文",
    )
    profile_name: str = Field(
        default="qwen3.6",
        description="模型 profile 名（见 config/user_config.json）",
    )
    vision_profile_name: str = Field(
        default="qwen3-vl",
        description="识图模型 profile 名",
    )
    recursion_limit: int = Field(default=200, ge=1, le=500)
    working_dir: str | None = Field(
        default=None,
        description="代理工作目录；不传用配置默认",
    )
    context_window_tokens: int = Field(
        default=32768,
        ge=1024,
        le=200000,
    )


def _get_runner(request: Request) -> Any:
    """从 app.state 取 runner（测试可注入 factory）。"""
    factory = request.app.state.runner_factory
    return factory()


def extract_answer(result: dict) -> str:
    """从 execute 返回的 state 里取最后一条消息内容。"""
    messages = result.get("messages") or []
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(getattr(last, "content", ""))
    return ""


def _runner_kwargs(request: ToolAgentRequest) -> dict:
    return {
        "question": request.question,
        "session_id": request.session_id,
        "profile_name": request.profile_name,
        "vision_profile_name": request.vision_profile_name,
        "recursion_limit": request.recursion_limit,
        "working_dir": request.working_dir,
        "context_window_tokens": request.context_window_tokens,
    }


@router.post("/tool")
async def tool_agent(request: ToolAgentRequest, req: Request) -> dict:
    """跑一轮工具 Agent，返回最终回答。"""
    runner = _get_runner(req)

    result = await runner.run(**_runner_kwargs(request))

    return {
        "ok": True,
        "session_id": request.session_id,
        "answer": extract_answer(result),
    }


@router.post("/turn")
async def tool_agent_stream(request: ToolAgentRequest, req: Request):
    """流式执行：以 SSE 事件逐节点输出更新，结束时给出最终回答。

    事件格式：
      data: {"node": "assistant", "messages": [...]}   节点更新
      data: {"done": true, "answer": "..."}            结束

    用 execute(on_update=...) 的队列适配，避免重复执行一轮。
    """
    from fastapi.responses import StreamingResponse

    runner = _get_runner(req)

    async def event_stream():
        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()

        async def on_update(update: dict) -> None:
            await queue.put(update)

        async def worker():
            try:
                return await runner.execute(
                    **_runner_kwargs(request),
                    on_update=on_update,
                )
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                payload = json.dumps(
                    update_to_jsonable(item),
                    ensure_ascii=False,
                    default=str,
                )
                yield f"data: {payload}\n\n"
            final = await task
            done = json.dumps(
                {
                    "done": True,
                    "answer": extract_answer(final),
                },
                ensure_ascii=False,
            )
            yield f"data: {done}\n\n"
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
