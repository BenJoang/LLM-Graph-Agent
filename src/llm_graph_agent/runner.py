"""Agent 图运行协调器：把对话轮次跑进持久化存储，并输出节点级流式更新。

与原项目 services/tool_agent_runner 的唯一职责差异：
- 去掉 GUI/特定 graph 的耦合
- 图入口从"字符串 entrypoint 反射"改为"直接注入 builder 可调用对象"，更直白

每轮 execute：
    生成 Run → 写 user 消息 → 用注入的 builder 构建图 → 流式跑图，
    同时把每个节点产出的新消息 append 到 SQL → 最后 commit 压缩上下文
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from llm_graph_agent.context.compression import MessageManage
from llm_graph_agent.context.compression.session import (
    make_empty_compression_session,
)
from llm_graph_agent.context.messages import set_context_state
from llm_graph_agent.persistence import (
    ConversationStore,
    RunStateConflictError,
    create_conversation_store,
)

logger = logging.getLogger(__name__)

GraphUpdate = dict[str, Any]
UpdateCallback = Callable[[GraphUpdate], Awaitable[None] | None]
GraphStream = Callable[..., AsyncIterator[GraphUpdate]]
GraphBuilder = Callable[..., Any]


def _node_outputs(update: GraphUpdate):
    """展开 stream_mode="updates" 的节点 → 输出 dict。"""
    for output in update.values():
        if isinstance(output, dict):
            yield output


def _messages_from_update(update: GraphUpdate):
    """从节点更新里抽出消息（单个或列表，跳过非消息）。"""
    for output in _node_outputs(update):
        messages = output.get("messages") or []
        if isinstance(messages, BaseMessage):
            yield messages
        elif isinstance(messages, (list, tuple)):
            for message in messages:
                if isinstance(message, BaseMessage):
                    yield message


def _merge_update(state: dict[str, Any], update: GraphUpdate) -> None:
    """把节点更新合并进累计 state，messages 走 add_messages。"""
    for output in _node_outputs(update):
        for key, value in output.items():
            if key == "messages":
                state["messages"] = add_messages(
                    state.get("messages", []),
                    value or [],
                )
            else:
                state[key] = value


class AgentRunner:
    """把任何 build_graph(**params) 可调用对象（接受 profile/working_dir 等）
    跑进持久化对话存储。"""

    def __init__(
        self,
        *,
        graph_builder: GraphBuilder,
        profile_name: str = "qwen3.6",
        vision_profile_name: str = "qwen3-vl",
        recursion_limit: int = 200,
        working_dir: str | None = None,
        context_window_tokens: int = 32768,
        store: ConversationStore | None = None,
    ) -> None:
        self._graph_builder = graph_builder
        self._profile_name = profile_name
        self._vision_profile_name = vision_profile_name
        self._recursion_limit = recursion_limit
        self._working_dir = working_dir
        self._context_window_tokens = context_window_tokens

        self.store = store or create_conversation_store()
        self._setup_lock = asyncio.Lock()
        self._is_setup = False

    async def setup(self) -> None:
        """建表 + 把遗留的 running Run 标成 interrupted（幂等）。"""
        if self._is_setup:
            return
        async with self._setup_lock:
            if self._is_setup:
                return
            await asyncio.to_thread(self.store.setup)
            recovered = await asyncio.to_thread(
                self.store.interrupt_running_runs,
                reason="process restarted",
            )
            if recovered:
                logger.warning(
                    "启动时将 %s 个遗留 running Run 标记为 interrupted",
                    recovered,
                )
            self._is_setup = True

    async def close(self) -> None:
        await asyncio.to_thread(self.store.close)
        self._is_setup = False

    async def _finish_cancelled(self, run_id: str) -> None:
        try:
            await asyncio.to_thread(
                self.store.cancel_run,
                run_id,
                reason="user cancelled",
            )
        except RunStateConflictError:
            logger.info("取消到达时 Run 已进入终态：%s", run_id)
        except Exception:
            logger.exception("标记 cancelled Run 失败：%s", run_id)

    async def _finish_failed(
        self,
        run_id: str,
        error: BaseException,
    ) -> None:
        error_text = f"{type(error).__name__}: {error}"[:4000]
        try:
            await asyncio.to_thread(
                self.store.fail_run,
                run_id,
                error=error_text,
            )
        except RunStateConflictError:
            logger.info("失败到达时 Run 已进入终态：%s", run_id)
        except Exception:
            logger.exception("标记 failed Run 失败：%s", run_id)

    async def execute(
        self,
        *,
        question: str,
        session_id: str,
        run_id: str | None = None,
        on_update: UpdateCallback | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """执行一轮；节点消息写入 SQL 后才通知流式调用方。"""
        await self.setup()
        requested_run_id = run_id or f"run-{uuid4().hex}"
        begin_task = asyncio.create_task(
            asyncio.to_thread(
                self.store.begin_run,
                session_id,
                run_id=requested_run_id,
            )
        )
        try:
            run = await asyncio.shield(begin_task)
        except asyncio.CancelledError as cancelled:
            try:
                run = await asyncio.shield(begin_task)
            except Exception:
                raise cancelled
            await asyncio.shield(self._finish_cancelled(run.run_id))
            raise

        async def execute_started_run() -> dict[str, Any]:
            context = await asyncio.to_thread(
                self.store.load_context,
                session_id,
            )
            if context is None:
                history = await asyncio.to_thread(
                    self.store.load_messages,
                    session_id,
                )
                compression_session: dict[str, Any] = (
                    make_empty_compression_session()
                )
            else:
                history = list(context.projected_messages)
                compression_session = dict(context.compression_session)

            user_message = HumanMessage(
                content=question,
                id=f"{run.run_id}:user",
            )
            set_context_state(user_message, turn_id=run.turn_no)
            user_event = await asyncio.to_thread(
                self.store.append_message,
                run.run_id,
                user_message,
            )
            persisted_ids = {
                str(message.id)
                for message in history
                if getattr(message, "id", None)
            }
            persisted_ids.add(str(user_message.id))

            state: dict[str, Any] = {
                "messages": [*history, user_message],
                "turn_id": run.turn_no,
                "compression_session": compression_session,
            }
            initial_state = {
                **state,
                "messages": list(state["messages"]),
            }
            next_ordinal = user_event.ordinal + 1

            graph = self._graph_builder(
                profile_name=self._profile_name,
                vision_profile_name=self._vision_profile_name,
                working_dir=self._working_dir,
                checkpointer=None,
                context_window_tokens=self._context_window_tokens,
            )

            async for update in graph.astream(
                initial_state,
                config={"recursion_limit": self._recursion_limit},
                stream_mode="updates",
            ):
                for message in _messages_from_update(update):
                    if message.id is None:
                        message.id = f"{run.run_id}:message:{next_ordinal}"
                    message_id = str(message.id)
                    if message_id in persisted_ids:
                        continue
                    event = await asyncio.to_thread(
                        self.store.append_message,
                        run.run_id,
                        message,
                    )
                    persisted_ids.add(message_id)
                    next_ordinal = max(next_ordinal, event.ordinal + 1)

                _merge_update(state, update)
                if on_update is not None:
                    callback_result = on_update(update)
                    if inspect.isawaitable(callback_result):
                        await callback_result

            # 压缩上下文投影 → 提交
            manager = MessageManage(max_tokens=self._context_window_tokens)
            projected, committed_compression = (
                manager.project_committed_context(
                    list(state.get("messages", [])),
                    state.get("compression_session"),
                )
            )
            state["compression_session"] = committed_compression
            commit_task = asyncio.create_task(
                asyncio.to_thread(
                    self.store.commit_run,
                    run.run_id,
                    projected_messages=projected,
                    compression_session=committed_compression,
                )
            )
            try:
                await asyncio.shield(commit_task)
            except asyncio.CancelledError:
                # completed/context 是同一事务的提交边界；事务已经启动后，
                # 让提交结果获胜，调用方也应看到 completed。
                await asyncio.shield(commit_task)
            return state

        try:
            if timeout_seconds is None:
                return await execute_started_run()
            return await asyncio.wait_for(
                execute_started_run(),
                timeout=timeout_seconds,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._finish_cancelled(run.run_id))
            raise
        except Exception as error:
            await self._finish_failed(run.run_id, error)
            raise

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """一次执行（等最终结果）。"""
        return await self.execute(**kwargs)

    async def astream(self, **kwargs: Any) -> AsyncIterator[GraphUpdate]:
        """流式执行：产出每个节点的更新，结束前不终止。"""
        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()

        async def on_update(update: GraphUpdate) -> None:
            await queue.put(update)

        async def worker() -> dict[str, Any]:
            try:
                return await self.execute(**kwargs, on_update=on_update)
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item  # type: ignore[misc]
            await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
