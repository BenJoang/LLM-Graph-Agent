"""Agent 图公共编排入口（Phase 5 收敛点）。

原项目里每个 graph（tool_agent / sub_agent / tts / qq_main / wuxi）都重复手写
同一套三件套：
    1) MessageManage（上下文压缩引擎）
    2) CompressionRetryAdapter（重试适配，持有压缩会话状态）
    3) invoke_with_retry（超时退避 + 上下文超限降级压缩重试）

这里的 CompressedAssistant 把这三件套 + 系统提示词组装收敛成"一次调用"：
    response = await assistant.invoke(messages, turn_id, should_finalize=False)

- 无工具时可传入未 bind_tools 的模型（事件轮次功能），
  有工具时传入 bind_tools 后的模型
- 压缩会话在 assistant 内部持久，供下一轮继续用
- 上下文超限时自动逐级降级（merge summary → snip tools → collapse turns）
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from llm_graph_agent.context.builder import build_system_context
from llm_graph_agent.context.compression import (
    CompressionRetryAdapter,
    MessageManage,
)
from llm_graph_agent.context.compression.session import CompressionSession
from llm_graph_agent.context.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def build_system_prompt(
    prompt_system: str,
    *,
    working_dir: str | None = None,
    skill_names: list[str] | None = None,
) -> str:
    """把静态系统提示词与动态上下文（工作目录/skills/项目指令）拼成一个 system。"""
    context_system = build_system_context(
        working_dir=working_dir,
        skill_names=skill_names,
        working_dir_need=True,
        instruction_need=True,
    )

    return "\n\n".join(
        part
        for part in [prompt_system, context_system]
        if part
    )


class CompressedAssistant:
    """一次 agent 轮次的"压缩 + 组装 + 重试调用"编排器。

    收敛了原项目 graphs 里重复的三件套样板：
    MessageManage / CompressionRetryAdapter / invoke_with_retry。
    会话状态（compression_session）在内部持久，跨多轮复用。
    """

    def __init__(
        self,
        *,
        message_manage: MessageManage,
        plane: Runnable,
        tooled: Runnable,
        system_content: str,
        turn_id: int,
        max_context_retries: int = 3,
        compression_session: CompressionSession | None = None,
    ) -> None:
        self._message_manage = message_manage
        self._plane = plane          # 未绑定工具的模型（收官/不调工具时用）
        self._tooled = tooled        # 绑定了工具的模型
        self._system_content = system_content
        self._turn_id = turn_id
        self._max_context_retries = max_context_retries
        self._compression_session = compression_session

    @property
    def compression_session(self) -> CompressionSession | None:
        return self._compression_session

    def reset_session(self) -> None:
        self._compression_session = None

    async def prepare_query_messages(
        self,
        messages: list,
    ) -> tuple[list, CompressionSession | None]:
        """把原始消息压到模型可用的查询消息列表（含折叠摘要，可能调用 LLM）。"""
        (
            messages_for_query,
            _compressed,
            compression_session,
        ) = await self._message_manage.prepare_messages_for_query(
            messages,
            self._compression_session,
        )
        self._compression_session = compression_session
        return messages_for_query, self._compression_session

    async def invoke(
        self,
        messages: list,
        *,
        should_finalize: bool = False,
    ) -> object:
        """带系统提示词 + 重试 + 上下文降级压缩地调用模型。

        should_finalize=True 时使用未绑定工具的模型（强制结束工具循环）。
        """
        messages_for_query, compression_session = (
            await self.prepare_query_messages(messages)
        )

        query_messages = [
            {
                "role": "system",
                "content": self._system_content,
            },
            *messages_for_query,
        ]

        retry_adapter = CompressionRetryAdapter(
            message_manage=self._message_manage,
            compression_session=compression_session,
            current_turn_id=self._turn_id,
        )

        try:
            return await invoke_with_retry(
                invoke_fn=self._plane.ainvoke if should_finalize else self._tooled.ainvoke,
                messages=query_messages,
                original_messages=query_messages,
                compress_fn=retry_adapter.acall,
                turn_id=self._turn_id,
                max_context_retries=self._max_context_retries,
            )
        finally:
            self._compression_session = retry_adapter.compression_session


def build_summarize_pass(
    *,
    llm: Runnable,
    collapse_prompt: str,
) -> Callable[[str], Awaitable[str]]:
    """构建压缩引擎的摘要回调：用主模型把序列化消息压成摘要文本。"""
    async def _summarize(text: str) -> str:
        response = await llm.ainvoke([
            SystemMessage(content=collapse_prompt),
            HumanMessage(content=text),
        ])
        return str(response.content)

    return _summarize


def make_message_manage(
    *,
    llm: Runnable,
    collapse_prompt: str,
    context_window_tokens: int = 32768,
) -> MessageManage:
    """构建带主模型摘要回调的压缩引擎。"""
    return MessageManage(
        max_tokens=context_window_tokens,
        asummarize_fn=build_summarize_pass(
            llm=llm,
            collapse_prompt=collapse_prompt,
        ),
    )
